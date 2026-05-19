import json
import os
import time
import random
import hashlib
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from state import AgentState

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="tenk_corpus",
    embedding_function=embeddings,
    persist_directory="data/chroma_db"
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_cache = {}

def _call_groq(prompt, max_tokens=300):
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
            _cache[key] = result
            return result
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise
    raise Exception("rate_limit: max retries exceeded")


def retrieve(state: AgentState) -> AgentState:
    query = state.get("reformulated_query") or state["query"]
    results = vectorstore.similarity_search(query, k=10)
    state["chunks"] = [doc.page_content for doc in results]
    return state


def generate(state: AgentState) -> AgentState:
    context = "\n\n".join(state["chunks"])
    prompt = f"""You are a financial data extraction assistant for Infosys annual reports.

RULES (follow strictly):
1. Extract the SINGLE most relevant fact or figure that directly answers the question.
2. State it in ONE or TWO sentences maximum. Do NOT show calculations, reasoning, or multi-year tables.
3. If the answer is a number, state it with its unit (crore, %, employees, etc.).
4. If the answer is genuinely absent from the context, output exactly: INSUFFICIENT_CONTEXT
5. Never say "based on the context" or explain your method.

Context:
{context}

Important: If multiple years are present in context, always pick FY2025 (year ended March 31, 2025) only.

Question: {state['query']}
Answer:"""
    state["answer"] = _call_groq(prompt, max_tokens=120)
    return state

def critic(state: AgentState) -> AgentState:
    answer = state["answer"]

    evasion_phrases = [
        "INSUFFICIENT_CONTEXT", "i cannot", "i can't", "cannot deduce",
        "cannot determine", "not mentioned", "not provided", "not available",
        "unfortunately", "does not provide", "no information",
        "unable to", "not stated", "not specified", "context does not",
        "not explicitly", "cannot be determined", "cannot find",
        "no data", "not found", "not present",
    ]
    if any(phrase in answer.lower() for phrase in evasion_phrases):
        state["critic_verdict"] = "fail"
        state["critic_reason"] = "Answer could not find the information — retrying with better query."
        return state

    context = "\n\n".join(state["chunks"])
    prompt = f"""Fact-check this answer against the context. Reply ONLY with valid JSON.
Fail if: answer contains facts not in context, is vague when context has a specific number, or does multi-year math instead of stating FY25 directly.
Pass if: answer is a direct, specific fact supported by context.

{{"verdict": "pass", "reason": "short reason"}} or {{"verdict": "fail", "reason": "short reason"}}

Context: {context[:1000]}
Question: {state['query']}
Answer: {answer}"""

    text = _call_groq(prompt, max_tokens=120)
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        state["critic_verdict"] = parsed.get("verdict", "fail").lower()
        state["critic_reason"] = parsed.get("reason", "no reason")
    except Exception:
        state["critic_verdict"] = "pass" if "pass" in text.lower() else "fail"
        state["critic_reason"] = "could not parse critic response"
    return state


def reformulate(state: AgentState) -> AgentState:
    original = state["query"]
    iteration = state.get("iterations", 0)

    strategies = [
        f"Rewrite as a precise keyword search for an Infosys FY2025 annual report. Use exact financial terms (e.g. 'voluntary attrition rate FY25', 'EBIT margin', 'consolidated revenue'). Original: {original}",
        f"Rephrase using common synonyms found in Indian corporate filings: 'headcount' → 'employee strength', 'attrition' → 'voluntary turnover / annualized attrition', 'revenue' → 'income from operations'. Original: {original}",
        f"Extract only the core metric being asked. Write a 3-5 word search phrase. Original: {original}",
    ]
    prompt = strategies[min(iteration, 2)]
    state["reformulated_query"] = _call_groq(prompt, max_tokens=60)
    state["iterations"] = iteration + 1
    return state