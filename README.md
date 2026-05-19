# Self-Healing RAG Pipeline

A stateful multi-node agent graph built with LangGraph that retrieves, generates, critiques, and self-heals answers from the Infosys Annual Report 2024-25.

## Stack
- **LangGraph** — agent graph (retrieve → generate → critic → reformulate)
- **Groq LLaMA 3.1-8b** — sub-second inference
- **ChromaDB** — vector store (1,104 chunks)
- **all-MiniLM-L6-v2** — embeddings
- **Streamlit** — interactive UI with evaluation dashboard

## Features
- LLM-as-judge critic agent detects hallucinations
- Automatic query reformulation (up to 3 iterations)
- 100% critic pass rate on 10-question benchmark

## Live Demo
[self-healing-rag.streamlit.app](https://self-healing-rag.streamlit.app)
