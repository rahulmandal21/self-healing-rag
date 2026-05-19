import json
import os
import csv
from dotenv import load_dotenv
load_dotenv()

from graph import graph

# Load eval dataset
with open("eval/eval_dataset.json") as f:
    dataset = json.load(f)

def run_pipeline(question):
    result = graph.invoke({"query": question, "iterations": 0})
    return result["answer"], result["critic_verdict"]

results = []

for item in dataset:
    question = item["question"]
    ground_truth = item["ground_truth"]
    
    print(f"\nQ: {question}")
    answer, verdict = run_pipeline(question)
    print(f"A: {answer}")
    print(f"Verdict: {verdict}")
    
    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "critic_verdict": verdict
    })

# Save results
with open("eval/eval_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "answer", "critic_verdict"])
    writer.writeheader()
    writer.writerows(results)

print("\n\nDone! Results saved to eval/eval_results.csv")