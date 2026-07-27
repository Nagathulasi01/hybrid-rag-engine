import os
import json
import time
import requests
import logging
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Assuming OPENAI_API_KEY is available in environment for RAGAS evaluation models
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

def load_evaluation_dataset() -> list:
    """
    Load a set of golden Q&A pairs for evaluation.
    In a real scenario, this would load from a CSV or JSON file.
    """
    return [
        {
            "question": "What is the main architecture of the proposed Hybrid RAG system?",
            "ground_truth": "The proposed Hybrid RAG system uses a combination of dense and sparse retrieval, followed by Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking, with a FastAPI backend and Qdrant vector database."
        },
        {
            "question": "How are sparse vectors generated in this pipeline?",
            "ground_truth": "Sparse vectors are generated using a SPLADE or BM25 model via the FastEmbed library."
        }
    ]

def collect_predictions(eval_data: list) -> Dataset:
    """
    Sends questions to the RAG API and formats the response for RAGAS.
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    logger.info("Collecting predictions from API...")
    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": question, "top_k": 5, "use_hybrid": True}
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["answer"]
            contexts = [cite["content"] for cite in result["citations"]]
            
            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(ground_truth)
            
            # Small delay to prevent API flooding
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error querying API for question '{question}': {e}", exc_info=True)
            
    return Dataset.from_dict(data)

def run_evaluation():
    logger.info("Starting RAGAS Evaluation...")
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is required for RAGAS evaluation models.")
        logger.warning("Please set the environment variable before running.")
        return
        
    eval_data = load_evaluation_dataset()
    dataset = collect_predictions(eval_data)
    
    if len(dataset) == 0:
        logger.error("No predictions collected. Ensure the API is running and documents are ingested.")
        return
        
    logger.info(f"Evaluating {len(dataset)} examples...")
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        raise_exceptions=False
    )
    
    df = result.to_pandas()
    
    # Calculate means
    summary = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean()
    }
    
    # Save report
    report_path = "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Evaluation Complete. Report saved to {report_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=4)}")

if __name__ == "__main__":
    run_evaluation()
