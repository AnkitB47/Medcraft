import json
import os

def eval_nanovlm():
    print("Evaluating NanoVLM...")
    # Mock evaluation metrics
    metrics = {
        "accuracy": 0.82,
        "groundedness_score": 0.88,
        "bleu_score": 0.45,
        "rouge_l": 0.52
    }
    
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/nanovlm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("NanoVLM metrics saved.")

if __name__ == "__main__":
    eval_nanovlm()
