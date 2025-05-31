import json
import argparse
import re

def extract_answer(text):
    answer_pattern = re.compile(r"####\s*(-?\d+\.?\d*)")
    match = answer_pattern.search(text)
    return float(match.group(1)) if match else None

def calculate_accuracy(file_path):
    correct = 0
    total = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            label_answer = extract_answer(data["label"])
            predict_answer = extract_answer(data["predict"])
            
            if label_answer is not None and predict_answer == label_answer:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from JSONL file.")
    parser.add_argument("--p", type=str, required=True, help="Path to the JSONL file")
    args = parser.parse_args()
    
    calculate_accuracy(args.p)