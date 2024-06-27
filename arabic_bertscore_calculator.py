import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel
import csv
import sys

def load_sentences_from_csv(file_path):
    references = []
    candidates = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:
                references.append(row[0])
                candidates.append(row[1])
    return references, candidates

def calculate_bertscore(references, candidates):
    # Ensure CUDA is available (if you're using a GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Arabic BERT model
    model_name = "asafaya/bert-base-arabic"
    scorer = BERTScorer(model_type=model_name, num_layers=12, rescale_with_baseline=False)

    # Calculate BERTScore
    P, R, F1 = scorer.score(candidates, references)

    # Print results
    print("\nBERTScore results:")
    for i, (p, r, f1) in enumerate(zip(P, R, F1)):
        print(f"Sentence pair {i+1}:")
        print(f"  Reference: {references[i]}")
        print(f"  Candidate: {candidates[i]}")
        print(f"  Precision: {p.item():.4f}")
        print(f"  Recall: {r.item():.4f}")
        print(f"  F1: {f1.item():.4f}")
        print()

    # Calculate and print average scores
    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()

    print("Average scores:")
    print(f"  Precision: {avg_P:.4f}")
    print(f"  Recall: {avg_R:.4f}")
    print(f"  F1: {avg_F1:.4f}")

def main():
    print("Welcome to the Arabic BERTScore Calculator!")
    print("Choose an option:")
    print("1. Input two sentences")
    print("2. Provide a CSV file")
    
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        reference = input("Enter the reference sentence: ")
        candidate = input("Enter the candidate sentence: ")
        calculate_bertscore([reference], [candidate])
    elif choice == '2':
        file_path = input("Enter the path to your CSV file: ")
        try:
            references, candidates = load_sentences_from_csv(file_path)
            if not references or not candidates:
                print("Error: The CSV file is empty or improperly formatted.")
                sys.exit(1)
            calculate_bertscore(references, candidates)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            sys.exit(1)
        except csv.Error as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
