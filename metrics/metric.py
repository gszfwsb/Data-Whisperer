from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
import nltk
import re



def compute_metrics(eval_preds, tokenizer, metric):
    """
    Compute evaluation metrics based on the specified metric.

    """
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if metric == "rougeL":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l_scores = [
            scorer.score(label, pred)["rougeL"].fmeasure
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        return {"rougeL": sum(rouge_l_scores) / len(rouge_l_scores)}

    elif metric == "bleu4":
        references = [nltk.word_tokenize(label) for label in decoded_labels]
        predictions = [nltk.word_tokenize(pred) for pred in decoded_preds]
        bleu4 = corpus_bleu([[ref] for ref in references], predictions, weights=(0.25, 0.25, 0.25, 0.25))
        return {"bleu4": bleu4}

    else:
        raise ValueError("Unsupported metric. Choose either 'rougeL' or 'bleu4'.")


def calculate_rouge_l(reference, prediction):
    """
    Calculate the ROUGE-L score 
        float: The ROUGE-L F1 score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    rouge_l_f1 = scores["rougeL"].fmeasure
    return rouge_l_f1

def exact_match(prediction, reference):
    """
    In gsm8k dataset, the exact match metric is implemented to test the accuracy of the final result.
    The final result is shown at the end by #### result.
    """
    ref_number = 0
    pred_number = 1
    try:
        ref_number = float(reference)
        pred_number = float(prediction)

    # Log the extracted numbers for debugging
        # print(f"Ref:{ref_number}, Pred:{pred_number}")
        # print(f'{ref_number == pred_number}')
    except ValueError:
        # Log error if conversion fails
        print(f"Error: Unable to convert to float. Reference: {reference}, Prediction: {prediction}")
    # Return True if numbers match (including equivalence of 2.50 and 2.5), False otherwise
    return ref_number == pred_number

METRICS = {
   
    "exact_match": exact_match,
    "rouge-L": calculate_rouge_l,
}

