#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
from typing import Dict, List


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate model evaluation metrics: accuracy, recall, precision, F1 score, and Matthews correlation coefficient
    
    Args:
        predictions: List of prediction results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    tp = 0 
    tn = 0 
    fp = 0 
    fn = 0 
    
    for pred in predictions:
        if pred["predicted_up"] and pred["label"] == 1:
            tp += 1
        elif not pred["predicted_up"] and pred["label"] == 0:
            tn += 1
        elif pred["predicted_up"] and pred["label"] == 0:
            fp += 1
        elif not pred["predicted_up"] and pred["label"] == 1:
            fn += 1
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
    mcc = mcc_numerator / mcc_denominator
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        },
        "total_predictions": tp + tn + fp + fn
    }


def main():
    try:
        with open('prediction_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("Warning: No prediction results found!")
            return
            
        metrics = calculate_metrics(predictions)
        
        print("\n" + "="*50)
        print("Market Prediction Model Evaluation Report")
        print("="*50)
        
        print(f"\nTotal predictions: {metrics['total_predictions']}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives (TP): {metrics['confusion_matrix']['tp']} - Predicted up and actually went up")
        print(f"True Negatives (TN): {metrics['confusion_matrix']['tn']} - Predicted down and actually went down")
        print(f"False Positives (FP): {metrics['confusion_matrix']['fp']} - Predicted up but actually went down")
        print(f"False Negatives (FN): {metrics['confusion_matrix']['fn']} - Predicted down but actually went up")
        
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Recall:   {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"MCC:     {metrics['mcc']:.4f}")
        
        if len(predictions) >= 50:
            window_size = 50
            print("\n" + "-"*50)
            print(f"Evaluation metrics for the most recent {window_size} predictions:")
            recent_metrics = calculate_metrics(predictions[-window_size:])
            print(f"Accuracy: {recent_metrics['accuracy']:.4f}")
            print(f"Recall:   {recent_metrics['recall']:.4f}")
            print(f"Precision: {recent_metrics['precision']:.4f}")
            print(f"F1 Score: {recent_metrics['f1']:.4f}")
            print(f"MCC:     {recent_metrics['mcc']:.4f}")
        
    except FileNotFoundError:
        print("Error: prediction_results.json file not found")
    except json.JSONDecodeError:
        print("Error: prediction_results.json has an incorrect format")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 