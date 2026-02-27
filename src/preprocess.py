"""
Dataset preprocessing and loading utilities for GSM8K.
"""

import re
from typing import Dict, List, Optional
from datasets import load_dataset
from pathlib import Path


def extract_numeric_answer(answer_text: str) -> Optional[str]:
    """
    Extract numeric answer from GSM8K answer string.
    GSM8K answers are in the format "...#### ANSWER"

    Args:
        answer_text: Raw answer text from GSM8K

    Returns:
        Extracted numeric answer as string, or None if extraction fails
    """
    # GSM8K format: "...#### 42"
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
        return answer
    return None


def normalize_number(num_str: str) -> Optional[str]:
    """
    Normalize a numeric string for comparison.

    Steps:
    1. Strip whitespace
    2. Remove commas
    3. Extract first signed number via regex
    4. Remove trailing .0

    Args:
        num_str: String potentially containing a number

    Returns:
        Normalized number string, or None if parsing fails
    """
    if not num_str:
        return None

    # Strip whitespace
    num_str = num_str.strip()

    # Remove commas
    num_str = num_str.replace(",", "")

    # Extract first signed number
    match = re.search(r"-?\d+(?:\.\d+)?", num_str)
    if not match:
        return None

    num_str = match.group(0)

    # Remove trailing .0
    if "." in num_str and num_str.endswith(".0"):
        num_str = num_str[:-2]

    return num_str


def numbers_equal(pred: str, gold: str, tolerance: float = 1e-4) -> bool:
    """
    Compare two numeric strings with normalization and tolerance.

    Args:
        pred: Predicted answer string
        gold: Gold answer string
        tolerance: Numeric tolerance for float comparison

    Returns:
        True if answers match after normalization
    """
    pred_norm = normalize_number(pred)
    gold_norm = normalize_number(gold)

    if pred_norm is None or gold_norm is None:
        return False

    # String comparison first
    if pred_norm == gold_norm:
        return True

    # Try numeric comparison with tolerance
    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        return abs(pred_float - gold_float) < tolerance
    except (ValueError, TypeError):
        return False


def load_gsm8k_dataset(
    split: str = "test",
    num_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    cache_dir: str = ".cache",
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ("train", "test")
        num_samples: Number of samples to load (None = all)
        shuffle_seed: Random seed for shuffling (None = no shuffle)
        cache_dir: Cache directory for downloaded data

    Returns:
        List of dictionaries with keys: 'question', 'answer', 'numeric_answer'
    """
    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Shuffle if requested
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Limit samples if requested
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Process into standardized format
    processed_data = []
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]
        numeric_answer = extract_numeric_answer(answer_text)

        if numeric_answer is not None:
            processed_data.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "numeric_answer": numeric_answer,
                }
            )

    return processed_data


def evaluate_predictions(
    predictions: List[str], gold_answers: List[str]
) -> Dict[str, float]:
    """
    Evaluate predictions against gold answers.

    Args:
        predictions: List of predicted numeric answers
        gold_answers: List of gold numeric answers

    Returns:
        Dictionary with evaluation metrics
    """
    assert len(predictions) == len(gold_answers), "Length mismatch"

    correct = 0
    total = len(predictions)

    for pred, gold in zip(predictions, gold_answers):
        if numbers_equal(pred, gold):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {"accuracy": accuracy, "correct": correct, "total": total}
