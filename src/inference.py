"""
Inference script for prompt-based evaluation.
Implements different prompting strategies for GSM8K.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import LLMWrapper
from src.preprocess import load_gsm8k_dataset, normalize_number


class InferenceMethod:
    """Base class for inference methods."""

    def __init__(self, config: DictConfig, model: LLMWrapper):
        self.config = config
        self.model = model

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run inference on a single question.

        Returns:
            Dict with 'answer' (final answer) and 'num_calls' (API calls used)
        """
        raise NotImplementedError


class ZeroShotDirect(InferenceMethod):
    """Zero-shot direct answer (single call)."""

    def run(self, question: str) -> Dict[str, Any]:
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: Config structure mismatch - method config is now passed as config.run
        # [CAUSE]: get_method now receives config.run instead of config
        # [FIX]: self.config already points to the run config, so no changes needed here
        # (The fix was in run_inference where we pass config.run to get_method)

        # [VALIDATOR FIX - Attempt 3]
        # [PROBLEM]: Model may output extra text, normalize_number extracts first number which may be wrong
        # [CAUSE]: Even with "output only number" prompt, model may add context or list multiple numbers
        # [FIX]: For consistency with HiddenCoT, support ANSWER: delimiter extraction
        #
        # [OLD CODE]:
        # prompt = f"{self.config.method.prompt}\n\n{question}"
        # answer = self.model.generate(prompt)
        # return {"answer": answer, "num_calls": 1}
        #
        # [NEW CODE]:
        prompt = f"{self.config.method.prompt}\n\n{question}"
        response = self.model.generate(prompt)

        # Extract final answer after delimiter if present (for consistency)
        if "ANSWER:" in response:
            answer = response.split("ANSWER:")[-1].strip()
        else:
            # For zero-shot, we expect model to output just the number
            answer = response

        return {"answer": answer, "num_calls": 1}


class HiddenCoT(InferenceMethod):
    """Hidden chain-of-thought (single call)."""

    def run(self, question: str) -> Dict[str, Any]:
        # [VALIDATOR FIX - Attempt 4]
        # [PROBLEM]: Previous fix used ANSWER: delimiter but model doesn't reliably output it
        # [CAUSE]: Config changed to force number-only output without ANSWER: delimiter
        # [FIX]: Simplified to just use model output directly (should be just a number now)
        #
        # [OLD CODE]:
        # prompt = f"{self.config.method.prompt}\n\n{question}"
        # response = self.model.generate(prompt)
        # if "ANSWER:" in response:
        #     answer = response.split("ANSWER:")[-1].strip()
        # else:
        #     answer = response
        # return {"answer": answer, "num_calls": 1}
        #
        # [NEW CODE]:
        prompt = f"{self.config.method.prompt}\n\n{question}"
        answer = self.model.generate(prompt)
        return {"answer": answer, "num_calls": 1}


class SelfConsistency(InferenceMethod):
    """Self-consistency with majority voting."""

    def run(self, question: str) -> Dict[str, Any]:
        # [VALIDATOR FIX - Attempt 3]
        # [PROBLEM]: normalize_number extracts first number which may not be the final answer
        # [CAUSE]: Responses contain reasoning with numbers before the final answer
        # [FIX]: Extract text after ANSWER: delimiter before normalization
        #
        # [OLD CODE]:
        # num_samples = self.config.method.num_samples
        # prompt = f"{self.config.method.prompt}\n\n{question}"
        # answers = self.model.batch_generate([prompt] * num_samples, temperature=self.config.model.temperature)
        # normalized = [normalize_number(ans) for ans in answers]
        #
        # [NEW CODE]:
        num_samples = self.config.method.num_samples
        prompt = f"{self.config.method.prompt}\n\n{question}"

        # Generate multiple samples with temperature > 0
        responses = self.model.batch_generate(
            [prompt] * num_samples, temperature=self.config.model.temperature
        )

        # Extract final answers after delimiter
        answers = []
        for response in responses:
            if "ANSWER:" in response:
                answer = response.split("ANSWER:")[-1].strip()
            else:
                answer = response
            answers.append(answer)

        # Normalize and count
        normalized = [normalize_number(ans) for ans in answers]
        normalized = [n for n in normalized if n is not None]

        if not normalized:
            # Fallback to first raw answer if normalization fails
            return {"answer": answers[0], "num_calls": num_samples}

        # Majority vote
        counter = Counter(normalized)
        majority_answer = counter.most_common(1)[0][0]

        return {"answer": majority_answer, "num_calls": num_samples}


class SGDRaME(InferenceMethod):
    """Stability-Gated Dual-Reasoning with Minimal Edits."""

    def run(self, question: str) -> Dict[str, Any]:
        # [VALIDATOR FIX - Attempt 3]
        # [PROBLEM]: normalize_number extracts first number from reasoning text, not final answer
        # [CAUSE]: Responses contain reasoning steps with numbers before ANSWER: delimiter
        # [FIX]: Extract text after ANSWER: before normalization and comparison
        #
        # [OLD CODE]:
        # prompt1 = f"{self.config.method.derive1_prompt}\n\n{question}"
        # answer1 = self.model.generate(prompt1)
        # prompt2 = f"{self.config.method.stability_probe_edit}\n\n{question}"
        # answer2 = self.model.generate(prompt2)
        # norm1 = normalize_number(answer1)
        # norm2 = normalize_number(answer2)
        #
        # [NEW CODE]:
        # Step 1: Derive-1 (hidden CoT)
        prompt1 = f"{self.config.method.derive1_prompt}\n\n{question}"
        response1 = self.model.generate(prompt1)
        answer1 = (
            response1.split("ANSWER:")[-1].strip()
            if "ANSWER:" in response1
            else response1
        )

        # Step 2: Stability probe (minimal edit)
        prompt2 = f"{self.config.method.stability_probe_edit}\n\n{question}"
        response2 = self.model.generate(prompt2)
        answer2 = (
            response2.split("ANSWER:")[-1].strip()
            if "ANSWER:" in response2
            else response2
        )

        # Normalize for comparison
        norm1 = normalize_number(answer1)
        norm2 = normalize_number(answer2)

        # Step 3: Check agreement
        if norm1 == norm2 and norm1 is not None:
            # High stability - use this answer
            return {"answer": answer1, "num_calls": 2}
        else:
            # Disagreement - run repair
            repair_prompt = self.config.method.repair_prompt.format(
                answer1=norm1 or answer1, answer2=norm2 or answer2
            )
            repair_prompt = f"{repair_prompt}\n\n{question}"
            response3 = self.model.generate(repair_prompt)
            final_answer = (
                response3.split("ANSWER:")[-1].strip()
                if "ANSWER:" in response3
                else response3
            )
            return {"answer": final_answer, "num_calls": 3}


def get_method(config: DictConfig, model: LLMWrapper) -> InferenceMethod:
    """Factory function to get the appropriate inference method."""
    method_type = config.method.type

    methods = {
        "zero_shot_direct": ZeroShotDirect,
        "hidden_cot": HiddenCoT,
        "self_consistency": SelfConsistency,
        "sg_drame": SGDRaME,
    }

    if method_type not in methods:
        raise ValueError(f"Unknown method type: {method_type}")

    return methods[method_type](config, model)


def run_inference(config: DictConfig) -> None:
    """
    Main inference function.

    Args:
        config: Hydra configuration
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'dataset' is not in struct (and similar for model, method)
    # [CAUSE]: Run-specific configs are loaded under config.run, not config root
    # [FIX]: Changed config.model to config.run.model, config.dataset to config.run.dataset, config.method to config.run.method
    #
    # [OLD CODE]:
    # print(f"Initializing model: {config.model.provider}/{config.model.model_name}")
    # model = LLMWrapper(
    #     provider=config.model.provider,
    #     model_name=config.model.model_name,
    #     temperature=config.model.temperature,
    #     max_tokens=config.model.max_tokens,
    # )
    #
    # [NEW CODE]:
    # Initialize model
    print(
        f"Initializing model: {config.run.model.provider}/{config.run.model.model_name}"
    )
    model = LLMWrapper(
        provider=config.run.model.provider,
        model_name=config.run.model.model_name,
        temperature=config.run.model.temperature,
        max_tokens=config.run.model.max_tokens,
    )

    # [VALIDATOR FIX - Attempt 1]
    # [OLD CODE]:
    # print(f"Loading dataset: {config.dataset.name}")
    # dataset = load_gsm8k_dataset(
    #     split=config.dataset.split,
    #     num_samples=config.dataset.num_samples,
    #     shuffle_seed=config.dataset.shuffle_seed,
    #     cache_dir=config.cache_dir,
    # )
    #
    # [NEW CODE]:
    # Load dataset
    print(f"Loading dataset: {config.run.dataset.name}")
    dataset = load_gsm8k_dataset(
        split=config.run.dataset.split,
        num_samples=config.run.dataset.num_samples,
        shuffle_seed=config.run.dataset.shuffle_seed,
        cache_dir=config.cache_dir,
    )
    print(f"Loaded {len(dataset)} samples")

    # [VALIDATOR FIX - Attempt 1]
    # [OLD CODE]:
    # method = get_method(config, model)
    # print(f"Using method: {config.method.type}")
    #
    # [NEW CODE]:
    # Get inference method
    method = get_method(config.run, model)
    print(f"Using method: {config.run.method.type}")

    # Initialize WandB
    wandb_enabled = config.wandb.mode != "disabled"
    if wandb_enabled:
        # In sanity_check mode, use a separate project namespace
        project = config.wandb.project
        if config.mode == "sanity_check" and not project.endswith("-sanity"):
            project = f"{project}-sanity"

        wandb.init(
            entity=config.wandb.entity,
            project=project,
            name=config.run.run_id,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.wandb.mode,
        )
        print(f"WandB initialized: {wandb.run.url}")

    # Run inference
    results = []
    predictions = []
    gold_answers = []
    total_calls = 0

    print("Running inference...")
    for item in tqdm(dataset, desc="Processing"):
        question = item["question"]
        gold_answer = item["numeric_answer"]

        # Run method
        result = method.run(question)
        pred_answer = result["answer"]
        num_calls = result["num_calls"]

        total_calls += num_calls
        predictions.append(pred_answer)
        gold_answers.append(gold_answer)

        results.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "num_calls": num_calls,
            }
        )

    # Evaluate
    from src.preprocess import evaluate_predictions

    metrics = evaluate_predictions(predictions, gold_answers)

    avg_calls = total_calls / len(dataset)
    metrics["avg_calls"] = avg_calls
    metrics["total_calls"] = total_calls

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Correct: {metrics['correct']}/{metrics['total']}")
    print(f"  Avg calls: {avg_calls:.2f}")

    # Log to WandB
    if wandb_enabled:
        wandb.log(metrics)
        for key, value in metrics.items():
            wandb.summary[key] = value

    # Save results
    results_dir = Path(config.results_dir) / config.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to: {results_file}")

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # Sanity validation
    if config.mode == "sanity_check":
        perform_sanity_validation(metrics, len(dataset))

    if wandb_enabled:
        wandb.finish()


def perform_sanity_validation(metrics: Dict[str, Any], num_samples: int) -> None:
    """
    Perform sanity validation for inference task.

    Args:
        metrics: Computed metrics
        num_samples: Number of samples processed
    """
    # Check if we processed enough samples
    if num_samples < 5:
        print(
            f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {num_samples}, need >=5)"
        )
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'required': 5})}"
        )
        sys.exit(1)

    # Check if metrics are present
    required_metrics = ["accuracy", "avg_calls"]
    missing = [m for m in required_metrics if m not in metrics]
    if missing:
        print(f"SANITY_VALIDATION: FAIL reason=missing_metrics ({', '.join(missing)})")
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps({'missing': missing})}")
        sys.exit(1)

    # Check if metrics are valid (no NaN/inf)
    for key, value in metrics.items():
        if isinstance(value, float):
            if not (value == value and abs(value) != float("inf")):  # NaN or inf check
                print(f"SANITY_VALIDATION: FAIL reason=invalid_metric ({key}={value})")
                print(
                    f"SANITY_VALIDATION_SUMMARY: {json.dumps({'invalid_metric': key, 'value': str(value)})}"
                )
                sys.exit(1)

    # Check if outputs are valid (accuracy should be in [0, 1])
    if not (0.0 <= metrics["accuracy"] <= 1.0):
        print(
            f"SANITY_VALIDATION: FAIL reason=invalid_accuracy ({metrics['accuracy']})"
        )
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'accuracy': metrics['accuracy']})}"
        )
        sys.exit(1)

    # All checks passed
    print(f"SANITY_VALIDATION: PASS")
    summary = {
        "samples": num_samples,
        "accuracy": metrics["accuracy"],
        "avg_calls": metrics["avg_calls"],
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
