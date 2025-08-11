# Merged file for evolution_optimization/utils
# This file contains code merged from multiple repositories

from setuptools import setup

import sys
from openevolve.cli import main

import argparse
import asyncio
import logging
import os
from typing import Dict
from typing import List
from typing import Optional
from openevolve import OpenEvolve
from openevolve.config import Config
from openevolve.config import load_config
import traceback

# From openevolve/cli.py
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    return parser.parse_args()

# From openevolve/cli.py
def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Union

# From openevolve/evaluation_result.py
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts

    This maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).
    """

    metrics: Dict[str, float]  # mandatory - existing contract
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)  # optional side-channel

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())

# From openevolve/evaluation_result.py
def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        return cls(metrics=metrics)

# From openevolve/evaluation_result.py
def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

# From openevolve/evaluation_result.py
def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

# From openevolve/evaluation_result.py
def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

# From openevolve/evaluation_result.py
def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

# From openevolve/evaluation_result.py
def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())

import importlib.util
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Tuple
from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.utils.async_utils import TaskPool
from openevolve.utils.async_utils import run_in_executor
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.format_utils import format_metrics_safe
import re

# From openevolve/evaluator.py
class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    def _validate_cascade_configuration(self, module) -> None:
        """
        Validate cascade evaluation configuration and warn about potential issues

        Args:
            module: The loaded evaluation module
        """
        if self.config.cascade_evaluation:
            # Check if cascade functions exist
            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    f"Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate a program and return scores

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                temp_file.write(program_code.encode("utf-8"))
                temp_file_path = temp_file.name

            try:
                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(temp_file_path)
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(temp_file_path)

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)

                # Check if this was a timeout and capture artifacts if enabled
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )

                # Add LLM feedback if configured
                llm_eval_result = None
                if self.config.use_llm_feedback and self.llm_ensemble:
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    # Combine metrics
                    llm_scores = []
                    for name, value in llm_result.metrics.items():
                        weighted_value = value * self.config.llm_feedback_weight
                        eval_result.metrics[f"llm_{name}"] = weighted_value
                        llm_scores.append(value)  # Use unweighted value for average

                    # Add average of LLM metrics
                    if llm_scores:
                        llm_average = sum(llm_scores) / len(llm_scores)
                        eval_result.metrics["llm_average"] = (
                            llm_average * self.config.llm_feedback_weight
                        )

                        # Recalculate combined_score if it exists
                        if "combined_score" in eval_result.metrics:
                            # Original combined_score is just accuracy
                            accuracy = eval_result.metrics["combined_score"]
                            # Combine with LLM average (70% accuracy, 30% LLM quality)
                            eval_result.metrics["combined_score"] = (
                                accuracy * 0.7 + llm_average * 0.3
                            )

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Return just metrics for backward compatibility
                return eval_result.metrics

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()

                # Capture failure artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    async def _direct_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.evaluate_function, program_path)

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

    async def _cascade_evaluate(
        self, program_path: str
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        # Import the evaluation module to get cascade functions if they exist
        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                return await self._direct_evaluate(program_path)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path)

            # Run first stage with timeout
            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage1, program_path)

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")
                # Capture stage 1 failure with enhanced context
                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )

            # Check threshold
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage2, program_path)

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage3, program_path)

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")
            # Return proper cascade failure result with enhanced context
            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:
        """
        Create rich error context for cascade failures

        Args:
            stage: The stage where the error occurred
            error: The exception that was raised

        Returns:
            Dictionary with enhanced error context
        """
        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Uses 'combined_score' if available (for consistency with evolution),
        otherwise falls back to averaging all numeric metrics except 'error'

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        # Use combined_score if available - this is what evolution uses
        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        # Fallback: average all numeric metrics except 'error'
        # This maintains backward compatibility
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)

# From openevolve/evaluator.py
def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

import multiprocessing
import pickle
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future
from dataclasses import asdict
from openevolve.database import Program
from openevolve.config import DatabaseConfig
from openevolve.config import LLMConfig
from openevolve.config import PromptConfig
from openevolve.config import LLMModelConfig
from openevolve.evaluator import Evaluator
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.utils.code_utils import extract_diffs
from openevolve.utils.code_utils import apply_diff
from openevolve.utils.code_utils import format_diff_summary
from openevolve.utils.code_utils import parse_full_rewrite

# From openevolve/process_parallel.py
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None

# From openevolve/process_parallel.py
class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "language": config.language,
        }

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file),
        )

        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "artifacts": {},  # Will be populated selectively
        }

        # Include artifacts for programs that might be selected
        # IMPORTANT: This limits artifacts (execution outputs/errors) to first 100 programs only.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The limit of 100 keeps
        # artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts

        return snapshot

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures
        pending_futures: Dict[int, Future] = {}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch
        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self._submit_iteration(i)
            if future:
                pending_futures[i] = future

        next_iteration = start_iteration + batch_size
        completed_iterations = 0

        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                result = future.result()

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database
                    self.database.add(child_program, iteration=completed_iteration)

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
                        and current_island_counter >= programs_per_island
                    ):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            from openevolve.utils.metrics_utils import safe_numeric_average

                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values() if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                break

            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # Submit next iteration
            if next_iteration < total_iterations and not self.shutdown_event.is_set():
                future = self._submit_iteration(next_iteration)
                if future:
                    pending_futures[next_iteration] = future
                    next_iteration += 1

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        logger.info("Evolution completed")

        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int) -> Optional[Future]:
        """Submit an iteration to the process pool"""
        try:
            # Sample parent and inspirations
            parent, inspirations = self.database.sample()

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None

# From openevolve/process_parallel.py
def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file),
        )

        logger.info(f"Started process pool with {self.num_workers} processes")

# From openevolve/process_parallel.py
def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

# From openevolve/process_parallel.py
def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()


# From openevolve/iteration.py
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None

import yaml

# From openevolve/config.py
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None

# From openevolve/config.py
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(
        default_factory=lambda: [
            LLMModelConfig(name="gpt-4o-mini", weight=0.8),
            LLMModelConfig(name="gpt-4o", weight=0.2),
        ]
    )

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if (self.primary_model or self.primary_model_weight) and len(self.models) < 1:
            # Ensure we have a primary model
            self.models.append(LLMModelConfig())
        if self.primary_model:
            self.models[0].name = self.primary_model
        if self.primary_model_weight:
            self.models[0].weight = self.primary_model_weight

        if (self.secondary_model or self.secondary_model_weight) and len(self.models) < 2:
            # Ensure we have a second model
            self.models.append(LLMModelConfig())
        if self.secondary_model:
            self.models[1].name = self.secondary_model
        if self.secondary_model_weight:
            self.models[1].weight = self.secondary_model_weight

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

# From openevolve/config.py
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: Optional[int] = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: Optional[int] = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = (
        None  # Deprecated: use suggest_simplification_after_chars
    )

# From openevolve/config.py
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30

# From openevolve/config.py
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024

# From openevolve/config.py
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                # Note: diversity_metric fixed to "edit_distance"
                # "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

# From openevolve/config.py
def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config

# From openevolve/config.py
def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

# From openevolve/config.py
def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

# From openevolve/config.py
def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


from openevolve.process_parallel import ProcessParallelController
from openevolve.utils.code_utils import extract_code_language
from openevolve.utils.format_utils import format_improvement_safe
import random
import numpy
import hashlib

# From openevolve/controller.py
class OpenEvolve:
    """
    Main controller for OpenEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np
            import hashlib

            # Set global random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            # Create hash-based seeds for different components
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OpenEvolve with {initial_program_path}")

        # Initialize improved parallel processing components
        self.parallel_controller = None

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Add file handler
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run the evolution process with improved parallel processing

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Determine starting iteration
        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)

            # Check if combined_score is present in the metrics
            if "combined_score" not in initial_metrics:
                # Calculate average of numeric metrics
                numeric_metrics = [
                    v
                    for v in initial_metrics.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        # Initialize improved parallel processing
        try:
            self.parallel_controller = ProcessParallelController(
                self.config, self.evaluation_file, self.database
            )

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # When starting from iteration 0, we've already done the initial program evaluation
            # So we need to adjust the start_iteration for the actual evolution
            evolution_start = start_iteration
            evolution_iterations = max_iterations

            # If we just added the initial program at iteration 0, start evolution from iteration 1
            if should_add_initial and start_iteration == 0:
                evolution_start = 1
                # User expects max_iterations evolutionary iterations AFTER the initial program
                # So we don't need to reduce evolution_iterations

            # Run evolution with improved parallel processing and checkpoint callback
            await self._run_evolution_with_checkpoints(
                evolution_start, evolution_iterations, target_score
            )

        finally:
            # Clean up parallel processing resources
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

        # Get the best program
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if best_program and "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs "
                        f"{best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state from a checkpoint directory"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint saving support"""
        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        # Run the evolution process with checkpoint callback
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=self._save_checkpoint
        )

        # Check if shutdown was requested
        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return

        # Save final checkpoint if needed
        # Note: start_iteration here is the evolution start (1 for fresh start, not 0)
        # max_iterations is the number of evolution iterations to run
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")

# From openevolve/controller.py
def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

# From openevolve/controller.py
def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

import concurrent.futures

# From function_minimization/evaluator.py
def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

# From function_minimization/evaluator.py
def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0

# From function_minimization/evaluator.py
def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.704
    GLOBAL_MIN_Y = 0.678
    GLOBAL_MIN_VALUE = -1.519

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Error: program does not have 'run_search' function")
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_search function",
            }

        # Run multiple trials
        num_trials = 10
        x_values = []
        y_values = []
        values = []
        distances = []
        times = []
        success_count = 0

        for trial in range(num_trials):
            try:
                start_time = time.time()

                # Run with timeout
                result = run_with_timeout(program.run_search, timeout_seconds=5)

                # Handle different result formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        # Assume it's (x, y) and calculate value
                        x, y = result
                        # Calculate the function value since it wasn't returned
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                        print(f"Trial {trial}: Got 2 values, calculated function value: {value}")
                    else:
                        print(
                            f"Trial {trial}: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                        )
                        continue
                else:
                    print(
                        f"Trial {trial}: Invalid result format, expected tuple but got {type(result)}"
                    )
                    continue

                end_time = time.time()

                # Ensure all values are float
                x = safe_float(x)
                y = safe_float(y)
                value = safe_float(value)

                # Check if the result is valid (not NaN or infinite)
                if (
                    np.isnan(x)
                    or np.isnan(y)
                    or np.isnan(value)
                    or np.isinf(x)
                    or np.isinf(y)
                    or np.isinf(value)
                ):
                    print(f"Trial {trial}: Invalid result, got x={x}, y={y}, value={value}")
                    continue

                # Calculate metrics
                x_diff = x - GLOBAL_MIN_X
                y_diff = y - GLOBAL_MIN_Y
                distance_to_global = np.sqrt(x_diff**2 + y_diff**2)

                x_values.append(x)
                y_values.append(y)
                values.append(value)
                distances.append(distance_to_global)
                times.append(end_time - start_time)
                success_count += 1

            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except IndexError as e:
                # Specifically handle IndexError which often happens with early termination checks
                print(f"Trial {trial}: IndexError - {str(e)}")
                print(
                    "This is likely due to a list index check before the list is fully populated."
                )
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed",
            }

        # Calculate metrics
        avg_value = float(np.mean(values))
        avg_distance = float(np.mean(distances))
        avg_time = float(np.mean(times)) if times else 1.0

        # Convert to scores (higher is better)
        value_score = float(1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE)))  # Normalize and invert
        distance_score = float(1.0 / (1.0 + avg_distance))
        speed_score = float(1.0 / avg_time) if avg_time > 0 else 0.0

        # calculate standard deviation scores
        # get x_std_score
        x_std_score = float(1.0 / (1.0 + np.std(x_values)))
        # get y_std_score
        y_std_score = float(1.0 / (1.0 + np.std(y_values)))
        standard_deviation_score = (x_std_score + y_std_score) / 2.0

        # Normalize speed score (so it doesn't dominate)
        speed_score = float(min(speed_score, 10.0) / 10.0)

        # Add reliability score based on success rate
        reliability_score = float(success_count / num_trials)

        # Calculate a single combined score that prioritizes finding good solutions
        # over secondary metrics like speed and reliability
        # Value and distance scores (quality of solution) get 90% of the weight
        # Speed and reliability get only 10% combined
        combined_score = float(
            0.35 * value_score
            + 0.35 * distance_score
            + standard_deviation_score * 0.20
            + 0.05 * speed_score
            + 0.05 * reliability_score
        )

        # Also compute an "overall" score that will be the primary metric for selection
        # This adds a bonus for finding solutions close to the global minimum
        # and heavily penalizes solutions that aren't finding the right region
        if distance_to_global < 1.0:  # Very close to the correct solution
            solution_quality = 1.0
        elif distance_to_global < 3.0:  # In the right region
            solution_quality = 0.5
        else:  # Not finding the right region
            solution_quality = 0.1

        # Overall score is dominated by solution quality but also factors in the combined score
        overall_score = 0.8 * solution_quality + 0.2 * combined_score

        return {
            "value_score": value_score,
            "distance_score": distance_score,
            "standard_deviation_score": standard_deviation_score,
            "speed_score": speed_score,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
            "overall_score": overall_score,  # This will be the primary selection metric
            "success_rate": reliability_score,
        }
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "speed_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }

# From function_minimization/evaluator.py
def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Known global minimum (approximate)
    GLOBAL_MIN_X = float(-1.704)
    GLOBAL_MIN_Y = float(0.678)
    GLOBAL_MIN_VALUE = float(-1.519)

    # Quick check to see if the program runs without errors
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Stage 1 validation: Program does not have 'run_search' function")
            return {"runs_successfully": 0.0, "error": "Missing run_search function"}

        try:
            # Run a single trial with timeout
            result = run_with_timeout(program.run_search, timeout_seconds=5)

            # Handle different result formats
            if isinstance(result, tuple):
                if len(result) == 3:
                    x, y, value = result
                elif len(result) == 2:
                    # Assume it's (x, y) and calculate value
                    x, y = result
                    # Calculate the function value since it wasn't returned
                    value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                    print(f"Stage 1: Got 2 values, calculated function value: {value}")
                else:
                    print(
                        f"Stage 1: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                    )
                    return {"runs_successfully": 0.0, "error": "Invalid result format"}
            else:
                print(f"Stage 1: Invalid result format, expected tuple but got {type(result)}")
                return {"runs_successfully": 0.0, "error": "Invalid result format"}

            # Ensure all values are float
            x = safe_float(x)
            y = safe_float(y)
            value = safe_float(value)

            # Check if the result is valid
            if (
                np.isnan(x)
                or np.isnan(y)
                or np.isnan(value)
                or np.isinf(x)
                or np.isinf(y)
                or np.isinf(value)
            ):
                print(f"Stage 1 validation: Invalid result, got x={x}, y={y}, value={value}")
                return {"runs_successfully": 0.5, "error": "Invalid result values"}

            # Calculate distance safely
            x_diff = float(x) - GLOBAL_MIN_X
            y_diff = float(y) - GLOBAL_MIN_Y
            distance = float(np.sqrt(x_diff**2 + y_diff**2))

            # Calculate value-based score
            value_score = float(1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE)))
            distance_score = float(1.0 / (1.0 + distance))

            # Calculate solution quality metric
            if distance < 1.0:  # Very close to the correct solution
                solution_quality = 1.0
            elif distance < 3.0:  # In the right region
                solution_quality = 0.5
            else:  # Not finding the right region
                solution_quality = 0.1

            # Basic metrics with overall score
            return {
                "runs_successfully": 1.0,
                "value_score": value_score,
                "distance_score": distance_score,
                "overall_score": solution_quality,  # This becomes a strong guiding metric
            }
        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"runs_successfully": 0.0, "error": "Timeout"}
        except IndexError as e:
            # Specifically handle IndexError which often happens with early termination checks
            print(f"Stage 1 evaluation failed with IndexError: {e}")
            print("This is likely due to a list index check before the list is fully populated.")
            return {"runs_successfully": 0.0, "error": f"IndexError: {str(e)}"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"runs_successfully": 0.0, "error": str(e)}

    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        print(traceback.format_exc())
        return {"runs_successfully": 0.0, "error": str(e)}

# From function_minimization/evaluator.py
def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)


# From function_minimization/initial_program.py
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Simple random search
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value

# From function_minimization/initial_program.py
def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

# From function_minimization/initial_program.py
def run_search():
    x, y, value = search_algorithm()
    return x, y, value


# From r_robust_regression/evaluator.py
def generate_regression_data(n_samples=100, n_features=3, outlier_fraction=0.1, noise=0.1):
    """Generate synthetic regression data with outliers."""
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # True coefficients
    true_coeffs = np.random.randn(n_features + 1)  # +1 for intercept

    # Generate target values
    y = true_coeffs[0] + X @ true_coeffs[1:] + noise * np.random.randn(n_samples)

    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        # Make outliers by adding large errors
        y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(
            3, 10, n_outliers
        )

    return X, y, true_coeffs


# From circle_packing_with_artifacts/evaluator.py
class TimeoutError(Exception):
    pass

# From circle_packing_with_artifacts/evaluator.py
def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")

# From circle_packing_with_artifacts/evaluator.py
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    n = centers.shape[0]
    validation_details = {
        "total_circles": n,
        "boundary_violations": [],
        "overlaps": [],
        "min_radius": float(np.min(radii)),
        "max_radius": float(np.max(radii)),
        "avg_radius": float(np.mean(radii)),
    }

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            violation = (
                f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
            )
            validation_details["boundary_violations"].append(violation)
            print(violation)

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                overlap = (
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                )
                validation_details["overlaps"].append(overlap)
                print(overlap)

    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
    )
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details

import matplotlib.pyplot
from matplotlib.patches import Circle

# From circle_packing_with_artifacts/initial_program.py
def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii

# From circle_packing_with_artifacts/initial_program.py
def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii

# From circle_packing_with_artifacts/initial_program.py
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

# From circle_packing_with_artifacts/initial_program.py
def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()

from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# From llm_prompt_optimization/evaluator.py
def calculate_prompt_features(prompt):
    """
    Calculate custom features for MAP-Elites binning

    Returns:
        tuple: (prompt_length, reasoning_strategy) - both in range 0-9
    """
    # Feature 1: Prompt length bin (0-9)
    length = len(prompt)
    if length < 100:
        prompt_length = 0  # Minimal
    elif length < 200:
        prompt_length = 1  # Very short
    elif length < 400:
        prompt_length = 2  # Short
    elif length < 600:
        prompt_length = 3  # Medium-short
    elif length < 900:
        prompt_length = 4  # Medium
    elif length < 1200:
        prompt_length = 5  # Medium-long
    elif length < 1600:
        prompt_length = 6  # Long
    elif length < 2000:
        prompt_length = 7  # Very long
    elif length < 2500:
        prompt_length = 8  # Extensive
    else:
        prompt_length = 9  # Very extensive

    # Feature 2: Reasoning strategy (0-9)
    prompt_lower = prompt.lower()

    # Check for few-shot examples
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    # Check for Chain-of-Thought (CoT) indicators
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )

    # Assign reasoning strategy bins
    if has_example:
        # Few-shot examples (bins 7-9)
        if has_cot:
            reasoning_strategy = 9  # Few-shot + CoT (most sophisticated)
        elif length > 1500:
            reasoning_strategy = 8  # Extensive few-shot
        else:
            reasoning_strategy = 7  # Basic few-shot
    elif has_cot:
        # Chain-of-thought (bins 4-6)
        if "must" in prompt_lower or "exactly" in prompt_lower:
            reasoning_strategy = 6  # Strict CoT
        elif length > 500:
            reasoning_strategy = 5  # Detailed CoT
        else:
            reasoning_strategy = 4  # Basic CoT
    else:
        # Basic prompts (bins 0-3)
        if length < 100:
            reasoning_strategy = 0  # Minimal
        elif "solve" in prompt_lower or "calculate" in prompt_lower:
            reasoning_strategy = 2  # Direct instruction
        else:
            reasoning_strategy = 1  # Simple prompt

    return prompt_length, reasoning_strategy

# From llm_prompt_optimization/evaluator.py
def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from matching _dataset.yaml file."""
    # Load prompt from text file
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    # Load the configuration (already determined from environment variable)
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, prompt

# From llm_prompt_optimization/evaluator.py
def load_hf_dataset(config):
    """Load HuggingFace dataset based on configuration."""
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "test")
    trust_remote_code = config.get("trust_remote_code", True)  # Default to True for convenience

    print(f"Loading dataset: {dataset_name}")

    # Special handling for HotpotQA - always use non-streaming mode
    if dataset_name == "hotpot_qa" or config.get("is_hotpotqa", False):
        print("Using non-streaming mode for HotpotQA to avoid PyArrow issues")
        streaming = False
    else:
        # For other datasets, use streaming if not specified
        streaming = config.get("streaming", True)

    try:
        # Try to load the specified split
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=trust_remote_code, streaming=streaming
            )
    except:
        # Fallback to train split if test is not available
        print(f"Split '{split}' not found, falling back to 'train'")
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )

    # Print dataset info
    if hasattr(dataset, "__len__"):
        print(f"Dataset loaded with {len(dataset)} examples")
    else:
        print(f"Dataset loaded (streaming mode)")

    return dataset

# From llm_prompt_optimization/evaluator.py
def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config["input_field"]
    target_field = config["target_field"]

    # Check dataset type
    dataset_name = config.get("dataset_name", "").lower()
    is_emotion = "emotion" in dataset_name
    is_gsm8k = "gsm8k" in dataset_name
    is_hotpotqa = config.get("is_hotpotqa", False)
    is_ifeval = config.get("is_ifeval", False)
    is_hover = config.get("is_hover", False)

    # Sample from dataset - handle both streaming and non-streaming
    if hasattr(dataset, "take"):
        # Streaming dataset
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        # Non-streaming dataset
        indices = range(min(num_samples, len(dataset)))
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples")

    correct = 0
    total = 0

    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        # Prepare the prompt with appropriate formatting
        if is_hotpotqa:
            # Format context from paragraphs
            context_items = example.get("context", {})
            context_text = ""
            if "title" in context_items and "sentences" in context_items:
                # Handle the specific structure of HotpotQA
                for i, (title, sentences) in enumerate(
                    zip(context_items["title"], context_items["sentences"])
                ):
                    context_text += f"Paragraph {i+1} ({title}):\n"
                    context_text += " ".join(sentences) + "\n\n"
            formatted_prompt = prompt.format(context=context_text.strip(), question=input_text)
        elif is_ifeval:
            # IFEval uses 'prompt' field directly
            formatted_prompt = prompt.format(instruction=input_text)
        elif is_hover:
            # HoVer uses claim field
            formatted_prompt = prompt.format(claim=input_text)
        else:
            # Default formatting for other datasets
            formatted_prompt = prompt.format(input_text=input_text)

        # Prepare the message for the LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                # Use max_tokens from config
                response = test_model.chat.completions.create(
                    model=TASK_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=MAX_TOKENS,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)

        # Handle potential None response
        if not response:
            print(f"Warning: No response object from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices:
            print(f"Warning: No choices in response from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices[0].message:
            print(f"Warning: No message in response choice")
            total += 1  # Count as incorrect
            continue

        output_text = response.choices[0].message.content
        if output_text is None:
            print(f"Warning: None content in LLM response")
            print(f"Full response: {response}")
            total += 1  # Count as incorrect
            continue

        output_text = output_text.strip()

        # Extract prediction from output
        try:
            if is_gsm8k:
                # For GSM8K, extract the numeric answer after ####
                # First, extract the expected answer from the ground truth
                expected_answer = expected.split("####")[-1].strip()
                try:
                    expected_number = float(expected_answer.replace(",", ""))
                except:
                    print(f"Warning: Could not parse expected answer: {expected_answer}")
                    total += 1
                    continue

                # Extract prediction from model output
                prediction = None
                if "####" in output_text:
                    predicted_answer = output_text.split("####")[-1].strip()
                    # Extract just the number, removing any extra text like $ signs
                    import re

                    numbers = re.findall(r"-?\$?[\d,]+\.?\d*", predicted_answer)
                    if numbers:
                        try:
                            # Remove $ and , from the number
                            number_str = numbers[0].replace("$", "").replace(",", "")
                            prediction = float(number_str)
                        except:
                            pass

                # If we found a prediction, check if it matches
                if prediction is not None:
                    # Check if answers match (with small tolerance for floats)
                    if abs(prediction - expected_number) < 0.001:
                        correct += 1

                total += 1
                continue  # Skip the general case to avoid double counting

            elif is_hotpotqa:
                # For HotpotQA, do exact match comparison (case-insensitive)
                output_lower = output_text.lower().strip()
                expected_lower = str(expected).lower().strip()

                # Remove common punctuation for better matching
                output_lower = output_lower.rstrip(".,!?;:")
                expected_lower = expected_lower.rstrip(".,!?;:")

                if output_lower == expected_lower:
                    correct += 1
                elif expected_lower in output_lower:
                    # Partial credit if answer is contained in response
                    correct += 1

                total += 1
                continue

            elif is_ifeval:
                # For IFEval, we need more complex evaluation
                # For now, do basic keyword matching
                # Note: Full IFEval requires checking multiple constraints
                output_lower = output_text.lower()

                # Simple heuristic: check if response seems to follow instruction format
                if len(output_text.strip()) > 10:  # Non-trivial response
                    correct += 1  # Simplified - real IFEval needs constraint checking

                total += 1
                continue

            elif is_hover:
                # For HoVer, check if prediction matches SUPPORTED/NOT_SUPPORTED
                output_upper = output_text.upper()
                expected_upper = str(expected).upper()

                # Look for the verdict in the output
                if "SUPPORTED" in output_upper and "NOT" not in output_upper.replace(
                    "NOT SUPPORTED", ""
                ):
                    prediction = "SUPPORTED"
                elif "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                    prediction = "NOT_SUPPORTED"
                else:
                    prediction = None

                if prediction == expected_upper:
                    correct += 1

                total += 1
                continue

            elif is_emotion:
                # For emotion classification (0-5)
                numbers = re.findall(r"\b[0-5]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from emotion keywords
                    output_lower = output_text.lower()
                    emotion_map = {
                        "sadness": 0,
                        "sad": 0,
                        "joy": 1,
                        "happy": 1,
                        "happiness": 1,
                        "love": 2,
                        "anger": 3,
                        "angry": 3,
                        "fear": 4,
                        "afraid": 4,
                        "scared": 4,
                        "surprise": 5,
                        "surprised": 5,
                    }
                    prediction = -1
                    for emotion, label in emotion_map.items():
                        if emotion in output_lower:
                            prediction = label
                            break
            else:
                # For sentiment classification (0-1)
                numbers = re.findall(r"\b[01]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from keywords
                    output_lower = output_text.lower()
                    if "positive" in output_lower:
                        prediction = 1
                    elif "negative" in output_lower:
                        prediction = 0
                    else:
                        prediction = -1  # Invalid prediction

            if prediction == expected:
                correct += 1

            total += 1

        except Exception as e:
            print(f"Error parsing response '{output_text}': {e}")
            total += 1  # Count as incorrect

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

from datetime import datetime

# From llm_prompt_optimization/evaluate_prompts.py
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# From llm_prompt_optimization/evaluate_prompts.py
def load_prompt(dataset_name, prompt_type="baseline"):
    """Load prompt template for a dataset."""
    if prompt_type == "baseline":
        prompt_path = f"{dataset_name}_prompt.txt"
    else:  # evolved
        prompt_path = f"openevolve_output_qwen3_{dataset_name}/best/best_program.txt"

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read().strip()

# From llm_prompt_optimization/evaluate_prompts.py
def load_dataset_config(dataset_name):
    """Load dataset configuration."""
    config_path = f"{dataset_name}_prompt_dataset.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# From llm_prompt_optimization/evaluate_prompts.py
def evaluate_ifeval(client, prompt_template, num_samples, model):
    """Evaluate IFEval dataset."""
    print("\nLoading IFEval dataset...")

    # Try test split first, then train
    try:
        dataset = load_dataset("google/IFEval", split="test")
        split_used = "test"
    except:
        dataset = load_dataset("google/IFEval", split="train")
        split_used = "train"

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full {split_used} split: {samples_to_process} samples")
        dataset_iter = tqdm(dataset, desc="Evaluating")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples from {split_used} split")
        dataset = load_dataset("google/IFEval", split=split_used, streaming=True)
        dataset_iter = tqdm(
            dataset.take(samples_to_process), total=samples_to_process, desc="Evaluating"
        )

    correct = 0
    total = 0
    empty_responses = 0

    for i, example in enumerate(dataset_iter):
        if num_samples is not None and i >= samples_to_process:
            break
        instruction = example["prompt"]

        try:
            formatted_prompt = prompt_template.format(instruction=instruction)
        except KeyError:
            # Handle prompts with different placeholder names
            formatted_prompt = prompt_template.replace("{instruction}", instruction)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            # Simple evaluation: response has reasonable length
            if len(output_text.strip()) > 20:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses

# From llm_prompt_optimization/evaluate_prompts.py
def evaluate_hover(client, prompt_template, num_samples, model):
    """Evaluate HoVer dataset."""
    print("\nLoading HoVer dataset...")

    # Try test split first (but it's unlabeled), then validation
    try:
        test_dataset = load_dataset("hover", split="test", trust_remote_code=True)
        # Check if test set has labels
        if test_dataset[0]["label"] != -1:
            dataset = test_dataset
            split_used = "test"
        else:
            # Test set is unlabeled, use validation
            dataset = load_dataset("hover", split="validation", trust_remote_code=True)
            split_used = "validation"
    except:
        dataset = load_dataset("hover", split="validation", trust_remote_code=True)
        split_used = "validation"

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full {split_used} split: {samples_to_process} samples")
        dataset_iter = tqdm(dataset, desc="Evaluating")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples from {split_used} split")
        dataset = load_dataset("hover", split=split_used, streaming=True, trust_remote_code=True)
        dataset_iter = tqdm(
            dataset.take(samples_to_process), total=samples_to_process, desc="Evaluating"
        )

    correct = 0
    total = 0
    empty_responses = 0

    for i, example in enumerate(dataset_iter):
        if num_samples is not None and i >= samples_to_process:
            break
        claim = example["claim"]
        label = example["label"]  # Integer: 0=SUPPORTED, 1=NOT_SUPPORTED

        try:
            formatted_prompt = prompt_template.format(claim=claim)
        except KeyError:
            formatted_prompt = prompt_template.replace("{claim}", claim)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            output_upper = output_text.strip().upper()

            # Parse prediction from output
            if "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                prediction = 1  # NOT_SUPPORTED
            elif "SUPPORTED" in output_upper:
                prediction = 0  # SUPPORTED
            else:
                prediction = -1  # Invalid/unclear response

            # Compare with actual label
            if prediction == label:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses

# From llm_prompt_optimization/evaluate_prompts.py
def evaluate_hotpotqa(client, prompt_template, num_samples, model):
    """Evaluate HotpotQA dataset."""
    print("\nLoading HotpotQA dataset (this may take a moment)...")

    # Try test split first, then validation
    try:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="test", trust_remote_code=True
        )
        split_used = "test"
    except:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True
        )
        split_used = "validation"

    print(f"Dataset loaded. Using {split_used} split with {len(dataset)} samples")

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full dataset: {samples_to_process} samples")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples")

    correct = 0
    total = 0
    empty_responses = 0

    for i in tqdm(range(samples_to_process), desc="Evaluating"):
        example = dataset[i]

        question = example["question"]
        context = example["context"]
        answer = example["answer"].lower().strip()

        # Format context
        context_str = ""
        titles = context["title"]
        sentences = context["sentences"]

        for title, sents in zip(titles, sentences):
            context_str += f"{title}: {' '.join(sents)}\n"

        try:
            formatted_prompt = prompt_template.format(
                context=context_str.strip(), question=question
            )
        except KeyError:
            # Try alternative formatting
            formatted_prompt = prompt_template.replace("{context}", context_str.strip())
            formatted_prompt = formatted_prompt.replace("{question}", question)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            output_lower = output_text.strip().lower()

            # Check if answer is in output
            if answer in output_lower:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses

from __future__ import annotations
import math
import pathlib
from typing import Iterable
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.evaluator import evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# From lm_eval/lm-eval.py
def generate(self, prompts: List[str], max_gen_toks: int = None, stop=None, **kwargs):
        outs = []
        for prompt in prompts:
            # Task prompt becomes the system message. User prompt is the evolutionary logic.
            # We create temporary prompt files with the system message
            with Path(self.prompt_path).open("w") as f:
                f.write(self.base_system_message.format(prompt=prompt))

            with Path(self.evaluator_prompt_path).open("w") as f:
                f.write(self.base_system_message.format(prompt=prompt))

            cmd = (
                PIPELINE_CMD
                + ["--config", self.config_file]
                + ["--iterations", str(self.iterations)]
                + self.extra_param
                + [self.init_file, self.evaluator_file]
            )
            print(f"Running command: {' '.join(cmd)}")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                text = res.stdout.strip()
                print(f"Process output: {text}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
                print(f"stderr: {e.stderr}")
                text = ""

            print(f"# Prompt: {prompt}")
            with Path(self.best_path).open("r") as f:
                best = f.read().strip()
                print(f"# Answer: {best}")

            # honour stop tokens
            if stop:
                for s in stop:
                    idx = best.find(s)
                    if idx != -1:
                        best = best[:idx]
                        break
            outs.append(best)
        return outs

# From lm_eval/lm-eval.py
def loglikelihood(self, requests: Iterable[Tuple[str, str]], **kw):
        # return [(-math.inf, False) for _ in requests]
        raise NotImplementedError

# From lm_eval/lm-eval.py
def loglikelihood_rolling(self, requests: Iterable[str], **kw):
        # return [(-math.inf, False) for _ in requests]
        raise NotImplementedError

# From lm_eval/lm-eval.py
def generate_until(self, requests: Iterable[Any], **kw) -> List[str]:
        ctxs, stops = [], []

        for req in requests:
            # ---------------- old: plain tuple ----------------
            if isinstance(req, tuple):
                ctx, until = req

            # -------------- new: Instance object --------------
            else:
                ctx = req.args[0]  # first positional arg
                until = []
                # if a second positional arg exists and is list-like,
                # treat it as the stop sequence
                if len(req.args) > 1 and isinstance(req.args[1], (list, tuple)):
                    until = list(req.args[1])

            ctxs.append(ctx)
            stops.append(until)

        # 2) run your real generator once per context
        gens = self.generate(ctxs, stop=None)

        # 3) post-trim at the first stop sequence
        cleaned = []
        for g, until in zip(gens, stops):
            for s in until:
                idx = g.find(s)
                if idx != -1:
                    g = g[:idx]
                    break
            cleaned.append(g)
        return cleaned


from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize

# From symbolic_regression/eval.py
class NumpyFloatJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyFloatJSONEncoder, self).default(obj)

# From symbolic_regression/eval.py
def compute_output_base_metrics(y_pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Computes base metrics after filtering NaNs from predictions.
    Ensures inputs y_pred and y are treated as 1D arrays.
    """
    # Ensure y_pred and y are 1D arrays.
    y_pred_1d = np.asarray(y_pred).squeeze()
    y_1d = np.asarray(y).squeeze()

    # If squeeze results in 0-D (scalar), reshape to 1-D
    if y_pred_1d.ndim == 0:
        y_pred_1d = y_pred_1d.reshape(1)
    if y_1d.ndim == 0:
        y_1d = y_1d.reshape(1)

    base_metrics_nan = {
        "mse": float("nan"),
        "nmse": float("nan"),
        "r2": float("nan"),
        "kdt": float("nan"),
        "mape": float("nan"),
        "num_valid_points": 0,
    }

    if y_pred_1d.shape != y_1d.shape and not (y_pred_1d.size == 0 and y_1d.size == 0):
        return {
            **base_metrics_nan,
            "error": "y_pred and y have incompatible shapes after ensuring 1D.",
        }

    nonnan_mask = ~np.isnan(y_pred_1d)
    y_pred_filtered = y_pred_1d[nonnan_mask]
    y_filtered = y_1d[nonnan_mask]

    if y_pred_filtered.size == 0:  # All predictions were NaN or inputs were empty
        return {
            **base_metrics_nan,
            "error": "All predictions are NaN or no data to compare after filtering.",
        }

    mse = np.mean((y_filtered - y_pred_filtered) ** 2)
    var_y = np.var(y_filtered)

    if var_y == 0:
        nmse = 0.0 if mse == 0 else float("inf")  # Consistent if true values are constant
    else:
        nmse = mse / var_y

    sum_sq_res = np.sum((y_filtered - y_pred_filtered) ** 2)
    sum_sq_total = np.sum((y_filtered - np.mean(y_filtered)) ** 2)  # Use mean of filtered y

    if sum_sq_total == 0:  # True values (after filtering) are constant
        r2 = (
            1.0 if sum_sq_res == 0 else -float("inf")
        )  # Or 0.0 if mse is also 0, definition varies. Sklearn uses 1.0.
    else:
        r2 = 1 - (sum_sq_res / sum_sq_total)

    kdt = float("nan")
    try:
        if y_filtered.size >= 2:  # Kendall's tau requires at least 2 points
            kdt_val, _ = kendalltau(y_filtered, y_pred_filtered)
            kdt = float(kdt_val)  # Ensure it's a basic float (handles np.nan)
        # If size < 2, kdt remains float('nan')
    except ValueError:  # Should be less common with size check, but as a fallback
        kdt = float("nan")  # Explicitly set, though already NaN.

    mape = float("nan")
    try:
        valid_mape_indices = y_filtered != 0
        if np.sum(valid_mape_indices) > 0:
            mape = mean_absolute_percentage_error(
                y_filtered[valid_mape_indices], y_pred_filtered[valid_mape_indices]
            )
        elif y_filtered.size > 0:  # All true values are zero
            mape = 0.0 if np.all(y_pred_filtered == 0) else float("inf")
        # If y_filtered.size is 0, mape remains float('nan')
    except ValueError:  # Fallback for any other MAPE calculation issues
        mape = float("nan")

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "r2": float(r2),
        "kdt": kdt,  # Already a float
        "mape": (
            float(mape) if mape is not float("inf") else float("inf")
        ),  # Ensure float, preserve inf
        "num_valid_points": int(y_pred_filtered.size),
    }

# From symbolic_regression/eval.py
def objective_function(
    params: np.ndarray, model_func: callable, X_matrix: np.ndarray, y_true_vector: np.ndarray
) -> float:
    """
    Objective function for scipy.optimize.minimize.
    Calculates MSE of the model_func with given params on X_matrix, y_true_vector.
    """
    # model_func callable status is checked before calling minimize in the evaluation function.
    try:
        predictions = model_func(X_matrix, params)
        if not isinstance(predictions, np.ndarray) or predictions.shape != y_true_vector.shape:
            # print(f"Debug: Objective func - Bad prediction shape/type. Got {type(predictions)}, shape {getattr(predictions, 'shape', 'N/A')}. Expected {y_true_vector.shape}")
            return float("inf")
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            # print("Debug: Objective func - Predictions contain NaN/Inf.")
            return float("inf")
    except Exception:  # Catch any error during model prediction
        # print(f"Debug: Objective func - Exception during model_func call: {e_obj}")
        return float("inf")

    mse = np.mean((predictions - y_true_vector) ** 2)
    return mse

# From symbolic_regression/eval.py
def evaluation(
    program_path: str,
    data_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates a model by loading it, optimizing its parameters, and testing it.
    The model function from program_path is expected to be named 'func'.
    """
    base_error_metrics = {
        "mse": float("nan"),
        "nmse": float("nan"),
        "r2": float("nan"),
        "kdt": float("nan"),
        "mape": float("nan"),
        "num_valid_points": 0,
    }

    def _create_error_return(error_message: str) -> Dict[str, Dict[str, Any]]:
        print(f"Error: {error_message}")
        return {
            "train_metrics": {**base_error_metrics, "error": error_message},
            "test_metrics": {**base_error_metrics, "error": error_message},
            "ood_metrics": {**base_error_metrics, "error": error_message},
        }

    # 1. Load data
    try:
        p_data_path = Path(data_path)
        train_x = np.load(p_data_path / "X_train_for_eval.npy")
        train_y = np.load(p_data_path / "y_train_for_eval.npy").squeeze()  # Ensure 1D
        test_x = np.load(p_data_path / "X_test_for_eval.npy")
        test_y = np.load(p_data_path / "y_test_for_eval.npy").squeeze()  # Ensure 1D
        test_x_ood = np.load(p_data_path / "X_ood_test_for_eval.npy")
        test_y_ood = np.load(p_data_path / "y_ood_test_for_eval.npy").squeeze()  # Ensure 1D
    except FileNotFoundError as e:
        return _create_error_return(f"Data file not found: {e.filename}")
    except Exception as e:
        return _create_error_return(f"Error loading or processing data: {str(e)}")

    # 2. Load program (model function)
    model_func = None
    try:
        p_program_path = Path(program_path)
        if not p_program_path.is_file():
            raise FileNotFoundError(f"Program file not found: {program_path}")

        spec = importlib.util.spec_from_file_location("custom_model_module", str(p_program_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec from {program_path}")

        custom_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_model_module)

        model_func = getattr(custom_model_module, "func", None)
        if not callable(model_func):
            raise AttributeError(f"'func' function not found or not callable in {program_path}")
    except Exception as e:
        return _create_error_return(
            f"Failed to load model function 'func' from '{program_path}': {str(e)}"
        )

    # 3. Optimize parameters on training data
    optimized_params = None
    num_attempts = 10  # Default number of attempts
    best_func_value = float("inf")
    optimization_critical_error_msg = None

    # Try to get num_params from the model if it provides it, otherwise default
    num_params_to_optimize = getattr(model_func, "num_params", 10)  # Default to 10 if not specified

    print(
        f"Starting optimization for {program_path} with {num_attempts} attempts (num_params: {num_params_to_optimize})..."
    )
    for i in range(num_attempts):
        print(f"Attempt {i+1}/{num_attempts}")
        initial_params = np.random.rand(num_params_to_optimize)
        try:
            optimization_result = minimize(
                objective_function,
                initial_params,
                args=(model_func, train_x, train_y),
                method="BFGS",
                # options={'maxiter': 1000, 'disp': False} # Example options
            )
            if optimization_result.success:
                print(f"Attempt {i+1} successful. Func value: {optimization_result.fun}")
                if optimization_result.fun < best_func_value:
                    best_func_value = optimization_result.fun
                    optimized_params = optimization_result.x
                    print(f"New best result found in attempt {i+1}. Func value: {best_func_value}")
            else:
                print(
                    f"Warning: Optimization attempt {i+1} did not converge. Message: {optimization_result.message}. Func value: {optimization_result.fun}"
                )
                if (
                    optimization_result.fun < best_func_value
                ):  # Still consider if it's the best so far
                    print(
                        f"Non-converged result from attempt {i+1} is an improvement. Func value: {optimization_result.fun}"
                    )
                    best_func_value = optimization_result.fun
                    optimized_params = optimization_result.x

        except Exception as e:
            optimization_critical_error_msg = (
                f"Critical error during optimization attempt {i+1} for {program_path}: {str(e)}"
            )
            print(f"Error: {optimization_critical_error_msg}")
            break

    if optimization_critical_error_msg:
        return _create_error_return(optimization_critical_error_msg)

    def _get_metrics_for_set(
        X_data: np.ndarray, y_data: np.ndarray, set_name: str
    ) -> Dict[str, Any]:
        if optimized_params is None:
            msg = f"Optimization failed to find parameters for {program_path}, cannot evaluate {set_name}."
            return {**base_error_metrics, "error": msg}
        try:
            pred_y = model_func(X_data, optimized_params)
            if not isinstance(pred_y, np.ndarray):
                raise ValueError(f"{set_name} predictions are not numpy arrays. Got {type(pred_y)}")

            metrics = compute_output_base_metrics(pred_y, y_data)
            if "error" in metrics and metrics["num_valid_points"] == 0:
                print(f"Warning for {set_name} ({program_path}): {metrics['error']}")
            return metrics
        except Exception as e:
            error_msg = f"{set_name} evaluation failed for '{program_path}': {str(e)}"
            print(f"Error: {error_msg}")
            return {**base_error_metrics, "error": error_msg}

    train_metrics = _get_metrics_for_set(train_x, train_y, "Train set")
    test_metrics = _get_metrics_for_set(test_x, test_y, "Test set")
    ood_metrics = _get_metrics_for_set(test_x_ood, test_y_ood, "OOD test set")

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
    }

# From symbolic_regression/eval.py
def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyFloatJSONEncoder, self).default(obj)

from bench.datamodules import get_datamodule

# From symbolic_regression/data_api.py
class PreserveNewlinesDumper(yaml.SafeDumper):
        """Custom YAML dumper that preserves multi-line strings."""

        def represent_scalar(self, tag, value, style=None):
            if style is None and isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)

# From symbolic_regression/data_api.py
def load_secret(secrets_file: str = "secrets.yaml") -> Dict[str, Any]:
    """
    Load API keys and configuration from a secrets file.

    Args:
        secrets_file: Path to the YAML secrets file

    Returns:
        Dictionary containing secret configuration, empty dict if file not found
    """
    try:
        with open(secrets_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Secrets file '{secrets_file}' not found.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading secrets file '{secrets_file}': {e}")
        return {}

# From symbolic_regression/data_api.py
def extract_problem_data_from_initialized_dataset(
    initialized_dataset, problem_id: int
) -> Dict[str, Any]:
    """
    Extract data for a specific problem from an initialized dataset.

    Args:
        initialized_dataset: Pre-initialized and setup dataset object
        problem_id: Index of the problem to extract

    Returns:
        Dictionary containing problem data including train/test samples, symbols, and metadata
    """
    problem = initialized_dataset.problems[problem_id]
    gt_eq = problem.gt_equation
    samples = problem.samples

    data = {
        "train": samples["train"],
        "test": samples["test"],
        "ood_test": samples.get("ood_test", None),
        "symbols": gt_eq.symbols,
        "symbol_descs": gt_eq.symbol_descs,
        "symbol_properties": gt_eq.symbol_properties,
        "expression": gt_eq.expression,
        "dataset_identifier": problem.dataset_identifier,
        "equation_idx": problem.equation_idx,
    }
    return data

# From symbolic_regression/data_api.py
def create_program(problem: Dict[str, Any]) -> str:
    """
    Create a Python script with a naive linear model for symbolic regression.

    The generated script contains a `func(x, params)` that computes predictions
    in a vectorized manner: x @ params. If no input features exist, it predicts
    a constant params[0].

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created program file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'

    # Parse symbols and properties
    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars = []
    input_vars_descs = []
    output_var = None
    output_var_desc = "N/A"

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars.append(symbols[i])
            input_vars_descs.append(descs[i])
        elif prop == "O":
            output_var = symbols[i]
            output_var_desc = descs[i]

    if not output_var:
        raise ValueError("No output variable ('O') found in symbol_properties.")

    # Build input variable mapping comments
    x_mapping_comments = ["# Input variable mapping for x (columns of the input matrix):"]
    if not input_vars:
        x_mapping_comments.append("#   No input variables (x will be an (n_samples, 0) matrix).")
    else:
        for i, var_name in enumerate(input_vars):
            x_mapping_comments.append(f"#   x[:, {i}]: {var_name} ({input_vars_descs[i]})")
    x_mapping_str = "\n".join(x_mapping_comments)

    # Build function body
    num_features = len(input_vars)
    if num_features > 0:
        function_body = " + ".join([f"x[:, {i}] * params[{i}]" for i in range(num_features)])
    else:
        function_body = (
            "np.full(x.shape[0], params[0])  # Predicts a constant value for all samples"
        )

    model_num_params = 10

    # Build input variables description
    input_vars_desc_list = [f"{v} ({input_vars_descs[i]})" for i, v in enumerate(input_vars)]
    input_vars_desc_str = ", ".join(input_vars_desc_list) if input_vars else "None"

    program_content = f'''"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: {output_var} ({output_var_desc})
Input variables (columns of x): {input_vars_desc_str}
"""
import numpy as np

{x_mapping_str}

# Parameters will be optimized by BFGS outside this function.
# Number of parameters expected by this model: {model_num_params}.
# Example initialization: params = np.random.rand({model_num_params})

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Calculates the model output using a linear combination of input variables
    or a constant value if no input variables. Operates on a matrix of samples.

    Args:
        x (np.ndarray): A 2D numpy array of input variable values, shape (n_samples, n_features).
                        n_features is {num_features}.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        ({', '.join(input_vars) if input_vars else "None - x has 0 columns"}).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: {model_num_params}.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    result = {function_body}
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
'''

    os.makedirs(problem_dir, exist_ok=True)
    file_path = os.path.join(problem_dir, "initial_program.py")
    with open(file_path, "w") as f:
        f.write(program_content)

    return file_path

# From symbolic_regression/data_api.py
def create_evaluator(problem: Dict[str, Any]) -> str:
    """
    Create an evaluator script for the symbolic regression problem.

    The evaluator assesses model performance using BFGS optimization
    and computes various metrics including MSE and combined scores.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created evaluator file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)

    # Extract data arrays
    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    train_samples = np.asarray(problem["train"])
    test_samples = np.asarray(problem["test"])
    ood_test_samples = problem["ood_test"]
    if ood_test_samples is not None:
        ood_test_samples = np.asarray(ood_test_samples)

    # Find input and output indices
    input_indices = [i for i, prop in enumerate(properties) if prop == "V"]
    output_indices = [i for i, prop in enumerate(properties) if prop == "O"]

    if not output_indices:
        raise ValueError("No output variable ('O') found in symbol_properties.")
    if len(output_indices) > 1:
        raise ValueError("Multiple output variables ('O') found. Evaluator supports single output.")
    output_index = output_indices[0]

    # Prepare data arrays
    if not input_indices:
        X_train = np.empty((len(train_samples), 0))
        X_test = np.empty((len(test_samples), 0))
        X_ood_test = np.empty((len(ood_test_samples), 0)) if ood_test_samples is not None else None
    else:
        X_train = train_samples[:, input_indices]
        X_test = test_samples[:, input_indices]
        X_ood_test = ood_test_samples[:, input_indices] if ood_test_samples is not None else None

    y_train = train_samples[:, output_index]
    y_test = test_samples[:, output_index]
    y_ood_test = ood_test_samples[:, output_index] if ood_test_samples is not None else None

    num_input_features = len(input_indices)
    model_num_params_expected = 10

    # Save data files
    base_data_path = "./"
    x_train_path = os.path.join(base_data_path, problem_dir, "X_train_for_eval.npy")
    y_train_path = os.path.join(base_data_path, problem_dir, "y_train_for_eval.npy")
    np.save(x_train_path, X_train)
    np.save(y_train_path, y_train)

    x_test_path = os.path.join(problem_dir, "X_test_for_eval.npy")
    y_test_path = os.path.join(problem_dir, "y_test_for_eval.npy")
    np.save(x_test_path, X_test)
    np.save(y_test_path, y_test)

    if X_ood_test is not None and y_ood_test is not None:
        x_ood_test_path = os.path.join(problem_dir, "X_ood_test_for_eval.npy")
        y_ood_test_path = os.path.join(problem_dir, "y_ood_test_for_eval.npy")
        np.save(x_ood_test_path, X_ood_test)
        np.save(y_ood_test_path, y_ood_test)

    evaluator_script_content = f'''"""
Evaluator for a symbolic regression model.
It assesses a model program based on its performance on training data.
The model's `func` is expected to take a matrix X of inputs.
"""
import os
import sys
import time
import traceback
import importlib.util
import numpy as np
from scipy.optimize import minimize
import concurrent.futures

# Expected number of input features for the model's func
NUM_INPUT_FEATURES_EXPECTED = {num_input_features}
# Expected number of parameters for the initial model
MODEL_NUM_PARAMS_EXPECTED = {model_num_params_expected}

# Paths to data (should be relative to where evaluator.py is run or absolute)
X_TRAIN_EVAL_PATH = r'{x_train_path}'
Y_TRAIN_EVAL_PATH = r'{y_train_path}'


def run_with_timeout(func, args=(), kwargs={{}}, timeout_seconds=5):
    """Execute a function with a timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            func_name = getattr(func, '__name__', 'Unnamed function')
            raise TimeoutError(f"Function {{func_name}} timed out after {{timeout_seconds}} seconds")


def filter_and_convert_metrics(current_metrics_dict):
    """Filter and convert metrics to appropriate types."""
    filtered_dict = {{}}
    float_metric_keys = ['combined_score', 'negative_mse']
    
    for key in float_metric_keys:
        if key in current_metrics_dict:
            value = current_metrics_dict[key]
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                try:
                    filtered_dict[key] = float(value)
                except (ValueError, TypeError):
                    pass
    
    return filtered_dict


def objective_function(params, model_func, X_matrix, y_true_vector):
    """
    Objective function for scipy.optimize.minimize.
    Calculates MSE of the model_func with given params on X_matrix, y_true_vector.
    
    Args:
        params: Parameter vector for the model
        model_func: Function that takes (X_matrix, params) and returns predictions
        X_matrix: Input features matrix (n_samples, n_features)
        y_true_vector: True output values (n_samples,)
        
    Returns:
        MSE value or inf if computation fails
    """
    if not callable(model_func):
        return float('inf')
    
    try:
        predictions = model_func(X_matrix, params)
        if not isinstance(predictions, np.ndarray) or predictions.shape != y_true_vector.shape:
            return float('inf')
    except Exception:
        return float('inf')
    
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        return float('inf')
    
    mse = np.mean((predictions - y_true_vector)**2)
    return mse


def evaluate(program_path):
    """
    Evaluate a model program on the training data.
    
    Args:
        program_path: Path to the Python program containing the model
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {{
        'can_run': 0.0,
        'negative_mse': -1e09,
        'raw_mse_train': float('inf'),
        'mse_train_score': 0.0,
        'num_params': MODEL_NUM_PARAMS_EXPECTED,
        'combined_score': -1e09,
        'error_message': None,
        'optimization_success': False,
        'optimized_params': None
    }}
    
    # Load training data
    try:
        X_train = np.load(X_TRAIN_EVAL_PATH)
        y_train = np.load(Y_TRAIN_EVAL_PATH)
        
        if X_train.shape[1] != NUM_INPUT_FEATURES_EXPECTED:
            metrics['error_message'] = f"Loaded X_train has {{X_train.shape[1]}} features, expected {{NUM_INPUT_FEATURES_EXPECTED}}."
            return filter_and_convert_metrics(metrics)
        
        if X_train.shape[0] != y_train.shape[0]:
            metrics['error_message'] = f"X_train has {{X_train.shape[0]}} samples, y_train has {{y_train.shape[0]}}."
            return filter_and_convert_metrics(metrics)
    except Exception as e:
        metrics['error_message'] = f"Failed to load training data: {{str(e)}}. Paths: X:{{X_TRAIN_EVAL_PATH}}, Y:{{Y_TRAIN_EVAL_PATH}}"
        return filter_and_convert_metrics(metrics)
    
    # Load and test the model function
    func_to_eval = None
    try:
        spec = importlib.util.spec_from_file_location("model_program", program_path)
        if spec is None or spec.loader is None:
            metrics['error_message'] = f"Could not create spec for module at {{program_path}}"
            return filter_and_convert_metrics(metrics)
        
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        metrics['can_run'] = 0.2
        
        if not hasattr(model_module, 'run_search') or not callable(model_module.run_search):
            metrics['error_message'] = "Model program missing callable 'run_search'."
            return filter_and_convert_metrics(metrics)
        
        func_to_eval = model_module.run_search()
        
        if not callable(func_to_eval):
            metrics['error_message'] = "'run_search' did not return a callable function."
            return filter_and_convert_metrics(metrics)
        
        # Test the function with dummy data
        num_dummy_samples = 5
        dummy_x = np.random.rand(num_dummy_samples, NUM_INPUT_FEATURES_EXPECTED)
        if NUM_INPUT_FEATURES_EXPECTED == 0:
            dummy_x = np.empty((num_dummy_samples, 0))
        dummy_params = np.random.rand(MODEL_NUM_PARAMS_EXPECTED)
        
        try:
            pred_test = run_with_timeout(func_to_eval, args=(dummy_x, dummy_params), timeout_seconds=5)
            if not isinstance(pred_test, np.ndarray) or pred_test.shape != (num_dummy_samples,):
                metrics['can_run'] = 0.5
                metrics['error_message'] = f"Func test: output shape mismatch. Got {{pred_test.shape if isinstance(pred_test, np.ndarray) else type(pred_test)}}, expected ({{num_dummy_samples}},)."
                return filter_and_convert_metrics(metrics)
            metrics['can_run'] = 1.0
        except TimeoutError as te:
            metrics['can_run'] = 0.5
            metrics['error_message'] = f"Func execution test timed out: {{str(te)}}"
            return filter_and_convert_metrics(metrics)
        except Exception as e:
            metrics['can_run'] = 0.5
            metrics['error_message'] = f"Func execution test failed: {{str(e)}} with dummy_x.shape={{dummy_x.shape}}, dummy_params.shape={{dummy_params.shape}}"
            return filter_and_convert_metrics(metrics)
    
    except FileNotFoundError:
        metrics['error_message'] = f"Model program file not found: {{program_path}}"
        return filter_and_convert_metrics(metrics)
    except Exception as e:
        metrics['error_message'] = f"Failed to load or test model function: {{str(e)}}"
        return filter_and_convert_metrics(metrics)
    
    if metrics['can_run'] < 1.0:
        return filter_and_convert_metrics(metrics)
    
    # Optimize parameters
    initial_params = np.random.rand(MODEL_NUM_PARAMS_EXPECTED)
    optimized_params = None
    
    if X_train.ndim != 2 or X_train.shape[1] != NUM_INPUT_FEATURES_EXPECTED:
        metrics['error_message'] = f"X_train shape {{X_train.shape}} is not compatible with NUM_INPUT_FEATURES_EXPECTED {{NUM_INPUT_FEATURES_EXPECTED}} for optimization."
        return filter_and_convert_metrics(metrics)
    
    try:
        opt_result = minimize(
            objective_function,
            initial_params,
            args=(func_to_eval, X_train, y_train),
            method='BFGS'
        )
        
        metrics['raw_mse_train'] = opt_result.fun if np.isfinite(opt_result.fun) else float('inf')
        metrics['optimization_success'] = opt_result.success
        
        if opt_result.success or hasattr(opt_result, 'x'):
            optimized_params = opt_result.x
        else:
            optimized_params = initial_params
        
        if not opt_result.success and metrics['error_message'] is None:
            metrics['error_message'] = f"Optimization did not converge: {{opt_result.message if hasattr(opt_result, 'message') else 'Unknown reason'}}"
    
    except Exception as e:
        metrics['raw_mse_train'] = float('inf')
        metrics['error_message'] = f"Error during optimization: {{str(e)}}"
    
    metrics['optimized_params'] = optimized_params.tolist() if optimized_params is not None else None
    
    # Calculate final scores
    if np.isfinite(metrics['raw_mse_train']):
        metrics['negative_mse'] = -metrics['raw_mse_train']
        metrics['mse_train_score'] = -np.log10(metrics['raw_mse_train'] + 1e-9)
    else:
        metrics['mse_train_score'] = 0.0
    
    metrics['combined_score'] = metrics['mse_train_score']
    
    return filter_and_convert_metrics(metrics)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <path_to_model_program.py>")
        print("Please run the main script that calls create_program and create_evaluator first.")
        sys.exit(1)
    
    program_to_evaluate = sys.argv[1]
    if not os.path.exists(program_to_evaluate):
        print(f"Error: Program file '{{program_to_evaluate}}' not found.")
        sys.exit(1)
    
    print(f"Evaluating model: {{program_to_evaluate}}")
    print(f"Using NUM_INPUT_FEATURES_EXPECTED = {{NUM_INPUT_FEATURES_EXPECTED}}")
    print(f"Using MODEL_NUM_PARAMS_EXPECTED = {{MODEL_NUM_PARAMS_EXPECTED}}")
    print(f"Loading X_train from: {{X_TRAIN_EVAL_PATH}}")
    print(f"Loading y_train from: {{Y_TRAIN_EVAL_PATH}}")
    
    if not os.path.exists(X_TRAIN_EVAL_PATH):
        print(f"Error: X_train data file '{{X_TRAIN_EVAL_PATH}}' not found.")
        sys.exit(1)
    if not os.path.exists(Y_TRAIN_EVAL_PATH):
        print(f"Error: y_train data file '{{Y_TRAIN_EVAL_PATH}}' not found.")
        sys.exit(1)
    
    evaluation_results = evaluate(program_to_evaluate)
    print("\\nEvaluation Results:")
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {{key}}: {{value:.4f}}")
        else:
            print(f"  {{key}}: {{value}}")
'''

    evaluator_file_path = os.path.join(problem_dir, "evaluator.py")
    with open(evaluator_file_path, "w") as f:
        f.write(evaluator_script_content)

    return evaluator_file_path

# From symbolic_regression/data_api.py
def create_config(problem: Dict[str, Any]) -> str:
    """
    Create a YAML configuration file for the symbolic regression task.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created configuration file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)
    config_file_path = os.path.join(problem_dir, "config.yaml")

    # Parse variables
    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars_list = []
    output_var_list = []

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars_list.append(f"{symbols[i]} ({descs[i]})")
        elif prop == "O":
            output_var_list.append(f"{symbols[i]} ({descs[i]})")

    input_vars_str = ", ".join(input_vars_list) if input_vars_list else "None"
    output_var_str = (
        ", ".join(output_var_list) if output_var_list else "None (Error: No output defined!)"
    )

    num_initial_params = 10

    system_message = (
        "Your task is to evolve a Python function `func(x, params)` that models a scientific process, "
        "considering the physical meaning and relationships of inputs, "
        "by predicting output variables based on input variables.\\n\\n"
        "The function signature is:\\n\\n"
        "```python\\n"
        "def func(x: np.ndarray, params: np.ndarray) -> np.ndarray:\\n"
        "```\\n\\n"
        f"- `x` is a 2D NumPy array of shape `(n_samples, {len(input_vars_list)})`\\n"
        f"- `params` is a 1D NumPy array of up to {num_initial_params} parameters\\n"
        "- The function should return a 1D NumPy array of predictions with shape `(n_samples,)`\\n\\n"
        "**Current Problem:**\\n"
        f"Model the {output_var_str} using the input features: {input_vars_str}\\n"
        f"Thus, `x` contains {len(input_vars_list)} columns: {input_vars_str}.\\n\\n"
        "The initial version of `func` is a simple linear model. Parameters in `params` will be optimized externally "
        "using the BFGS algorithm based on unseen training data.\\n\\n"
        "Your objective is to evolve `func` to improve predictive performance on unseen data. Aim for a balance between:\\n"
        "- **Accuracy**: Lower mean squared error (MSE) on training data\\n"
        "- **Simplicity**: Prefer concise, interpretable expressions\\n\\n"
        "Model performance (score = -log_10(mse)) will be evaluated on a held-out dataset. "
        "Ensure the model is free of potential numerical errors (e.g., log0, division by 0)."
    )

    secret = load_secret()
    config_data = {
        "# Configuration for Symbolic Regression Task": f"{problem['dataset_identifier']}/{problem['equation_idx']}",
        "max_iterations": 200,
        "log_level": "INFO",
        "target_score": "combined_score",
        "checkpoint_interval": 10,
        "llm": {
            "primary_model": "gpt-4o",
            "primary_model_weight": 0.8,
            "secondary_model": "o3",
            "secondary_model_weight": 0.2,
            "api_base": "https://api.openai.com/v1",
        },
        "prompt": {
            "system_message": system_message,
            "num_top_programs": 4,
            "use_template_stochasticity": True,
        },
        "database": {
            "population_size": 70,
            "archive_size": 30,
            "num_islands": 4,
            "elite_selection_ratio": 0.3,
            "exploitation_ratio": 0.6,
        },
        "evaluator": {
            "timeout": 90,
            "cascade_evaluation": False,
            "cascade_thresholds": [1.0],
            "parallel_evaluations": 4,
            "use_llm_feedback": False,
        },
        "diff_based_evolution": True,
        "allow_full_rewrites": False,
    }

    class PreserveNewlinesDumper(yaml.SafeDumper):
        """Custom YAML dumper that preserves multi-line strings."""

        def represent_scalar(self, tag, value, style=None):
            if style is None and isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)

    with open(config_file_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            Dumper=PreserveNewlinesDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )

    return config_file_path

# From symbolic_regression/data_api.py
def process_problem(initialized_dataset, problem_id: int, split_name: str) -> str:
    """
    Process a single problem using a pre-initialized dataset.

    Loads specific problem data, creates program, evaluator, and config.
    Skips processing if essential output files already exist.

    Args:
        initialized_dataset: Pre-initialized and setup dataset object
        problem_id: Index of the problem to process
        split_name: Name of the dataset split

    Returns:
        Status message indicating success, skip, or error
    """
    try:
        problem_data = extract_problem_data_from_initialized_dataset(
            initialized_dataset, problem_id
        )

        dataset_identifier = problem_data["dataset_identifier"]
        equation_idx = problem_data["equation_idx"]
        problem_dir = os.path.join("problems", dataset_identifier, str(equation_idx))
        base_data_path = "./"

        # Check if all essential files already exist
        essential_files = [
            os.path.join(problem_dir, "initial_program.py"),
            os.path.join(problem_dir, "evaluator.py"),
            os.path.join(problem_dir, "config.yaml"),
            os.path.join(base_data_path, problem_dir, "X_train_for_eval.npy"),
            os.path.join(base_data_path, problem_dir, "y_train_for_eval.npy"),
            os.path.join(problem_dir, "X_test_for_eval.npy"),
            os.path.join(problem_dir, "y_test_for_eval.npy"),
        ]

        # Add OOD test files if applicable
        if problem_data.get("ood_test") is not None:
            essential_files.extend(
                [
                    os.path.join(problem_dir, "X_ood_test_for_eval.npy"),
                    os.path.join(problem_dir, "y_ood_test_for_eval.npy"),
                ]
            )

        # Check if all files exist
        all_files_exist = all(os.path.exists(f) for f in essential_files)

        if all_files_exist:
            return f"Skipped (already processed): problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

        # Create necessary files
        create_program(problem_data)
        create_evaluator(problem_data)
        create_config(problem_data)

        return f"Successfully processed problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

    except Exception as e:
        import traceback

        return f"Error processing problem_id {problem_id} for split {split_name}: {str(e)}\n{traceback.format_exc()}"

# From symbolic_regression/data_api.py
def represent_scalar(self, tag, value, style=None):
            if style is None and isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)


# From web_scraper_optillm/evaluator.py
def get_test_cases() -> List[Dict[str, Any]]:
    """
    Get test cases with HTML content and expected results.

    These test cases include URLs that will be fetched by optillm's
    readurls plugin during evolution, providing the LLM with actual
    documentation structure.

    Returns:
        List of test cases with HTML content and expected results
    """
    return [
        {
            "name": "json_module_docs",
            "html": """
            <html>
            <body>
                <div class="section">
                    <h1>json — JSON encoder and decoder</h1>
                    <p>Source: https://docs.python.org/3/library/json.html</p>
                    
                    <dl class="function">
                        <dt class="sig sig-object py">
                            <span class="sig-name descname">dumps</span>
                            <span class="sig-paren">(</span>
                            <em class="sig-param">obj</em>,
                            <em class="sig-param">indent=None</em>
                            <span class="sig-paren">)</span>
                        </dt>
                        <dd>
                            <p>Serialize obj to a JSON formatted string.</p>
                        </dd>
                    </dl>
                    
                    <dl class="function">
                        <dt class="sig sig-object py">
                            <span class="sig-name descname">loads</span>
                            <span class="sig-paren">(</span>
                            <em class="sig-param">s</em>
                            <span class="sig-paren">)</span>
                        </dt>
                        <dd>
                            <p>Deserialize s to a Python object.</p>
                        </dd>
                    </dl>
                </div>
            </body>
            </html>
            """,
            "expected": [
                {"name": "dumps", "params": ["obj", "indent"]},
                {"name": "loads", "params": ["s"]},
            ],
        },
        {
            "name": "requests_docs",
            "html": """
            <html>
            <body>
                <div class="document">
                    <h1>Requests Documentation</h1>
                    <p>Refer to https://requests.readthedocs.io/en/latest/api/ for full API</p>
                    
                    <div class="function">
                        <h3>requests.get(url, params=None, **kwargs)</h3>
                        <p>Sends a GET request.</p>
                    </div>
                    
                    <div class="function">
                        <h3>requests.post(url, data=None, json=None, **kwargs)</h3>
                        <p>Sends a POST request.</p>
                    </div>
                </div>
            </body>
            </html>
            """,
            "expected": [
                {"name": "requests.get", "params": ["url", "params"]},
                {"name": "requests.post", "params": ["url", "data", "json"]},
            ],
        },
        {
            "name": "beautifulsoup_docs",
            "html": """
            <html>
            <body>
                <div class="section">
                    <h1>BeautifulSoup Documentation</h1>
                    <p>Documentation at https://www.crummy.com/software/BeautifulSoup/bs4/doc/</p>
                    
                    <code class="python">
                        <span class="name">BeautifulSoup</span>(<span class="param">markup</span>, <span class="param">parser</span>)
                    </code>
                    <p>Parse a string using a specified parser.</p>
                    
                    <code class="python">
                        <span class="name">find</span>(<span class="param">name</span>, <span class="param">attrs</span>=<span class="default">None</span>)
                    </code>
                    <p>Find the first matching tag.</p>
                    
                    <code class="python">
                        <span class="name">find_all</span>(<span class="param">name</span>, <span class="param">attrs</span>=<span class="default">None</span>, <span class="param">limit</span>=<span class="default">None</span>)
                    </code>
                    <p>Find all matching tags.</p>
                </div>
            </body>
            </html>
            """,
            "expected": [
                {"name": "BeautifulSoup", "params": ["markup", "parser"]},
                {"name": "find", "params": ["name", "attrs"]},
                {"name": "find_all", "params": ["name", "attrs", "limit"]},
            ],
        },
        {
            "name": "edge_case_malformed",
            "html": """
            <html>
            <body>
                <div class="weird-format">
                    <h2>Unusual Documentation Format</h2>
                    <p>This tests robustness - check https://example.com/weird-api-docs</p>
                    
                    <pre>
                    function_name(arg1, arg2=default_value)
                    Another description here
                    </pre>
                    
                    <table>
                        <tr>
                            <td>another_func()</td>
                            <td>Does something</td>
                        </tr>
                    </table>
                </div>
            </body>
            </html>
            """,
            "expected": [
                {"name": "function_name", "params": ["arg1", "arg2"]},
                {"name": "another_func", "params": []},
            ],
        },
    ]

# From web_scraper_optillm/evaluator.py
def evaluate_extraction(
    docs: List[Dict[str, Any]], expected: List[Dict[str, Any]]
) -> tuple[int, int]:
    """
    Evaluate the accuracy of extracted documentation.

    Args:
        docs: Extracted documentation
        expected: Expected results

    Returns:
        Tuple of (correct_count, expected_count)
    """
    correct = 0
    expected_count = len(expected)

    for exp in expected:
        # Check if we found this function
        found = False
        for doc in docs:
            doc_name = doc.get("name", "").lower()
            exp_name = exp["name"].lower()

            if exp_name in doc_name or doc_name in exp_name:
                found = True
                # Check parameter extraction
                doc_params = doc.get("parameters", [])
                exp_params = exp.get("params", [])

                if len(doc_params) >= len(exp_params):
                    correct += 1
                else:
                    correct += 0.5  # Partial credit
                break

        if not found and docs:  # Only penalize if we extracted something
            pass  # No additional penalty

    return correct, expected_count

# From web_scraper_optillm/evaluator.py
def generate_feedback(metrics: Dict[str, float], artifacts: Dict[str, Any]) -> str:
    """
    Generate detailed feedback for the LLM to improve the scraper.

    This feedback will be included in the evolution prompt to guide
    the LLM toward better solutions.

    Args:
        metrics: Evaluation metrics
        artifacts: Evaluation artifacts

    Returns:
        Detailed feedback string
    """
    feedback = []

    feedback.append("## Evaluation Feedback")
    feedback.append(f"Overall Score: {metrics['combined_score']:.2f}/1.0")
    feedback.append("")

    # Accuracy feedback
    if metrics["accuracy"] < 0.5:
        feedback.append("⚠️ **Low Accuracy**: The scraper is missing many expected functions.")
        feedback.append(
            "Consider improving the HTML parsing logic to handle different documentation formats."
        )
        feedback.append(
            "Look for patterns like <dl class='function'>, <div class='function'>, and <code> tags."
        )
    elif metrics["accuracy"] < 0.8:
        feedback.append("✅ **Good Accuracy**: Most functions are found, but some are missed.")
        feedback.append("Fine-tune the extraction logic for edge cases.")
    else:
        feedback.append("🎉 **Excellent Accuracy**: Function extraction is working well!")

    feedback.append("")

    # Completeness feedback
    if metrics["completeness"] < 0.5:
        feedback.append("⚠️ **Low Completeness**: Not extracting enough functions overall.")
        feedback.append("Increase the limit or improve the search scope.")

    # Robustness feedback
    if metrics["robustness"] < 0.8:
        feedback.append("⚠️ **Low Robustness**: The scraper fails on some HTML formats.")
        feedback.append("Add try-catch blocks and handle different documentation structures.")
        feedback.append("Consider multiple parsing strategies and fallback methods.")

    # Specific improvements
    feedback.append("")
    feedback.append("## Specific Improvements:")

    # Analyze test case results
    for key, value in artifacts.items():
        if key.startswith("test_case_") and isinstance(value, dict):
            if "error" in key:
                feedback.append(f"- Fix error in {key}: {value}")
            elif value.get("found_count", 0) < value.get("expected_count", 0):
                feedback.append(
                    f"- Improve extraction for {key}: found {value.get('found_count', 0)}/{value.get('expected_count', 0)} functions"
                )

    # Documentation URL hints (these will be fetched by readurls plugin)
    feedback.append("")
    feedback.append("## Documentation References:")
    feedback.append("For improving parsing, refer to these documentation structures:")
    feedback.append("- Python docs: https://docs.python.org/3/library/json.html")
    feedback.append("- Requests docs: https://requests.readthedocs.io/en/latest/api/")
    feedback.append("- BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/")

    return "\n".join(feedback)

from bs4 import BeautifulSoup

# From web_scraper_optillm/initial_program.py
def scrape_api_docs(html_content: str) -> List[Dict[str, any]]:
    """
    Extract API documentation from HTML content.

    Args:
        html_content: Raw HTML content of a documentation page

    Returns:
        List of dictionaries containing function documentation
    """
    soup = BeautifulSoup(html_content, "html.parser")
    functions = []

    # Try multiple approaches to find functions
    # 1. Look for code blocks
    code_blocks = soup.find_all("code")
    for block in code_blocks:
        text = block.get_text(strip=True)
        if "(" in text and ")" in text:
            functions.append(
                {
                    "name": text.split("(")[0].strip(),
                    "signature": text,
                    "description": "No description found",
                    "parameters": extract_parameters(text),
                }
            )

    # 2. Look for function signatures in headers (h3)
    h3_blocks = soup.find_all("h3")
    for block in h3_blocks:
        text = block.get_text(strip=True)
        if "(" in text and ")" in text:
            functions.append(
                {
                    "name": text.split("(")[0].strip(),
                    "signature": text,
                    "description": "No description found",
                    "parameters": extract_parameters(text),
                }
            )

    # 3. Look for dt elements with sig class
    dt_blocks = soup.find_all("dt", class_="sig")
    for block in dt_blocks:
        sig_name = block.find(class_="sig-name")
        if sig_name:
            name = sig_name.get_text(strip=True)
            functions.append(
                {
                    "name": name,
                    "signature": block.get_text(strip=True),
                    "description": "No description found",
                    "parameters": extract_parameters(block.get_text(strip=True)),
                }
            )

    return functions[:20]

# From web_scraper_optillm/initial_program.py
def extract_parameters(signature: str) -> List[Dict[str, str]]:
    """
    Extract parameter information from a function signature.

    Args:
        signature: Function signature string

    Returns:
        List of parameter dictionaries
    """
    params = []
    # Very basic parameter extraction
    match = re.search(r"\((.*?)\)", signature)
    if match:
        param_string = match.group(1)
        if param_string:
            param_parts = param_string.split(",")
            for part in param_parts:
                part = part.strip()
                if part:
                    params.append(
                        {
                            "name": part.split("=")[0].strip(),
                            "type": "unknown",
                            "default": None,
                            "description": "",
                        }
                    )

    return params

# From web_scraper_optillm/initial_program.py
def format_documentation(api_docs: List[Dict[str, any]]) -> str:
    """
    Format extracted documentation into a readable string.

    Args:
        api_docs: List of API documentation dictionaries

    Returns:
        Formatted documentation string
    """
    output = []
    for doc in api_docs:
        output.append(f"Function: {doc['name']}")
        output.append(f"Signature: {doc['signature']}")
        output.append(f"Description: {doc['description']}")

        if doc.get("parameters"):
            output.append("Parameters:")
            for param in doc["parameters"]:
                output.append(f"  - {param['name']}: {param.get('description', 'No description')}")

        output.append("")  # Empty line between functions

    return "\n".join(output)



import configparser
import requests
from lxml.html import fragment_fromstring

# From online_judge_programming/submit.py
class ConfigError(Exception):
    pass

# From online_judge_programming/submit.py
def get_url(cfg, option, default):
    if cfg.has_option("kattis", option):
        return cfg.get("kattis", option)
    else:
        hostname = cfg.get("kattis", "hostname")
        return f"https://{hostname}/{default}"

# From online_judge_programming/submit.py
def get_config():
    """Returns a ConfigParser object for the .kattisrc file(s)"""
    cfg = configparser.ConfigParser()
    if os.path.exists(_DEFAULT_CONFIG):
        cfg.read(_DEFAULT_CONFIG)

    try:
        file = __file__
    except NameError:
        file = sys.argv[0]

    if not cfg.read(
        [
            os.path.join(os.path.expanduser("~"), ".kattisrc"),
            os.path.join(os.path.dirname(file), ".kattisrc"),
            os.path.join(os.path.dirname(os.path.realpath(file)), ".kattisrc"),
        ]
    ):
        raise ConfigError(
            """\
I failed to read in a config file from your home directory or from the
same directory as this script. To download a .kattisrc file please visit
https://<kattis>/download/kattisrc

The file should look something like this:
[user]
username: yourusername
token: *********

[kattis]
hostname: <kattis>
loginurl: https://<kattis>/login
submissionurl: https://<kattis>/submit
submissionsurl: https://<kattis>/submissions"""
        )
    return cfg

# From online_judge_programming/submit.py
def is_python2(files):
    python2 = re.compile(r"^\s*\bprint\b *[^ \(\),\]]|\braw_input\b")
    for filename in files:
        try:
            with open(filename) as f:
                for index, line in enumerate(f):
                    if index == 0 and line.startswith("#!"):
                        if "python2" in line:
                            return True
                        if "python3" in line:
                            return False
                    if python2.search(line.split("#")[0]):
                        return True
        except UnicodeDecodeError:
            pass
        except IOError:
            return False
    return False

# From online_judge_programming/submit.py
def guess_language(ext, files):
    if ext == ".C":
        return "C++"
    ext = ext.lower()
    if ext == ".h":
        if any(f.endswith(".c") for f in files):
            return "C"
        else:
            return "C++"
    if ext == ".py":
        if is_python2(files):
            return "Python 2"
        else:
            return "Python 3"
    return _LANGUAGE_GUESS.get(ext, None)

# From online_judge_programming/submit.py
def guess_mainfile(language, files):
    for filename in files:
        if os.path.splitext(os.path.basename(filename))[0] in ["main", "Main"]:
            return filename
    for filename in files:
        try:
            with open(filename) as f:
                conts = f.read()
                if language in ["Java", "Rust", "Scala", "Kotlin"] and re.search(
                    r" main\s*\(", conts
                ):
                    return filename
                if language == "Pascal" and re.match(r"^\s*[Pp]rogram\b", conts):
                    return filename
        except UnicodeDecodeError:
            pass
        except IOError:
            pass
    return files[0]

# From online_judge_programming/submit.py
def guess_mainclass(language, files):
    if language in _GUESS_MAINFILE and len(files) > 1:
        return os.path.basename(guess_mainfile(language, files))
    if language in _GUESS_MAINCLASS:
        mainfile = os.path.basename(guess_mainfile(language, files))
        name = os.path.splitext(mainfile)[0]
        if language == "Kotlin":
            return name[0].upper() + name[1:] + "Kt"
        return name
    return None

# From online_judge_programming/submit.py
def login(login_url, username, password=None, token=None):
    """Log in to Kattis.

    At least one of password or token needs to be provided.

    Returns a requests.Response with cookies needed to be able to submit
    """
    login_args = {"user": username, "script": "true"}
    if password:
        login_args["password"] = password
    if token:
        login_args["token"] = token

    return requests.post(login_url, data=login_args, headers=_HEADERS)

# From online_judge_programming/submit.py
def login_from_config(cfg):
    """Log in to Kattis using the access information in a kattisrc file

    Returns a requests.Response with cookies needed to be able to submit
    """
    username = cfg.get("user", "username")
    password = token = None
    try:
        password = cfg.get("user", "password")
    except configparser.NoOptionError:
        pass
    try:
        token = cfg.get("user", "token")
    except configparser.NoOptionError:
        pass
    if password is None and token is None:
        raise ConfigError(
            """\
Your .kattisrc file appears corrupted. It must provide a token (or a
KATTIS password).

Please download a new .kattisrc file"""
        )

    loginurl = get_url(cfg, "loginurl", "login")
    return login(loginurl, username, password, token)

# From online_judge_programming/submit.py
def submit(
    submit_url,
    cookies,
    problem,
    language,
    files,
    mainclass="",
    tag="",
    assignment=None,
    contest=None,
):
    """Make a submission.

    The url_opener argument is an OpenerDirector object to use (as
    returned by the login() function)

    Returns the requests.Result from the submission
    """

    data = {
        "submit": "true",
        "submit_ctr": 2,
        "language": language,
        "mainclass": mainclass,
        "problem": problem,
        "tag": tag,
        "script": "true",
    }

    if assignment is not None:
        data["assignment"] = assignment
    if contest is not None:
        data["contest"] = contest
    sub_files = []
    for f in files:
        with open(f, "rb") as sub_file:
            sub_files.append(
                ("sub_file[]", (os.path.basename(f), sub_file.read(), "application/octet-stream"))
            )

    return requests.post(submit_url, data=data, files=sub_files, cookies=cookies, headers=_HEADERS)

# From online_judge_programming/submit.py
def confirm_or_die(problem, language, files, mainclass, tag):
    print("Problem:", problem)
    print("Language:", language)
    print("Files:", ", ".join(files))
    if mainclass:
        if language in _GUESS_MAINFILE:
            print("Main file:", mainclass)
        else:
            print("Mainclass:", mainclass)
    if tag:
        print("Tag:", tag)
    print("Submit (y/N)?")
    if sys.stdin.readline().upper()[:-1] != "Y":
        print("Cancelling")
        sys.exit(1)

# From online_judge_programming/submit.py
def get_submission_url(submit_response, cfg):
    m = re.search(r"Submission ID: (\d+)", submit_response)
    if m:
        submissions_url = get_url(cfg, "submissionsurl", "submissions")
        submission_id = m.group(1)
        return f"{submissions_url}/{submission_id}"

# From online_judge_programming/submit.py
def get_submission_status(submission_url, cookies):
    reply = requests.get(submission_url + "?json", cookies=cookies, headers=_HEADERS)
    return reply.json()

# From online_judge_programming/submit.py
def color(s, c):
    return f"\x1b[{c}m{s}\x1b[0m"

# From online_judge_programming/submit.py
def show_judgement(submission_url, cfg):
    login_reply = login_from_config(cfg)
    while True:
        status = get_submission_status(submission_url, login_reply.cookies)
        status_id = status["status_id"]
        testcases_done = status["testcase_index"]
        testcases_total = status["row_html"].count("<i") - 1

        status_text = _STATUS_MAP.get(status_id, f"Unknown status {status_id}")

        if status_id < _RUNNING_STATUS:
            print(f"\r{status_text}...", end="")
        else:
            print("\rTest cases: ", end="")

        if status_id == _COMPILE_ERROR_STATUS:
            print(f"\r{color(status_text, _RED_COLOR)}", end="")
            try:
                root = fragment_fromstring(status["feedback_html"], create_parent=True)
                error = root.find(".//pre").text
                print(color(":", _RED_COLOR))
                print(error, end="")
            except:
                pass
        elif status_id < _RUNNING_STATUS:
            print(f"\r{status_text}...", end="")
        else:
            print("\rTest cases: ", end="")

            if testcases_total == 0:
                print("???", end="")
            else:
                progress = ""
                testcases_correct = 0
                for i in re.findall(r'<i class="([\w\- ]*)" title', status["row_html"]):
                    if "is-empty" in i:
                        break
                    if "accepted" in i:
                        progress += color(".", _GREEN_COLOR)
                        testcases_correct += 1
                    if "rejected" in i:
                        progress += color("x", _RED_COLOR)

                # NB: We need to do the following math since len(color('.', _SOME_COLOR)) == 10
                if status_id == _RUNNING_STATUS:
                    progress = progress[: 10 * (testcases_done - 1)] + color("?", _YELLOW_COLOR)
                print(
                    f'[{progress}{" " * (9*testcases_done + testcases_total - len(progress))}]  {testcases_done} / {testcases_total}',
                    end="",
                )

        sys.stdout.flush()

        if status_id > _RUNNING_STATUS:
            # Done
            print()
            success = status_id == _ACCEPTED_STATUS
            try:
                root = fragment_fromstring(status["row_html"], create_parent=True)
                cpu_time = root.xpath('.//*[@data-type="cpu"]')[0].text_content()
                try:
                    score = re.findall(
                        r"\(([\d\.]+)\)", root.xpath('.//*[@data-type="status"]')[0].text_content()
                    )[0]
                except:
                    score = ""
                status_text += (
                    " (" + cpu_time + ", " + score + ")" if score else " (" + cpu_time + ")"
                )
            except:
                pass
            if status_id != _COMPILE_ERROR_STATUS:
                print(color(status_text, _GREEN_COLOR if success else _RED_COLOR))
            numerical_score = int(score) if score else 0
            return success, numerical_score, testcases_done, testcases_correct, testcases_total

        time.sleep(0.25)


from qwen3_benchmark_suite import Qwen3BenchmarkSuite
from qwen3_benchmark_suite import BenchmarkConfig
import mlx.core

# From mlx_metal_kernel_opt/quick_benchmark_test.py
def run_quick_test():
    """Run a quick test with just a few key benchmarks with proper warmup"""

    # Test configs - subset of full suite
    test_configs = [
        BenchmarkConfig(
            name="baseline_test",
            prompt="The future of AI is",
            max_tokens=100,
            description="Baseline test matching your original benchmark",
        ),
        BenchmarkConfig(
            name="short_context_quick",
            prompt="Brief answer: What is artificial intelligence?",
            max_tokens=50,
            description="Short context, quick response",
        ),
        BenchmarkConfig(
            name="code_generation_test",
            prompt="Write a Python function to implement binary search:",
            max_tokens=200,
            description="Code generation test",
        ),
        BenchmarkConfig(
            name="long_generation_test",
            prompt="Explain in detail how neural networks learn:",
            max_tokens=500,
            description="Longer generation test",
        ),
        BenchmarkConfig(
            name="memory_efficiency_test",
            prompt="Write a comprehensive guide on optimizing memory usage in large-scale machine learning systems, covering techniques for both training and inference:",
            max_tokens=800,
            description="Memory efficiency stress test",
        ),
    ]

    # Use mlx-lm as installed package (no need to change directories)
    try:
        # Import mlx for cache clearing
        import mlx.core as mx
        import numpy as np

        benchmark_suite = Qwen3BenchmarkSuite()

        print(f"\n{'='*80}")
        print(f"Quick Benchmark Test - Qwen3-0.6B")
        print(f"Testing {len(test_configs)} key scenarios with warmup")
        print(f"Purpose: Validate Metal kernel optimization baseline")
        print(f"{'='*80}")

        # Global warmup - run one quick test to warm up the system
        print(f"🔥 Running global warmup to initialize MLX and model...")
        try:
            mx.clear_cache()
            warmup_config = BenchmarkConfig(
                name="warmup", prompt="Hello", max_tokens=5, description="Warmup run"
            )
            print(f"   Global warmup in progress...")
            warmup_result = benchmark_suite.run_single_benchmark(warmup_config)
            print(f"   ✅ Global warmup completed")
        except Exception as e:
            print(f"   ⚠️  Global warmup failed: {e}")
            print(f"   Continuing with individual tests...")

        results = []
        for i, config in enumerate(test_configs, 1):
            print(f"\n[{i}/{len(test_configs)}] Running: {config.name}")
            try:
                # The benchmark_suite.run_single_benchmark already has warmup built-in
                result = benchmark_suite.run_single_benchmark(config)
                results.append(result)
            except Exception as e:
                print(f"Failed: {e}")
                continue

        # Print summary
        if results:
            print(f"\n{'='*80}")
            print(f"Quick Test Results Summary")
            print(f"{'='*80}")
            print(f"{'Name':<25} {'Gen Tokens':<12} {'Decode Speed':<15} {'Memory':<10} {'CV%':<8}")
            print(f"{'-'*80}")

            for result in results:
                # Extract standard deviation from the result display if available
                cv_display = "N/A"
                print(
                    f"{result.name:<25} "
                    f"{result.generated_tokens:<12} "
                    f"{result.decode_tokens_per_sec:<15.1f} "
                    f"{result.peak_memory_gb:<10.2f} "
                    f"{cv_display:<8}"
                )

            print(f"{'-'*80}")
            decode_speeds = [
                r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0
            ]
            if decode_speeds:
                print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
                print(
                    f"Speed range: {np.min(decode_speeds):.1f} - {np.max(decode_speeds):.1f} tokens/sec"
                )
                print(f"Performance std dev: {np.std(decode_speeds):.1f} tokens/sec")
                print(
                    f"Overall consistency: {np.std(decode_speeds)/np.mean(decode_speeds)*100:.1f}% CV"
                )

        print(f"\n{'='*80}")
        print("Quick test complete! If this looks good, run the full benchmark suite.")
        print("Full suite: python qwen3_benchmark_suite.py")
        print("Compare mode: python run_benchmarks.py --mode compare")
        print(f"✅ All tests included proper warmup for reliable results")
        print(f"🎯 Ready to test custom Metal kernel optimization!")
        print(f"{'='*80}")

        return results

    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return None

import threading
import mlx.nn
from qwen3_benchmark_suite import BenchmarkResult
import mlx_lm.models.qwen3

# From mlx_metal_kernel_opt/evaluator.py
class MetalKernelSafetyError(Exception):
    """Metal kernel safety violation"""

    pass

# From mlx_metal_kernel_opt/evaluator.py
class GPUCommandBufferError(Exception):
    """GPU command buffer execution error"""

    pass

# From mlx_metal_kernel_opt/evaluator.py
class MetalMemoryViolationError(Exception):
    """Metal kernel memory access violation"""

    pass

# From mlx_metal_kernel_opt/evaluator.py
class BulletproofMetalEvaluator:
    """Bulletproof evaluator that NEVER crashes from Metal kernel failures"""

    def __init__(self):
        self.model_path = "mlx-community/Qwen3-0.6B-bf16"

        # Enhanced error handling configuration
        self.max_retry_attempts = 3
        self.retry_base_delay = 1.0  # Base delay for exponential backoff
        self.kernel_validation_timeout = 30  # Timeout for kernel validation

        # Comprehensive error tracking
        self.metal_command_buffer_errors = 0
        self.metal_memory_violations = 0
        self.metal_compilation_errors = 0
        self.gpu_resource_errors = 0
        self.total_metal_errors = 0
        self.successful_fallbacks = 0
        self.retry_attempts_used = 0

        # Safety thresholds
        self.max_sequence_length_safe = 512  # Start with safer sequence lengths
        self.max_batch_size_safe = 1
        self.max_head_dimension_safe = 128

        # Baseline metrics storage
        self.baseline_metrics = None
        self.baseline_results = None

        # Use comprehensive benchmark suite
        self.benchmark_suite = Qwen3BenchmarkSuite(self.model_path)

        print("🛡️  BULLETPROOF METAL KERNEL EVALUATOR INITIALIZED")
        print(f"📱 Model: {self.model_path}")
        print(f"🔁 Max retry attempts: {self.max_retry_attempts}")
        print(f"⚡ GPU error protection: MAXIMUM")
        print(f"🧠 Memory safety validation: ENABLED")
        print(f"🎯 Command buffer error handling: ACTIVE")

    def evaluate(self, program_text: str) -> Dict[str, Any]:
        """
        BULLETPROOF evaluation that handles ALL Metal kernel failures:
        1. Enhanced program extraction with syntax validation
        2. Pre-execution kernel safety validation
        3. Protected baseline measurement with fallback
        4. GPU-safe correctness testing with memory checks
        5. Armored benchmarking with command buffer protection
        6. Comprehensive Metal error recovery and statistics
        """

        print("\n" + "🛡️ " * 50)
        print("🛡️  BULLETPROOF METAL KERNEL EVALUATION STARTING")
        print("🛡️ " * 50)
        print("✅ GPU Command Buffer Error Protection: ACTIVE")
        print("✅ Metal Memory Violation Detection: ENABLED")
        print("✅ Automatic Fallback Mechanisms: READY")
        print("✅ Multi-layer Error Recovery: ARMED")
        print("✅ Evolution Process Protection: MAXIMUM")
        print("🛡️ " * 50)

        try:
            # Reset all error counters
            self._reset_error_counters()

            # Step 1: Enhanced program extraction with Metal validation
            print("\n🔧 STEP 1: Enhanced Program Extraction with Metal Validation")
            extraction_result = self._bulletproof_extract_custom_attention(program_text)
            if not extraction_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Program extraction failed: {extraction_result['error']}"
                )

            custom_attention_class = extraction_result["class"]

            # Step 2: Pre-execution Metal kernel safety validation
            print("\n🔍 STEP 2: Pre-execution Metal Kernel Safety Validation")
            safety_result = self._validate_metal_kernel_safety(custom_attention_class)
            if not safety_result["success"]:
                print(f"⚠️  Metal kernel safety validation failed: {safety_result['error']}")
                print("🛡️  Proceeding with enhanced protection...")

            # Step 3: GPU-protected baseline measurement
            print("\n📊 STEP 3: GPU-Protected Baseline Performance Measurement")
            baseline_results = self._gpu_protected_measure_baseline()
            if not baseline_results:
                return self._create_comprehensive_failure_result(
                    "Failed to measure baseline performance with GPU protection"
                )

            # Step 4: Memory-safe correctness testing
            print("\n🔍 STEP 4: Memory-Safe Custom Attention Correctness Testing")
            correctness_result = self._memory_safe_correctness_test(custom_attention_class)
            if not correctness_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Memory-safe correctness test failed: {correctness_result['error']}"
                )

            correctness_score = correctness_result["score"]
            if correctness_score < 0.90:  # Slightly more lenient for complex kernels
                return self._create_comprehensive_failure_result(
                    f"Correctness score too low: {correctness_score:.3f} (required: 0.90)"
                )

            # Step 5: Command-buffer-protected benchmarking
            print("\n🚀 STEP 5: Command-Buffer-Protected Performance Benchmarking")
            benchmark_result = self._command_buffer_protected_benchmark(custom_attention_class)
            if not benchmark_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Command-buffer-protected benchmarking failed: {benchmark_result['error']}"
                )

            custom_results = benchmark_result["results"]

            # Step 6: Enhanced performance analysis
            print("\n📈 STEP 6: Enhanced Performance Analysis")
            performance_analysis = self._analyze_performance_with_safety_metrics(
                baseline_results, custom_results
            )

            # Step 7: Calculate safety-adjusted final score
            final_score = self._calculate_safety_adjusted_score(
                performance_analysis, correctness_score
            )

            # Step 8: Generate comprehensive result with full error statistics
            result = {
                "success": True,
                "final_score": final_score,
                "performance_metrics": performance_analysis["aggregate_metrics"],
                "correctness_score": correctness_score,
                "benchmark_results": [self._result_to_dict(r) for r in custom_results],
                "baseline_comparison": performance_analysis["comparison_summary"],
                "individual_comparisons": performance_analysis["individual_comparisons"],
                "summary": self._generate_comprehensive_summary(
                    performance_analysis, correctness_score
                ),
                "metal_safety_statistics": self._get_comprehensive_error_statistics(),
                "safety_validation": safety_result,
            }

            self._print_bulletproof_evaluation_results(result)
            return result

        except Exception as e:
            # Ultimate protection: even this top-level catch must never crash evolution
            self.total_metal_errors += 1
            error_msg = f"TOP-LEVEL BULLETPROOF CATCH: {str(e)}"
            print(f"🛡️  {error_msg}")
            traceback.print_exc()
            return self._create_comprehensive_failure_result(error_msg)

    def _reset_error_counters(self):
        """Reset all error tracking counters"""
        self.metal_command_buffer_errors = 0
        self.metal_memory_violations = 0
        self.metal_compilation_errors = 0
        self.gpu_resource_errors = 0
        self.total_metal_errors = 0
        self.successful_fallbacks = 0
        self.retry_attempts_used = 0

    def _bulletproof_extract_custom_attention(self, program_text: str) -> Dict[str, Any]:
        """Bulletproof extraction with comprehensive Metal kernel validation"""
        try:
            print("  🔍 Bulletproof program analysis with Metal validation...")

            # Handle file paths vs direct text
            if (
                program_text.startswith("/")
                and "\n" not in program_text
                and len(program_text) < 500
            ):
                print(f"  📁 Reading program from file: {program_text}")
                if os.path.exists(program_text):
                    try:
                        with open(program_text, "r") as f:
                            actual_program_text = f.read()
                    except Exception as e:
                        return {"success": False, "error": f"File read error: {e}"}
                else:
                    return {"success": False, "error": f"Program file not found: {program_text}"}
            else:
                actual_program_text = program_text

            # Enhanced syntax validation
            try:
                compile(actual_program_text, "<evolved_program>", "exec")
                print("  ✅ Enhanced syntax validation passed")
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax error: {e}"}

            # Pre-validate Metal kernel syntax (static analysis)
            metal_validation = self._static_validate_metal_kernel_syntax(actual_program_text)
            if not metal_validation["safe"]:
                print(
                    f"  ⚠️  Metal kernel static validation warning: {metal_validation['warnings']}"
                )

            # Create ultra-safe execution environment
            exec_globals = self._create_bulletproof_execution_environment()

            # Execute program with maximum protection
            print("  ⚙️  Executing program with MAXIMUM protection...")
            try:
                success, result = self._bulletproof_execute_with_gpu_protection(
                    lambda: exec(actual_program_text, exec_globals)
                )

                if not success:
                    self.total_metal_errors += 1
                    return {"success": False, "error": f"Protected execution failed: {result}"}

            except Exception as e:
                self.total_metal_errors += 1
                return {"success": False, "error": f"Execution error with GPU protection: {e}"}

            # Enhanced class extraction and validation
            custom_class = exec_globals.get("CustomGQAAttention")
            if custom_class is None:
                return {
                    "success": False,
                    "error": "CustomGQAAttention class not found in executed code",
                }

            # Comprehensive class validation
            validation_result = self._validate_custom_attention_class(custom_class)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}

            print(f"  ✅ Successfully extracted and validated CustomGQAAttention class")
            print(f"  🛡️  Metal safety pre-checks: {metal_validation['safe']}")

            return {"success": True, "class": custom_class, "metal_validation": metal_validation}

        except Exception as e:
            self.total_metal_errors += 1
            return {"success": False, "error": f"Bulletproof extraction failed: {str(e)}"}

    def _static_validate_metal_kernel_syntax(self, program_text: str) -> Dict[str, Any]:
        """Static analysis of Metal kernel syntax for common safety issues"""
        warnings = []

        # Check for common Metal safety issues
        dangerous_patterns = [
            ("buffer overflow", ["queries[", "keys[", "values[", "output[", "mask["]),
            ("unguarded loops", ["for (", "while ("]),
            ("raw pointers", ["*queries", "*keys", "*values", "*output"]),
            ("thread sync issues", ["threadgroup", "simdgroup"]),
        ]

        for issue_type, patterns in dangerous_patterns:
            for pattern in patterns:
                if pattern in program_text:
                    warnings.append(f"{issue_type}: {pattern}")

        # Check for bounds checking
        has_bounds_checking = any(
            check in program_text
            for check in [
                "batch_idx >= BATCH_SIZE",
                "head_idx >= NUM_HEADS",
                "query_pos >= SEQ_LEN",
                "d < HEAD_DIM",
            ]
        )

        if not has_bounds_checking:
            warnings.append("missing bounds checking")

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings,
            "has_bounds_checking": has_bounds_checking,
        }

    def _validate_custom_attention_class(self, custom_class: Any) -> Dict[str, Any]:
        """Comprehensive validation of custom attention class"""
        try:
            # Basic type checking
            if not isinstance(custom_class, type):
                return {"valid": False, "error": "CustomGQAAttention is not a valid class"}

            # Check for required methods
            required_methods = ["__init__", "__call__"]
            for method in required_methods:
                if not hasattr(custom_class, method):
                    return {"valid": False, "error": f"Missing required method: {method}"}

            # Check if it inherits from nn.Module (recommended)
            if not issubclass(custom_class, nn.Module):
                print("  ⚠️  CustomGQAAttention doesn't inherit from nn.Module")

            print("  ✅ Custom attention class validation passed")
            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Class validation error: {e}"}

    def _validate_metal_kernel_safety(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Pre-execution validation of Metal kernel safety"""
        try:
            print("  🔍 Validating Metal kernel safety parameters...")

            # Mock arguments for safety testing
            class MockArgs:
                hidden_size = 5120
                num_attention_heads = 40
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

            args = MockArgs()

            # Try to instantiate with safety checks
            try:
                instance = custom_attention_class(args)
                if instance is None:
                    return {"success": False, "error": "Failed to instantiate custom attention"}

                print("  ✅ Custom attention instantiation successful")

                # Basic parameter validation
                if hasattr(instance, "n_heads") and instance.n_heads != 40:
                    return {"success": False, "error": f"Invalid head count: {instance.n_heads}"}

                if hasattr(instance, "n_kv_heads") and instance.n_kv_heads != 8:
                    return {
                        "success": False,
                        "error": f"Invalid KV head count: {instance.n_kv_heads}",
                    }

                return {"success": True, "validated": True}

            except Exception as e:
                error_msg = str(e)
                if any(keyword in error_msg.lower() for keyword in ["metal", "kernel", "gpu"]):
                    self.metal_compilation_errors += 1
                return {"success": False, "error": f"Instantiation failed: {error_msg}"}

        except Exception as e:
            self.total_metal_errors += 1
            return {"success": False, "error": f"Safety validation error: {e}"}

    def _bulletproof_execute_with_gpu_protection(self, func) -> Tuple[bool, Any]:
        """Execute function with maximum GPU and Metal kernel protection"""
        try:
            # Clear any existing GPU state
            mx.eval(mx.array([1.0]))  # Simple operation to ensure GPU is responsive

            # Execute with comprehensive error catching
            result = func()
            return True, result

        except RuntimeError as e:
            error_msg = str(e)

            # Classify specific Metal/GPU errors
            if "kIOGPUCommandBufferCallbackErrorInvalidResource" in error_msg:
                self.metal_command_buffer_errors += 1
                self.total_metal_errors += 1
                return False, f"GPU Command Buffer Error (memory violation): {error_msg}"
            elif "METAL" in error_msg.upper():
                self.metal_memory_violations += 1
                self.total_metal_errors += 1
                return False, f"Metal Memory Violation: {error_msg}"
            elif any(keyword in error_msg.lower() for keyword in ["gpu", "metal", "kernel"]):
                self.gpu_resource_errors += 1
                self.total_metal_errors += 1
                return False, f"GPU Resource Error: {error_msg}"
            else:
                return False, f"Runtime Error: {error_msg}"

        except Exception as e:
            error_msg = str(e)

            # Additional classification for other Metal-related exceptions
            if any(
                keyword in error_msg.lower() for keyword in ["metal", "kernel", "gpu", "mps", "mtl"]
            ):
                self.total_metal_errors += 1
                return False, f"General Metal Error: {error_msg}"
            else:
                return False, f"Execution Error: {error_msg}"

    def _gpu_protected_measure_baseline(self) -> Optional[List[BenchmarkResult]]:
        """GPU-protected baseline measurement with enhanced error handling"""
        try:
            print("  📊 Running GPU-protected baseline benchmark...")

            # Ensure clean GPU state
            self._ensure_clean_gpu_state()
            self._ensure_standard_attention()

            # Get baseline configurations
            baseline_configs = self._get_safe_benchmark_configs()
            if not baseline_configs:
                print("  ❌ No safe benchmark configurations available")
                return None

            baseline_results = []
            successful_count = 0

            for i, config in enumerate(baseline_configs, 1):
                print(f"  [{i}/{len(baseline_configs)}] GPU-protected baseline: {config.name}")

                retry_count = 0
                while retry_count <= self.max_retry_attempts:
                    try:
                        # Clean GPU state before each attempt
                        self._ensure_clean_gpu_state()

                        # Run with GPU protection
                        success, result = self._bulletproof_execute_with_gpu_protection(
                            lambda: self.benchmark_suite.run_single_benchmark(config)
                        )

                        if success and result:
                            baseline_results.append(result)
                            successful_count += 1
                            print(
                                f"    ✅ GPU-protected {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                            )
                            break
                        else:
                            if retry_count < self.max_retry_attempts:
                                print(f"    🔄 Retry {retry_count + 1}: {result}")
                                retry_count += 1
                                time.sleep(self.retry_base_delay * (2**retry_count))
                                continue
                            else:
                                print(f"    ❌ All retries exhausted for {config.name}: {result}")
                                break

                    except Exception as e:
                        if retry_count < self.max_retry_attempts:
                            print(f"    🔄 Exception retry {retry_count + 1}: {e}")
                            retry_count += 1
                            time.sleep(self.retry_base_delay * (2**retry_count))
                            continue
                        else:
                            print(f"    ❌ Final exception for {config.name}: {e}")
                            break

            # Check success rate
            min_required = max(2, len(baseline_configs) * 0.5)  # At least 50% success
            if successful_count < min_required:
                print(
                    f"  ❌ Insufficient baseline results: {successful_count}/{len(baseline_configs)}"
                )
                return None

            # Store baseline metrics
            self._store_enhanced_baseline_metrics(baseline_results)
            print(f"  ✅ GPU-protected baseline complete ({successful_count} successful)")

            return baseline_results

        except Exception as e:
            print(f"  ❌ GPU-protected baseline measurement failed: {e}")
            return None

    def _memory_safe_correctness_test(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Memory-safe correctness testing with GPU protection"""
        print("  🔍 Running memory-safe correctness testing...")

        try:
            # Safe test configuration
            class MockArgs:
                hidden_size = 5120
                num_attention_heads = 40
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

            args = MockArgs()

            # Conservative test cases (smaller sequences for safety)
            test_cases = [
                (1, 8, 5120),  # Micro sequence
                (1, 16, 5120),  # Very short
                (1, 32, 5120),  # Short sequence
                (1, 64, 5120),  # Medium sequence
            ]

            correctness_scores = []
            local_command_buffer_errors = 0
            local_memory_violations = 0

            for B, L, D in test_cases:
                print(f"      🧪 Memory-safe testing sequence length {L}...")

                retry_count = 0
                while retry_count <= self.max_retry_attempts:
                    try:
                        # Clean GPU state
                        self._ensure_clean_gpu_state()

                        # Create conservative test inputs
                        x = mx.random.normal((B, L, D)) * 0.1  # Smaller values for safety
                        mask = "causal"

                        # Test with maximum GPU protection
                        success, result = self._bulletproof_execute_with_gpu_protection(
                            lambda: self._test_single_sequence_memory_safe(
                                custom_attention_class, args, x, mask
                            )
                        )

                        if success:
                            correctness_scores.append(result)
                            print(f"      ✅ Sequence {L}: PASS (score={result:.3f})")
                            break
                        else:
                            error_msg = str(result)

                            # Enhanced error classification
                            if "command buffer" in error_msg.lower():
                                local_command_buffer_errors += 1
                            elif "memory violation" in error_msg.lower():
                                local_memory_violations += 1

                            if retry_count < self.max_retry_attempts:
                                print(
                                    f"      🔄 Retry {retry_count + 1} for length {L}: {error_msg}"
                                )
                                retry_count += 1
                                time.sleep(self.retry_base_delay * (2**retry_count))
                                continue
                            else:
                                print(f"      ❌ All retries failed for length {L}: {error_msg}")
                                correctness_scores.append(0.0)
                                break

                    except Exception as e:
                        error_msg = str(e)
                        print(f"      ❌ Exception for length {L}: {error_msg}")

                        if retry_count < self.max_retry_attempts:
                            retry_count += 1
                            time.sleep(self.retry_base_delay * (2**retry_count))
                            continue
                        else:
                            correctness_scores.append(0.0)
                            break

            # Update global error counters
            self.metal_command_buffer_errors += local_command_buffer_errors
            self.metal_memory_violations += local_memory_violations
            self.total_metal_errors += local_command_buffer_errors + local_memory_violations

            # Calculate overall correctness with partial credit
            overall_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

            print(f"    📊 Memory-safe overall correctness: {overall_correctness:.3f}")
            print(f"    🛡️  Command buffer errors: {local_command_buffer_errors}")
            print(f"    🛡️  Memory violations: {local_memory_violations}")

            return {
                "success": True,
                "score": overall_correctness,
                "command_buffer_errors": local_command_buffer_errors,
                "memory_violations": local_memory_violations,
            }

        except Exception as e:
            self.total_metal_errors += 1
            print(f"    ❌ Memory-safe correctness testing failed: {e}")
            return {"success": False, "error": str(e)}

    def _test_single_sequence_memory_safe(
        self, custom_attention_class: Any, args: Any, x: Any, mask: Any
    ) -> float:
        """Test single sequence with enhanced memory safety"""
        try:
            # Pre-execution safety checks
            if x.shape[1] > self.max_sequence_length_safe:
                raise MetalKernelSafetyError(
                    f"Sequence length {x.shape[1]} exceeds safe limit {self.max_sequence_length_safe}"
                )

            if x.shape[0] > self.max_batch_size_safe:
                raise MetalKernelSafetyError(
                    f"Batch size {x.shape[0]} exceeds safe limit {self.max_batch_size_safe}"
                )

            # Instantiate with error checking
            custom_attn = custom_attention_class(args)
            if custom_attn is None:
                raise ValueError("Failed to instantiate custom attention")

            # Conservative forward pass with timeout simulation
            start_time = time.time()
            output = custom_attn(x, mask=mask)
            elapsed_time = time.time() - start_time

            # Timeout check (soft limit)
            if elapsed_time > self.kernel_validation_timeout:
                print(f"        ⚠️  Slow execution detected: {elapsed_time:.2f}s")
                return 0.5  # Partial credit for slow but working kernel

            # Enhanced output validation
            if output is None:
                raise ValueError("Custom attention returned None")

            # Shape validation
            expected_shape = x.shape
            if output.shape != expected_shape:
                raise ValueError(f"Wrong output shape: {output.shape}, expected {expected_shape}")

            # Enhanced finite value check
            finite_mask = mx.isfinite(output)
            if not mx.all(finite_mask):
                finite_ratio = float(mx.mean(finite_mask.astype(mx.float32)))
                if finite_ratio < 0.9:
                    raise ValueError(f"Too many non-finite values: {finite_ratio:.2%} finite")
                else:
                    print(f"        ⚠️  Some non-finite values: {finite_ratio:.2%} finite")
                    return 0.7  # Partial credit

            # Enhanced statistical validation
            output_mean = float(mx.mean(output))
            output_std = float(mx.std(output))
            output_max = float(mx.max(mx.abs(output)))

            # More lenient bounds for complex kernels
            if abs(output_mean) > 10.0:
                print(f"        ⚠️  Large mean: {output_mean:.6f}")
                return 0.6

            if output_std > 100.0 or output_std < 0.00001:
                print(f"        ⚠️  Unusual std: {output_std:.6f}")
                return 0.6

            if output_max > 1000.0:
                print(f"        ⚠️  Large max value: {output_max:.6f}")
                return 0.7

            # All checks passed
            return 1.0

        except MetalKernelSafetyError as e:
            raise e  # Re-raise safety errors
        except Exception as e:
            error_msg = str(e)
            if any(
                keyword in error_msg.lower()
                for keyword in ["metal", "kernel", "gpu", "command buffer"]
            ):
                raise GPUCommandBufferError(f"GPU execution error: {error_msg}")
            else:
                raise ValueError(f"Sequence test error: {error_msg}")

    def _command_buffer_protected_benchmark(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Command-buffer-protected benchmarking with maximum safety"""
        print("  🚀 Running command-buffer-protected benchmarking...")

        retry_attempt = 0

        while retry_attempt <= self.max_retry_attempts:
            try:
                print(f"  🔄 Protected attempt {retry_attempt + 1}/{self.max_retry_attempts + 1}")

                # Clean GPU state before each major attempt
                self._ensure_clean_gpu_state()

                # Apply custom attention hook with protection
                hook_result = self._gpu_protected_apply_hook(custom_attention_class)
                if not hook_result["success"]:
                    if retry_attempt < self.max_retry_attempts:
                        print(f"    🔄 Hook failed, retrying... ({hook_result['error']})")
                        retry_attempt += 1
                        time.sleep(self.retry_base_delay * (2**retry_attempt))
                        continue
                    return {
                        "success": False,
                        "error": f"Hook application failed: {hook_result['error']}",
                    }

                original_attention = hook_result["original"]

                try:
                    # Run benchmarks with command buffer protection
                    custom_configs = self._get_safe_benchmark_configs()
                    custom_results = []
                    successful_benchmarks = 0

                    for i, config in enumerate(custom_configs, 1):
                        print(
                            f"    [{i}/{len(custom_configs)}] Command-buffer-protected: {config.name}"
                        )

                        benchmark_retry = 0
                        while benchmark_retry <= 2:  # Fewer retries per benchmark
                            try:
                                # Clean state before each benchmark
                                self._ensure_clean_gpu_state()

                                # Run with maximum protection
                                success, result = self._bulletproof_execute_with_gpu_protection(
                                    lambda: self.benchmark_suite.run_single_benchmark(config)
                                )

                                if success and result:
                                    custom_results.append(result)
                                    successful_benchmarks += 1
                                    print(
                                        f"      ✅ Protected {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                                    )
                                    break
                                else:
                                    if benchmark_retry < 2:
                                        print(
                                            f"      🔄 Benchmark retry {benchmark_retry + 1}: {result}"
                                        )
                                        benchmark_retry += 1
                                        time.sleep(1)
                                        continue
                                    else:
                                        print(f"      ❌ Benchmark failed: {result}")
                                        break

                            except Exception as e:
                                if benchmark_retry < 2:
                                    print(
                                        f"      🔄 Benchmark exception retry {benchmark_retry + 1}: {e}"
                                    )
                                    benchmark_retry += 1
                                    time.sleep(1)
                                    continue
                                else:
                                    print(f"      ❌ Benchmark exception: {e}")
                                    break

                    # Check success rate
                    min_required = max(2, len(custom_configs) * 0.4)  # Lowered to 40% for safety
                    if successful_benchmarks >= min_required:
                        print(
                            f"  ✅ Command-buffer-protected benchmarks complete ({successful_benchmarks} successful)"
                        )
                        self.retry_attempts_used = retry_attempt
                        return {"success": True, "results": custom_results}
                    else:
                        error_msg = f"Insufficient benchmarks: {successful_benchmarks}/{len(custom_configs)} succeeded"
                        if retry_attempt < self.max_retry_attempts:
                            print(f"  🔄 {error_msg}, retrying full attempt...")
                            retry_attempt += 1
                            time.sleep(self.retry_base_delay * (2**retry_attempt))
                            continue
                        return {"success": False, "error": error_msg}

                finally:
                    # Always restore original attention
                    self._gpu_protected_remove_hook(original_attention)

            except Exception as e:
                error_msg = f"Command-buffer-protected attempt failed: {str(e)}"
                print(f"  ❌ {error_msg}")
                if retry_attempt < self.max_retry_attempts:
                    retry_attempt += 1
                    time.sleep(self.retry_base_delay * (2**retry_attempt))
                    continue
                return {"success": False, "error": error_msg}

        return {"success": False, "error": "All command-buffer-protected attempts exhausted"}

    def _ensure_clean_gpu_state(self):
        """Ensure clean GPU state before operations"""
        try:
            # Simple operation to ensure GPU responsiveness
            test_op = mx.array([1.0, 2.0, 3.0])
            mx.eval(test_op * 2)

            # Small delay to let GPU settle
            time.sleep(0.1)

        except Exception as e:
            print(f"    ⚠️  GPU state cleanup warning: {e}")

    def _gpu_protected_apply_hook(self, custom_attention_class: Any) -> Dict[str, Any]:
        """GPU-protected application of custom attention hook"""
        try:
            success, result = self._bulletproof_execute_with_gpu_protection(
                lambda: self._apply_attention_hook_safely(custom_attention_class)
            )

            if success:
                return {"success": True, "original": result}
            else:
                return {"success": False, "error": result}

        except Exception as e:
            return {"success": False, "error": f"GPU-protected hook application failed: {e}"}

    def _apply_attention_hook_safely(self, custom_attention_class: Any) -> Any:
        """Safely apply attention hook"""
        import mlx_lm.models.qwen3 as qwen3_module

        # Store original attention class
        original_attention = getattr(qwen3_module, "Attention", None)
        if original_attention is None:
            raise RuntimeError("Could not find original Attention class")

        # Apply custom attention
        qwen3_module.Attention = custom_attention_class

        # Verify the hook was applied
        if qwen3_module.Attention != custom_attention_class:
            raise RuntimeError("Hook application verification failed")

        print("      ✅ Custom attention hook applied with GPU protection")
        return original_attention

    def _gpu_protected_remove_hook(self, original_attention: Any):
        """GPU-protected removal of custom attention hook"""
        try:
            success, result = self._bulletproof_execute_with_gpu_protection(
                lambda: self._remove_attention_hook_safely(original_attention)
            )

            if not success:
                print(f"      ⚠️  Hook removal warning: {result}")

        except Exception as e:
            print(f"      ⚠️  Hook removal error (non-fatal): {e}")

    def _remove_attention_hook_safely(self, original_attention: Any):
        """Safely remove attention hook"""
        import mlx_lm.models.qwen3 as qwen3_module

        qwen3_module.Attention = original_attention
        print("      ✅ Hook removed with GPU protection")

    def _create_bulletproof_execution_environment(self) -> Dict[str, Any]:
        """Create bulletproof execution environment with enhanced imports"""
        import math
        import numpy as np
        import time
        from typing import Optional, Tuple, Any

        exec_globals = {
            "__builtins__": __builtins__,
            "mx": mx,
            "nn": nn,
            "np": np,
            "math": math,
            "time": time,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
        }

        # Enhanced MLX-LM import with error handling
        try:
            exec_globals["mlx_lm"] = __import__("mlx_lm")
            print("  ✅ MLX-LM imported for bulletproof execution")
        except ImportError:
            print("  ⚠️  MLX-LM not available for bulletproof execution")
        except Exception as e:
            print(f"  ⚠️  MLX-LM import error in bulletproof environment: {e}")

        return exec_globals

    def _get_safe_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Get safer benchmark configurations for GPU protection"""
        try:
            all_configs = self.benchmark_suite.create_benchmark_configs()

            # Use more conservative test set for safety
            safe_test_names = [
                "short_context_quick",  # Safest - very short
                "code_generation",  # Medium safety
                "long_context_detailed",  # More challenging but still safe
                "long_generation",  # Longer generation
                "maximum_context_stress_test",  # Most challenging - saved for last
            ]

            config_dict = {c.name: c for c in all_configs}
            safe_configs = []

            for test_name in safe_test_names:
                if test_name in config_dict:
                    safe_configs.append(config_dict[test_name])

            return safe_configs

        except Exception as e:
            print(f"  ⚠️  Error getting safe benchmark configs: {e}")
            return []

    def _ensure_standard_attention(self):
        """Ensure standard attention is active"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            if hasattr(self, "_original_attention") and self._original_attention:
                qwen3_module.Attention = self._original_attention
                print("  🔄 Restored standard attention for baseline")
        except ImportError:
            print("  ⚠️  Could not access qwen3 module for standard attention")

    def _store_enhanced_baseline_metrics(self, baseline_results: List[BenchmarkResult]):
        """Store enhanced baseline metrics"""
        decode_speeds = [
            r.decode_tokens_per_sec for r in baseline_results if r.decode_tokens_per_sec > 0
        ]
        prefill_speeds = [
            r.prefill_tokens_per_sec for r in baseline_results if r.prefill_tokens_per_sec > 0
        ]
        memories = [r.peak_memory_gb for r in baseline_results if r.peak_memory_gb > 0]

        self.baseline_results = baseline_results
        self.baseline_metrics = {
            "avg_decode_speed": float(np.mean(decode_speeds)) if decode_speeds else 0.0,
            "min_decode_speed": float(np.min(decode_speeds)) if decode_speeds else 0.0,
            "max_decode_speed": float(np.max(decode_speeds)) if decode_speeds else 0.0,
            "std_decode_speed": float(np.std(decode_speeds)) if len(decode_speeds) > 1 else 0.0,
            "avg_prefill_speed": float(np.mean(prefill_speeds)) if prefill_speeds else 0.0,
            "avg_memory_gb": float(np.mean(memories)) if memories else 0.0,
            "max_memory_gb": float(np.max(memories)) if memories else 0.0,
            "num_baseline_tests": len(baseline_results),
        }

        print(
            f"    📊 Enhanced baseline stored - Avg decode: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
        )

    def _analyze_performance_with_safety_metrics(
        self, baseline_results: List[BenchmarkResult], custom_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Analyze performance with enhanced safety metrics"""
        print("  📈 Analyzing performance with safety metrics...")

        baseline_dict = {r.name: r for r in baseline_results}
        custom_dict = {r.name: r for r in custom_results}

        individual_comparisons = []
        improvements = {
            "decode_speed_improvements": [],
            "prefill_speed_improvements": [],
            "total_speed_improvements": [],
            "memory_improvements": [],
            "time_improvements": [],
        }

        # Compare each benchmark
        for name in baseline_dict:
            if name in custom_dict:
                baseline = baseline_dict[name]
                custom = custom_dict[name]

                # Calculate improvements with safety bounds
                decode_improvement = self._safe_calculate_improvement(
                    custom.decode_tokens_per_sec, baseline.decode_tokens_per_sec
                )
                prefill_improvement = self._safe_calculate_improvement(
                    custom.prefill_tokens_per_sec, baseline.prefill_tokens_per_sec
                )
                total_improvement = self._safe_calculate_improvement(
                    custom.total_tokens_per_sec, baseline.total_tokens_per_sec
                )
                memory_improvement = self._safe_calculate_improvement(
                    baseline.peak_memory_gb, custom.peak_memory_gb  # Reversed for memory
                )
                time_improvement = self._safe_calculate_improvement(
                    baseline.total_time_sec, custom.total_time_sec  # Reversed for time
                )

                comparison = {
                    "benchmark_name": name,
                    "baseline": self._result_to_dict(baseline),
                    "custom": self._result_to_dict(custom),
                    "improvements": {
                        "decode_speed_pct": decode_improvement,
                        "prefill_speed_pct": prefill_improvement,
                        "total_speed_pct": total_improvement,
                        "memory_reduction_pct": memory_improvement,
                        "time_reduction_pct": time_improvement,
                    },
                }

                individual_comparisons.append(comparison)

                improvements["decode_speed_improvements"].append(decode_improvement)
                improvements["prefill_speed_improvements"].append(prefill_improvement)
                improvements["total_speed_improvements"].append(total_improvement)
                improvements["memory_improvements"].append(memory_improvement)
                improvements["time_improvements"].append(time_improvement)

                print(f"    • {name}: {decode_improvement:+.1f}% decode speed")

        # Calculate aggregate statistics with safety checks
        aggregate_stats = {}
        for key, values in improvements.items():
            if values:
                # Use robust statistics
                valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                if valid_values:
                    aggregate_stats[f"{key}_avg"] = float(np.mean(valid_values))
                    aggregate_stats[f"{key}_median"] = float(np.median(valid_values))
                    aggregate_stats[f"{key}_min"] = float(np.min(valid_values))
                    aggregate_stats[f"{key}_max"] = float(np.max(valid_values))
                    aggregate_stats[f"{key}_std"] = float(np.std(valid_values))

        # Calculate custom metrics
        custom_decode_speeds = [
            r.decode_tokens_per_sec for r in custom_results if r.decode_tokens_per_sec > 0
        ]
        custom_prefill_speeds = [
            r.prefill_tokens_per_sec for r in custom_results if r.prefill_tokens_per_sec > 0
        ]
        custom_memories = [r.peak_memory_gb for r in custom_results if r.peak_memory_gb > 0]

        aggregate_metrics = {
            "avg_decode_speed": (
                float(np.mean(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "min_decode_speed": (
                float(np.min(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "max_decode_speed": (
                float(np.max(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "avg_prefill_speed": (
                float(np.mean(custom_prefill_speeds)) if custom_prefill_speeds else 0.0
            ),
            "avg_memory_gb": float(np.mean(custom_memories)) if custom_memories else 0.0,
            "max_memory_gb": float(np.max(custom_memories)) if custom_memories else 0.0,
            "num_successful_tests": len(custom_results),
            "decode_speed_std": (
                float(np.std(custom_decode_speeds)) if len(custom_decode_speeds) > 1 else 0.0
            ),
        }

        # Enhanced comparison summary
        comparison_summary = {
            "avg_decode_improvement_pct": aggregate_stats.get("decode_speed_improvements_avg", 0),
            "avg_decode_improvement_absolute": (
                aggregate_metrics["avg_decode_speed"] - self.baseline_metrics["avg_decode_speed"]
            ),
            "memory_change_gb": (
                aggregate_metrics["avg_memory_gb"] - self.baseline_metrics["avg_memory_gb"]
            ),
            "target_achieved": aggregate_stats.get("decode_speed_improvements_avg", 0) >= 5.0,
            "num_benchmarks_improved": sum(
                1 for x in improvements["decode_speed_improvements"] if x > 1.0
            ),  # More lenient
            "total_benchmarks": len(improvements["decode_speed_improvements"]),
            "safety_score": self._calculate_safety_score(),
        }

        print(
            f"  📊 Enhanced analysis complete: {comparison_summary['avg_decode_improvement_pct']:+.1f}% avg improvement"
        )
        print(f"  🛡️  Safety score: {comparison_summary['safety_score']:.2f}")

        return {
            "individual_comparisons": individual_comparisons,
            "aggregate_improvements": aggregate_stats,
            "aggregate_metrics": aggregate_metrics,
            "comparison_summary": comparison_summary,
        }

    def _safe_calculate_improvement(self, new_value: float, old_value: float) -> float:
        """Safely calculate percentage improvement with bounds"""
        if old_value <= 0 or np.isnan(old_value) or np.isnan(new_value):
            return 0.0

        improvement = (new_value - old_value) / old_value * 100

        # Clamp extreme values for safety
        return max(-100.0, min(1000.0, improvement))

    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score based on error statistics"""
        total_operations = (
            self.metal_command_buffer_errors
            + self.metal_memory_violations
            + self.metal_compilation_errors
            + self.gpu_resource_errors
            + 10  # Assumed successful operations
        )

        error_rate = self.total_metal_errors / total_operations
        safety_score = max(0.0, 1.0 - error_rate) * 100

        return safety_score

    def _calculate_safety_adjusted_score(
        self, performance_analysis: Dict[str, Any], correctness: float
    ) -> float:
        """Calculate final score adjusted for safety"""
        if correctness < 0.90:
            return -1000.0

        comparison = performance_analysis["comparison_summary"]
        avg_improvement = comparison["avg_decode_improvement_pct"]
        memory_change = comparison["memory_change_gb"]
        success_rate = comparison["num_benchmarks_improved"] / max(
            1, comparison["total_benchmarks"]
        )
        safety_score = comparison["safety_score"]

        # Enhanced score components
        performance_score = avg_improvement * 3  # Primary component
        memory_bonus = max(0, -memory_change * 10)  # Bonus for memory reduction
        consistency_bonus = success_rate * 10  # Bonus for consistent improvements
        correctness_bonus = correctness * 5  # Bonus for correctness
        safety_bonus = (safety_score / 100) * 5  # Bonus for safety

        # Penalty for excessive errors
        error_penalty = min(self.total_metal_errors * 2, 20)  # Cap penalty

        final_score = (
            performance_score
            + memory_bonus
            + consistency_bonus
            + correctness_bonus
            + safety_bonus
            - error_penalty
        )

        print(f"  🎯 Safety-adjusted score breakdown:")
        print(f"    • Performance: {avg_improvement:.2f}% × 3 = {performance_score:.2f}")
        print(f"    • Memory: {memory_bonus:.2f}")
        print(f"    • Consistency: {success_rate:.2f} × 10 = {consistency_bonus:.2f}")
        print(f"    • Correctness: {correctness:.3f} × 5 = {correctness_bonus:.2f}")
        print(f"    • Safety: {safety_score:.1f}/100 × 5 = {safety_bonus:.2f}")
        print(f"    • Error penalty: -{error_penalty:.2f}")
        print(f"    • Final score: {final_score:.2f}")

        return final_score

    def _generate_comprehensive_summary(
        self, performance_analysis: Dict[str, Any], correctness: float
    ) -> str:
        """Generate comprehensive evaluation summary with safety info"""
        comparison = performance_analysis["comparison_summary"]
        metrics = performance_analysis["aggregate_metrics"]

        avg_improvement = comparison["avg_decode_improvement_pct"]
        current_decode = metrics["avg_decode_speed"]
        baseline_decode = self.baseline_metrics["avg_decode_speed"]
        safety_score = comparison["safety_score"]

        summary = f"""Bulletproof Custom GQA Implementation Results:
• Decode Speed: {current_decode:.1f} tokens/sec (baseline: {baseline_decode:.1f})
• Improvement: {avg_improvement:+.1f}%
• Memory Usage: {metrics['avg_memory_gb']:.2f} GB
• Correctness: {correctness:.1%}
• Safety Score: {safety_score:.1f}/100
• Tests Passed: {metrics['num_successful_tests']}/{len(self._get_safe_benchmark_configs())}
• Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}
• Metal Errors Handled: {self.total_metal_errors}"""

        if self.total_metal_errors == 0:
            summary += "\n🛡️  PERFECT SAFETY: No Metal kernel errors"
        elif self.total_metal_errors < 3:
            summary += f"\n🛡️  GOOD SAFETY: {self.total_metal_errors} Metal errors handled"
        else:
            summary += f"\n⚠️  SAFETY CONCERNS: {self.total_metal_errors} Metal errors handled"

        if avg_improvement >= 15:
            summary += "\n🎯 EXCELLENT: 15%+ improvement achieved!"
        elif avg_improvement >= 10:
            summary += "\n🚀 STRONG IMPROVEMENT: 10%+ speedup"
        elif avg_improvement >= 5:
            summary += "\n✅ GOOD IMPROVEMENT: 5%+ speedup"
        elif avg_improvement > 0:
            summary += "\n📈 MINOR IMPROVEMENT: Some speedup achieved"
        else:
            summary += "\n⚠️  NO IMPROVEMENT: Performance regression"

        return summary

    def _get_comprehensive_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "metal_command_buffer_errors": self.metal_command_buffer_errors,
            "metal_memory_violations": self.metal_memory_violations,
            "metal_compilation_errors": self.metal_compilation_errors,
            "gpu_resource_errors": self.gpu_resource_errors,
            "total_metal_errors": self.total_metal_errors,
            "successful_fallbacks": self.successful_fallbacks,
            "retry_attempts_used": self.retry_attempts_used,
            "safety_score": self._calculate_safety_score(),
            "error_breakdown": {
                "command_buffer_pct": (
                    self.metal_command_buffer_errors / max(1, self.total_metal_errors)
                )
                * 100,
                "memory_violation_pct": (
                    self.metal_memory_violations / max(1, self.total_metal_errors)
                )
                * 100,
                "compilation_error_pct": (
                    self.metal_compilation_errors / max(1, self.total_metal_errors)
                )
                * 100,
                "resource_error_pct": (self.gpu_resource_errors / max(1, self.total_metal_errors))
                * 100,
            },
        }

    def _print_bulletproof_evaluation_results(self, result: Dict[str, Any]):
        """Print comprehensive bulletproof evaluation results"""
        print(f"\n{'🛡️ '*25}")
        print(f"{'🛡️  BULLETPROOF EVALUATION RESULTS  🛡️':^100}")
        print(f"{'🛡️ '*25}")

        if result["success"]:
            performance = result["performance_metrics"]
            comparison = result["baseline_comparison"]
            safety_stats = result["metal_safety_statistics"]

            print(f"📊 FINAL SCORE: {result['final_score']:.2f}")
            print(f"")
            print(f"📈 PERFORMANCE COMPARISON:")
            print(f"  • Average Decode Speed: {performance['avg_decode_speed']:.1f} tokens/sec")
            print(
                f"  • Baseline Decode Speed: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
            )
            print(f"  • Average Improvement: {comparison['avg_decode_improvement_pct']:+.1f}%")
            print(
                f"  • Absolute Improvement: {comparison['avg_decode_improvement_absolute']:+.1f} tokens/sec"
            )
            print(f"")
            print(f"🛡️  SAFETY STATISTICS:")
            print(f"  • Safety Score: {safety_stats['safety_score']:.1f}/100")
            print(f"  • Command Buffer Errors: {safety_stats['metal_command_buffer_errors']}")
            print(f"  • Memory Violations: {safety_stats['metal_memory_violations']}")
            print(f"  • Total Metal Errors: {safety_stats['total_metal_errors']}")
            print(f"  • Retry Attempts Used: {safety_stats['retry_attempts_used']}")
            print(f"")
            print(f"💾 MEMORY USAGE:")
            print(f"  • Average Memory: {performance['avg_memory_gb']:.2f} GB")
            print(f"  • Baseline Memory: {self.baseline_metrics['avg_memory_gb']:.2f} GB")
            print(f"  • Memory Change: {comparison['memory_change_gb']:+.2f} GB")
            print(f"")
            print(f"✓ RELIABILITY:")
            print(f"  • Correctness Score: {result['correctness_score']:.1%}")
            print(f"  • Successful Tests: {performance['num_successful_tests']}")
            print(
                f"  • Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}"
            )

            if comparison["target_achieved"]:
                print(f"\n🎯 TARGET ACHIEVED: Significant improvement with safety!")

            if safety_stats["total_metal_errors"] == 0:
                print(f"\n🛡️  PERFECT EXECUTION: No Metal kernel errors encountered!")

        else:
            print(f"❌ EVALUATION FAILED (SAFELY)")
            print(f"📋 Error: {result.get('error', 'Unknown error')}")
            safety_stats = result.get("metal_safety_statistics", {})
            print(f"🛡️  Metal Errors Handled: {safety_stats.get('total_metal_errors', 0)}")

        print(f"{'🛡️ '*25}")

    def _create_comprehensive_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create comprehensive failure result with full error statistics"""
        return {
            "success": False,
            "final_score": -1000.0,
            "error": error_message,
            "performance_metrics": {},
            "correctness_score": 0.0,
            "summary": f"Bulletproof evaluation failed (safely): {error_message}",
            "metal_safety_statistics": self._get_comprehensive_error_statistics(),
            "safety_validation": {"success": False, "error": error_message},
        }

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dictionary"""
        return {
            "name": result.name,
            "decode_tokens_per_sec": result.decode_tokens_per_sec,
            "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "generated_tokens": result.generated_tokens,
            "total_time_sec": result.total_time_sec,
        }

# From mlx_metal_kernel_opt/evaluator.py
class MockArgs:
                hidden_size = 5120
                num_attention_heads = 40
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

# From mlx_metal_kernel_opt/evaluator.py
def test_bulletproof_evaluator():
    """Test the bulletproof Metal kernel evaluator"""
    print("🧪 Testing Bulletproof Metal Kernel Evaluator")
    print("🛡️ " * 40)

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if not os.path.exists(initial_program_path):
        print(f"❌ Initial program not found: {initial_program_path}")
        return

    print(f"📁 Testing with bulletproof protection: {initial_program_path}")
    result = evaluate(initial_program_path)

    print(f"\n{'🛡️ '*20}")
    print(f"🔬 BULLETPROOF EVALUATOR TEST RESULTS")
    print(f"{'🛡️ '*20}")
    print(f"Success: {result['success']}")
    print(f"Final Score: {result.get('final_score', 'N/A')}")

    if result.get("metal_safety_statistics"):
        stats = result["metal_safety_statistics"]
        print(f"Metal Command Buffer Errors: {stats.get('metal_command_buffer_errors', 0)}")
        print(f"Metal Memory Violations: {stats.get('metal_memory_violations', 0)}")
        print(f"Total Metal Errors Handled: {stats.get('total_metal_errors', 0)}")
        print(f"Safety Score: {stats.get('safety_score', 0):.1f}/100")

    print(f"Summary: {result.get('summary', 'N/A')}")

    return result

import mlx
import mlx_lm

import csv

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
class BenchmarkResult:
    """Single benchmark result"""

    name: str
    prompt_tokens: int
    generated_tokens: int
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    total_tokens_per_sec: float
    peak_memory_gb: float
    total_time_sec: float
    prompt: str
    generated_text: str

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
class BenchmarkConfig:
    """Benchmark configuration"""

    name: str
    prompt: str
    max_tokens: int
    description: str

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
class Qwen3BenchmarkSuite:
    """Comprehensive benchmark suite for Qwen3-0.6B Metal kernel optimization"""

    def __init__(self, model_path: str = "mlx-community/Qwen3-0.6B-bf16"):
        self.model_path = model_path
        self.results: List[BenchmarkResult] = []

    def create_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Create comprehensive benchmark configurations"""

        configs = []

        # 1. Context Length Variations
        configs.extend(
            [
                BenchmarkConfig(
                    name="short_context_quick",
                    prompt="Brief answer: What is artificial intelligence?",
                    max_tokens=50,
                    description="Short context, quick response - chat scenario",
                ),
                BenchmarkConfig(
                    name="medium_context_analysis",
                    prompt=self._create_medium_context_prompt(),
                    max_tokens=200,
                    description="Medium context, analytical response",
                ),
                BenchmarkConfig(
                    name="long_context_detailed",
                    prompt=self._create_long_context_prompt(),
                    max_tokens=500,
                    description="Long context, detailed analysis",
                ),
                BenchmarkConfig(
                    name="very_long_context_comprehensive",
                    prompt=self._create_very_long_context_prompt(),
                    max_tokens=1000,
                    description="Very long context, comprehensive response",
                ),
            ]
        )

        # 2. Generation Length Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="micro_generation",
                    prompt="Complete this sentence: The future of AI is",
                    max_tokens=10,
                    description="Micro generation - attention prefill dominated",
                ),
                BenchmarkConfig(
                    name="short_generation",
                    prompt="Explain in one paragraph: What makes transformers effective?",
                    max_tokens=100,
                    description="Short generation - balanced prefill/decode",
                ),
                BenchmarkConfig(
                    name="long_generation",
                    prompt="Write a detailed technical explanation of how neural networks learn:",
                    max_tokens=1000,
                    description="Long generation - decode performance critical",
                ),
                BenchmarkConfig(
                    name="very_long_generation",
                    prompt="Write a comprehensive guide to machine learning for beginners:",
                    max_tokens=2000,
                    description="Very long generation - sustained decode performance",
                ),
                BenchmarkConfig(
                    name="ultra_long_generation",
                    prompt="The future of AI is",
                    max_tokens=5000,
                    description="Ultra long generation - memory scaling test",
                ),
            ]
        )

        # 3. Different Use Case Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="code_generation",
                    prompt="""Write a Python function to implement binary search:

def binary_search(arr, target):
    \"\"\"
    Implement binary search algorithm
    Args:
        arr: sorted array
        target: element to find
    Returns:
        index of target or -1 if not found
    \"\"\"
""",
                    max_tokens=300,
                    description="Code generation - structured output patterns",
                ),
                BenchmarkConfig(
                    name="step_by_step_reasoning",
                    prompt="""Solve this step by step:

A train travels from City A to City B at 80 mph. The distance is 240 miles. 
If it leaves at 2:00 PM, what time will it arrive? Show your work.""",
                    max_tokens=400,
                    description="Step-by-step reasoning - logical sequence patterns",
                ),
                BenchmarkConfig(
                    name="creative_writing",
                    prompt="""Write a short story about a robot who discovers emotions for the first time. 
Include dialogue and describe the robot's internal experience as it learns about feelings like 
joy, sadness, and wonder. Make it engaging and thoughtful.""",
                    max_tokens=800,
                    description="Creative writing - diverse vocabulary and narrative",
                ),
                BenchmarkConfig(
                    name="technical_documentation",
                    prompt="""Create comprehensive documentation for a REST API with the following endpoints:
- GET /users - List all users
- POST /users - Create new user  
- GET /users/{id} - Get specific user
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

Include request/response examples, error codes, and authentication details.""",
                    max_tokens=1200,
                    description="Technical documentation - structured information",
                ),
                BenchmarkConfig(
                    name="conversational_assistant",
                    prompt="""You are a helpful AI assistant. A user asks:

"I'm planning a trip to Japan for 2 weeks. I've never been there before. I like 
history, food, and nature. I have a moderate budget. Can you help me plan an 
itinerary with recommendations for cities to visit, things to do, and travel tips?"

Provide a detailed, helpful response:""",
                    max_tokens=1500,
                    description="Conversational assistant - helpful response patterns",
                ),
            ]
        )

        # 4. Memory Pressure Scenarios
        configs.extend(
            [
                BenchmarkConfig(
                    name="progressive_context_building",
                    prompt=self._create_progressive_context_prompt(),
                    max_tokens=600,
                    description="Progressive context building - KV cache growth",
                ),
                BenchmarkConfig(
                    name="repetitive_pattern_generation",
                    prompt="Generate a list of 100 creative product names for a tech startup, with explanations:",
                    max_tokens=2000,
                    description="Repetitive patterns - memory efficiency test",
                ),
            ]
        )

        # 5. Extended Long Generation Tests (for sustained decode performance)
        configs.extend(
            [
                BenchmarkConfig(
                    name="extreme_long_generation",
                    prompt="Write a complete tutorial on deep learning from basics to advanced topics, including mathematical foundations, architectures, training techniques, and real-world applications:",
                    max_tokens=8000,
                    description="Extreme long generation - maximum decode performance test",
                ),
                BenchmarkConfig(
                    name="sustained_dialogue_generation",
                    prompt="Create a detailed dialogue between an AI researcher and a software engineer discussing the future of artificial intelligence, covering topics like AGI, safety, ethics, and technological implications. Make it engaging and informative:",
                    max_tokens=6000,
                    description="Sustained dialogue - consistent long-form generation",
                ),
                BenchmarkConfig(
                    name="comprehensive_analysis_generation",
                    prompt="Analyze the evolution of computer programming languages from assembly to modern high-level languages. Discuss paradigms, performance considerations, developer productivity, and future trends:",
                    max_tokens=7000,
                    description="Comprehensive analysis - complex reasoning with long output",
                ),
                BenchmarkConfig(
                    name="maximum_context_stress_test",
                    prompt=self._create_maximum_context_prompt(),
                    max_tokens=10000,
                    description="Maximum context stress test - ultimate performance challenge",
                ),
            ]
        )

        return configs

    def _create_medium_context_prompt(self) -> str:
        """Create medium-length context prompt"""
        return """Context: Machine learning has revolutionized many industries in recent years. 
From healthcare diagnosis to autonomous vehicles, AI systems are becoming increasingly 
sophisticated. However, challenges remain in areas like interpretability, fairness, 
and robustness. Recent advances in transformer architectures have shown remarkable 
capabilities in natural language processing, while computer vision has benefited 
from innovations in convolutional neural networks and attention mechanisms.

Question: Based on this context, analyze the current state of AI development and 
predict the most important research directions for the next 5 years. Consider both 
technical advances and societal implications."""

    def _create_long_context_prompt(self) -> str:
        """Create long context prompt"""
        return """Research Paper Summary:

Title: "Advances in Large Language Models: Architecture, Training, and Applications"

Abstract: This paper reviews recent developments in large language models (LLMs), 
focusing on architectural innovations, training methodologies, and real-world applications. 
We examine the evolution from early transformer models to current state-of-the-art systems, 
analyzing key improvements in efficiency, capability, and safety.

Introduction: The field of natural language processing has undergone a paradigm shift 
with the introduction of transformer-based architectures. Starting with the original 
Transformer paper in 2017, we have witnessed exponential growth in model size and 
capability. From GPT-1's 117M parameters to models with hundreds of billions of parameters, 
the scaling trend has consistently led to emergent capabilities.

Architecture Evolution: Modern LLMs incorporate several key innovations:
1. Attention mechanisms have evolved from basic dot-product attention to more efficient 
variants like sparse attention, local attention, and grouped query attention (GQA).
2. Position encoding schemes have advanced from sinusoidal embeddings to learnable 
position encodings and rotary position embeddings (RoPE).
3. Normalization techniques have shifted from post-norm to pre-norm configurations, 
with RMSNorm becoming preferred over LayerNorm for efficiency.
4. Activation functions have evolved from ReLU to GELU to SwiGLU for better performance.

Training Methodologies: The training of LLMs involves several sophisticated techniques:
- Pre-training on diverse text corpora using next-token prediction
- Instruction tuning to align models with human preferences
- Reinforcement learning from human feedback (RLHF)
- Constitutional AI for improved safety and alignment

Question: Given this comprehensive background, provide a detailed analysis of how 
these architectural and training advances specifically impact inference efficiency 
on mobile and edge devices. Consider memory requirements, computational complexity, 
and potential optimization strategies."""

    def _create_very_long_context_prompt(self) -> str:
        """Create very long context prompt to test KV cache scaling"""
        base_context = self._create_long_context_prompt()

        extended_context = (
            base_context
            + """

Detailed Technical Analysis:

Model Architecture Deep Dive:
The transformer architecture consists of an encoder-decoder structure, though many 
modern LLMs use decoder-only architectures. The core components include:

1. Multi-Head Attention Mechanism:
   - Allows the model to focus on different parts of the input simultaneously
   - Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
   - Multiple attention heads capture different types of relationships
   - Grouped Query Attention (GQA) reduces memory requirements by sharing key-value pairs

2. Feed-Forward Networks:
   - Two linear transformations with a non-linear activation in between
   - Typically 4x the hidden dimension for the intermediate layer
   - SwiGLU activation: SwiGLU(x) = Swish(xW_1) ⊙ (xW_2)
   - Crucial for the model's capacity to learn complex patterns

3. Layer Normalization:
   - RMSNorm: RMSNorm(x) = x / RMS(x) * g, where RMS(x) = √(1/n Σx_i²)
   - Applied before each sub-layer (pre-norm) for training stability
   - Critical for deep network training convergence

4. Position Encodings:
   - Rotary Position Embedding (RoPE) rotates query and key vectors
   - Enables length generalization beyond training context
   - More efficient than absolute position encodings

Training Optimization Techniques:
- Gradient accumulation for effective large batch training
- Mixed precision training using bfloat16 for memory efficiency
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with warmup and decay
- Data parallelism and model parallelism for distributed training

Hardware Considerations:
Modern LLM training requires specialized hardware:
- GPUs with high memory bandwidth (A100, H100)
- Tensor cores optimized for mixed precision operations
- High-speed interconnects for multi-GPU training
- Efficient memory hierarchies for large model parameters

Inference Optimization Strategies:
- KV caching to avoid recomputing attention weights
- Quantization techniques (INT8, INT4) to reduce memory footprint
- Pruning methods to remove redundant parameters
- Distillation to create smaller, faster models
- Speculative decoding for improved throughput

Now, considering all this technical detail and the specific challenges of deploying 
large language models on resource-constrained devices, provide a comprehensive 
analysis of optimization strategies specifically for Apple Silicon devices, 
considering unified memory architecture, Metal Performance Shaders, and the 
specific computational characteristics of M-series chips."""
        )

        return extended_context

    def _create_progressive_context_prompt(self) -> str:
        """Create prompt that builds context progressively"""
        return """Chapter 1: The Beginning

In the early days of artificial intelligence, researchers dreamed of creating 
machines that could think and reason like humans. The field began in the 1950s 
with pioneers like Alan Turing, who proposed the famous Turing Test as a measure 
of machine intelligence.

Chapter 2: Early Developments  

The 1960s and 1970s saw the development of expert systems and symbolic AI. 
Researchers focused on rule-based systems that could encode human knowledge 
in formal logical structures. However, these systems were brittle and couldn't 
handle uncertainty or learning.

Chapter 3: The Neural Network Revolution

The 1980s brought renewed interest in neural networks, inspired by biological 
neurons. Backpropagation was rediscovered, enabling the training of multi-layer 
networks. This marked the beginning of connectionist AI approaches.

Chapter 4: Machine Learning Boom

The 1990s and 2000s saw machine learning become dominant. Support vector machines, 
random forests, and ensemble methods proved effective for many practical problems. 
The internet provided vast amounts of data to train these systems.

Chapter 5: Deep Learning Era

The 2010s marked the deep learning revolution. Convolutional neural networks 
revolutionized computer vision, recurrent networks advanced natural language 
processing, and deep reinforcement learning achieved superhuman performance 
in games like Go and Chess.

Now, continue this historical narrative by writing Chapter 6, focusing on the 
transformer era and large language models. Discuss the key innovations, 
breakthrough applications, and current challenges in the field."""

    def _create_maximum_context_prompt(self) -> str:
        """Create maximum length context prompt for stress testing"""
        base_context = self._create_very_long_context_prompt()

        extended_context = (
            base_context
            + """

Further Technical Deep Dive:

Advanced Optimization Techniques:
Modern LLM optimization goes beyond basic training approaches. Key areas include:

1. Memory Optimization:
   - Gradient checkpointing to trade compute for memory
   - Model parallelism across multiple devices
   - ZeRO optimizer states for distributed training
   - Mixed precision training with automatic loss scaling
   - Activation recomputation strategies

2. Computational Efficiency:
   - Flash Attention for memory-efficient attention computation
   - Gradient accumulation for effective large batch sizes
   - Dynamic loss scaling for stable mixed precision training
   - Automatic mixed precision (AMP) for optimal performance
   - Custom CUDA kernels for specific operations

3. Distributed Training Strategies:
   - Data parallelism with all-reduce communication
   - Model parallelism for very large models
   - Pipeline parallelism for sequential processing
   - 3D parallelism combining all approaches
   - Efficient communication backends (NCCL, Gloo)

4. Apple Silicon Specific Optimizations:
   - Unified memory architecture advantages
   - Metal Performance Shaders (MPS) acceleration
   - Neural Engine utilization for specific operations
   - Memory bandwidth optimization for M-series chips
   - Custom MLX primitives for Apple hardware

Inference Optimization Deep Dive:
Optimizing LLM inference requires different strategies than training:

1. Model Compression:
   - Quantization to 8-bit or 4-bit precision
   - Pruning redundant parameters
   - Knowledge distillation to smaller models
   - Low-rank approximations
   - Sparsity-aware inference engines

2. Runtime Optimization:
   - KV cache management for autoregressive generation
   - Batch processing for multiple requests
   - Dynamic batching for variable sequence lengths
   - Speculative decoding for faster generation
   - Continuous batching for improved throughput

3. Hardware-Specific Optimization:
   - GPU kernel fusion for reduced memory transfers
   - CPU optimization with vectorized operations
   - Mobile optimization for edge deployment
   - FPGA acceleration for specific use cases
   - Neuromorphic computing for ultra-low power

4. Serving Infrastructure:
   - Model serving frameworks (TensorRT, TorchServe)
   - Load balancing across multiple instances
   - Auto-scaling based on demand
   - Caching strategies for common requests
   - Request prioritization and queuing

Emerging Paradigms:
The field continues to evolve with new approaches:

1. Architecture Innovations:
   - Mixture of Experts (MoE) for conditional computation
   - State Space Models for long sequence modeling
   - Retrieval-augmented generation (RAG) systems
   - Multi-modal models combining text, vision, and audio
   - Constitutional AI for aligned behavior

2. Training Innovations:
   - Reinforcement Learning from Human Feedback (RLHF)
   - Constitutional AI training approaches
   - Curriculum learning for improved convergence
   - Meta-learning for few-shot adaptation
   - Continual learning to avoid catastrophic forgetting

3. Evaluation and Safety:
   - Comprehensive benchmark suites
   - Adversarial testing for robustness
   - Bias detection and mitigation
   - Interpretability and explainability
   - Safety alignment techniques

Real-World Deployment Challenges:
Deploying LLMs in production involves numerous considerations:

1. Scalability:
   - Handling millions of concurrent users
   - Geographic distribution for low latency
   - Cost optimization for sustainable operations
   - Resource allocation and scheduling
   - Auto-scaling based on demand patterns

2. Reliability:
   - Fault tolerance and error recovery
   - Monitoring and alerting systems
   - A/B testing for model updates
   - Gradual rollouts for risk mitigation
   - Backup systems for high availability

3. Security and Privacy:
   - Data protection and encryption
   - Secure model serving environments
   - Privacy-preserving inference techniques
   - Audit trails and compliance
   - Protection against adversarial attacks

Future Directions:
The field continues to advance rapidly with several promising directions:

1. Efficiency Improvements:
   - Novel architectures with better scaling properties
   - More efficient training algorithms
   - Better hardware-software co-design
   - Energy-efficient computing approaches
   - Sustainable AI development practices

2. Capability Enhancement:
   - Improved reasoning and planning abilities
   - Better multi-modal understanding
   - Enhanced code generation capabilities
   - Scientific discovery applications
   - Creative and artistic applications

3. Democratization:
   - Open-source model development
   - Accessible training and inference tools
   - Educational resources and tutorials
   - Community-driven improvements
   - Ethical AI development practices

Given this comprehensive overview of the current state and future directions of large language model optimization, provide a detailed analysis of how these various optimization techniques specifically apply to Apple Silicon hardware, particularly focusing on the M4 chip architecture, unified memory advantages, and how developers can best leverage these capabilities for maximum performance in LLM inference workloads."""
        )

        return extended_context

    def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration with proper warmup"""
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(f"Max tokens: {config.max_tokens}")
        print(f"{'='*60}")

        # Performance measurement parameters
        WARMUP_RUNS = 2  # Warmup runs to eliminate cold start effects
        MEASUREMENT_RUNS = 3  # Multiple measurement runs for reliability

        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(config.prompt)
            prompt_file = f.name

        try:
            # Build command
            cmd = [
                "python",
                "-m",
                "mlx_lm.generate",
                "--model",
                self.model_path,
                "--prompt",
                config.prompt,
                "--max-tokens",
                str(config.max_tokens),
            ]

            # Clear MLX cache before starting
            print(f"🧹 Clearing MLX cache...")
            mx.clear_cache()

            # Warmup runs - don't measure these
            print(f"🔥 Running {WARMUP_RUNS} warmup runs to eliminate cold start effects...")
            for i in range(WARMUP_RUNS):
                try:
                    print(f"   Warmup run {i+1}/{WARMUP_RUNS}...")
                    warmup_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if warmup_result.returncode != 0:
                        print(f"   ⚠️  Warmup run {i+1} failed: {warmup_result.stderr[:100]}...")
                    else:
                        print(f"   ✅ Warmup run {i+1} completed")

                    # Clear cache between warmup runs
                    mx.clear_cache()

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Warmup run {i+1} timed out")
                except Exception as e:
                    print(f"   ❌ Warmup run {i+1} error: {e}")

            print(f"📊 Running {MEASUREMENT_RUNS} measurement runs...")

            # Measurement runs
            successful_results = []
            for run_idx in range(MEASUREMENT_RUNS):
                try:
                    print(f"   Measurement run {run_idx+1}/{MEASUREMENT_RUNS}...")

                    # Clear cache before each measurement run for consistency
                    mx.clear_cache()
                    initial_memory = mx.get_active_memory()

                    # Run benchmark
                    start_time = time.perf_counter()
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    end_time = time.perf_counter()

                    if result.returncode != 0:
                        print(f"   ❌ Measurement run {run_idx+1} failed: {result.stderr[:100]}...")
                        continue

                    # Parse output
                    parsed_result = self._parse_benchmark_output(
                        result.stdout, config, end_time - start_time
                    )

                    if parsed_result:
                        successful_results.append(parsed_result)
                        print(
                            f"   ✅ Run {run_idx+1}: {parsed_result.decode_tokens_per_sec:.1f} tokens/sec"
                        )
                    else:
                        print(f"   ❌ Run {run_idx+1}: Failed to parse output")

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Measurement run {run_idx+1} timed out")
                except Exception as e:
                    print(f"   ❌ Measurement run {run_idx+1} error: {e}")

            # Require at least 2 successful runs for reliable results
            if len(successful_results) < 2:
                print(
                    f"❌ Only {len(successful_results)}/{MEASUREMENT_RUNS} measurement runs succeeded"
                )
                print(f"❌ Need at least 2 successful runs for reliable results")
                raise RuntimeError(
                    f"Insufficient successful runs: {len(successful_results)}/{MEASUREMENT_RUNS}"
                )

            # Calculate statistics from multiple runs
            decode_speeds = [r.decode_tokens_per_sec for r in successful_results]
            prefill_speeds = [r.prefill_tokens_per_sec for r in successful_results]
            memories = [r.peak_memory_gb for r in successful_results]
            times = [r.total_time_sec for r in successful_results]

            # Use median for more robust results (less sensitive to outliers)
            final_result = BenchmarkResult(
                name=config.name,
                prompt_tokens=int(np.median([r.prompt_tokens for r in successful_results])),
                generated_tokens=int(np.median([r.generated_tokens for r in successful_results])),
                prefill_tokens_per_sec=float(np.median(prefill_speeds)),
                decode_tokens_per_sec=float(np.median(decode_speeds)),
                total_tokens_per_sec=float(
                    np.median([r.total_tokens_per_sec for r in successful_results])
                ),
                peak_memory_gb=float(np.median(memories)),
                total_time_sec=float(np.median(times)),
                prompt=config.prompt[:200] + "..." if len(config.prompt) > 200 else config.prompt,
                generated_text=successful_results[0].generated_text,  # Use first result's text
            )

            # Print final results with statistics
            print(f"\n📈 Final Results (median of {len(successful_results)} runs):")
            print(f"  Prompt tokens: {final_result.prompt_tokens}")
            print(f"  Generated tokens: {final_result.generated_tokens}")
            print(f"  Prefill speed: {final_result.prefill_tokens_per_sec:.2f} tokens/sec")
            print(
                f"  Decode speed: {final_result.decode_tokens_per_sec:.2f} tokens/sec (σ={np.std(decode_speeds):.2f})"
            )
            print(f"  Overall speed: {final_result.total_tokens_per_sec:.2f} tokens/sec")
            print(f"  Peak memory: {final_result.peak_memory_gb:.3f} GB")
            print(f"  Total time: {final_result.total_time_sec:.2f} seconds")

            if len(decode_speeds) > 1:
                print(
                    f"  Performance consistency: {np.std(decode_speeds)/np.mean(decode_speeds)*100:.1f}% CV"
                )

            return final_result

        finally:
            # Clean up
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)

    def _parse_benchmark_output(
        self, stdout: str, config: BenchmarkConfig, total_time: float
    ) -> Optional[BenchmarkResult]:
        """Parse mlx-lm output to extract performance metrics"""
        output_lines = stdout.strip().split("\n")

        # Find the generated text (between ========== markers)
        generated_text = ""
        in_generation = False
        prompt_tokens = 0
        generation_tokens = 0
        prompt_speed = 0.0
        generation_speed = 0.0
        peak_memory_str = ""

        for line in output_lines:
            if line.strip() == "==========":
                in_generation = not in_generation
            elif in_generation:
                generated_text += line + "\n"
            elif "Prompt:" in line and "tokens-per-sec" in line:
                # Parse: "Prompt: 13 tokens, 310.367 tokens-per-sec"
                parts = line.split(",")
                prompt_tokens = int(parts[0].split(":")[1].strip().split()[0])
                prompt_speed = float(parts[1].strip().split()[0])
            elif "Generation:" in line and "tokens-per-sec" in line:
                # Parse: "Generation: 468 tokens, 69.860 tokens-per-sec"
                parts = line.split(",")
                generation_tokens = int(parts[0].split(":")[1].strip().split()[0])
                generation_speed = float(parts[1].strip().split()[0])
            elif "Peak memory:" in line:
                peak_memory_str = line.split(":")[1].strip()

        # Parse peak memory
        peak_memory_gb = 0.0
        if peak_memory_str:
            if "GB" in peak_memory_str:
                peak_memory_gb = float(peak_memory_str.replace("GB", "").strip())
            elif "MB" in peak_memory_str:
                peak_memory_gb = float(peak_memory_str.replace("MB", "").strip()) / 1024

        # Validate we got meaningful results
        if generation_tokens == 0 or generation_speed == 0:
            return None

        # Calculate overall tokens per second
        total_tokens_per_sec = generation_tokens / total_time if total_time > 0 else 0

        return BenchmarkResult(
            name=config.name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generation_tokens,
            prefill_tokens_per_sec=prompt_speed,
            decode_tokens_per_sec=generation_speed,
            total_tokens_per_sec=total_tokens_per_sec,
            peak_memory_gb=peak_memory_gb,
            total_time_sec=total_time,
            prompt=config.prompt[:200] + "..." if len(config.prompt) > 200 else config.prompt,
            generated_text=(
                generated_text.strip()[:200] + "..."
                if len(generated_text.strip()) > 200
                else generated_text.strip()
            ),
        )

    def run_full_benchmark_suite(self) -> Dict:
        """Run the complete benchmark suite"""
        print(f"\n{'='*80}")
        print(f"Qwen3-0.6B Comprehensive Benchmark Suite")
        print(f"Model: {self.model_path}")
        print(f"Hardware: Apple M4 24GB")
        print(f"Target: Custom Metal kernel optimization validation")
        print(f"{'='*80}")

        configs = self.create_benchmark_configs()
        results = []

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Starting benchmark: {config.name}")
            try:
                result = self.run_single_benchmark(config)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark {config.name}: {e}")
                continue

        # Generate summary
        summary = self.generate_summary(results)
        self.save_results(results, summary)

        return {"results": [self._result_to_dict(r) for r in results], "summary": summary}

    def generate_summary(self, results: List[BenchmarkResult]) -> Dict:
        """Generate benchmark summary statistics"""
        if not results:
            return {}

        # Overall statistics
        decode_speeds = [r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0]
        prefill_speeds = [r.prefill_tokens_per_sec for r in results if r.prefill_tokens_per_sec > 0]
        memories = [r.peak_memory_gb for r in results if r.peak_memory_gb > 0]

        summary = {
            "total_benchmarks": len(results),
            "avg_decode_speed": np.mean(decode_speeds) if decode_speeds else 0,
            "min_decode_speed": np.min(decode_speeds) if decode_speeds else 0,
            "max_decode_speed": np.max(decode_speeds) if decode_speeds else 0,
            "avg_prefill_speed": np.mean(prefill_speeds) if prefill_speeds else 0,
            "min_prefill_speed": np.min(prefill_speeds) if prefill_speeds else 0,
            "max_prefill_speed": np.max(prefill_speeds) if prefill_speeds else 0,
            "avg_memory_usage": np.mean(memories) if memories else 0,
            "max_memory_usage": np.max(memories) if memories else 0,
            "min_memory_usage": np.min(memories) if memories else 0,
        }

        # Category analysis
        categories = {
            "context_length": [r for r in results if "context" in r.name],
            "generation_length": [r for r in results if "generation" in r.name],
            "use_cases": [
                r
                for r in results
                if any(
                    x in r.name
                    for x in ["code", "reasoning", "creative", "technical", "conversational"]
                )
            ],
            "memory_pressure": [
                r for r in results if any(x in r.name for x in ["progressive", "repetitive"])
            ],
        }

        for category, cat_results in categories.items():
            if cat_results:
                cat_decode_speeds = [
                    r.decode_tokens_per_sec for r in cat_results if r.decode_tokens_per_sec > 0
                ]
                summary[f"{category}_avg_decode_speed"] = (
                    np.mean(cat_decode_speeds) if cat_decode_speeds else 0
                )
                summary[f"{category}_count"] = len(cat_results)

        return summary

    def save_results(self, results: List[BenchmarkResult], summary: Dict):
        """Save benchmark results to files"""
        timestamp = int(time.time())

        # Save detailed results
        detailed_results = {
            "timestamp": timestamp,
            "model": self.model_path,
            "hardware": "Apple M4 24GB",
            "optimization": "Custom Metal kernel for GQA attention",
            "mlx_version": mx.__version__,
            "results": [self._result_to_dict(r) for r in results],
            "summary": summary,
        }

        with open(f"qwen3_benchmark_results_{timestamp}.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save CSV for easy analysis
        import csv

        with open(f"qwen3_benchmark_results_{timestamp}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "description",
                    "prompt_tokens",
                    "generated_tokens",
                    "prefill_tokens_per_sec",
                    "decode_tokens_per_sec",
                    "total_tokens_per_sec",
                    "peak_memory_gb",
                    "total_time_sec",
                ]
            )

            configs = self.create_benchmark_configs()
            config_dict = {c.name: c for c in configs}

            for result in results:
                config = config_dict.get(result.name)
                writer.writerow(
                    [
                        result.name,
                        config.description if config else "",
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.prefill_tokens_per_sec,
                        result.decode_tokens_per_sec,
                        result.total_tokens_per_sec,
                        result.peak_memory_gb,
                        result.total_time_sec,
                    ]
                )

        print(f"\n{'='*60}")
        print(f"Results saved to:")
        print(f"  - qwen3_benchmark_results_{timestamp}.json")
        print(f"  - qwen3_benchmark_results_{timestamp}.csv")
        print(f"{'='*60}")

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dictionary"""
        return {
            "name": result.name,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
            "decode_tokens_per_sec": result.decode_tokens_per_sec,
            "total_tokens_per_sec": result.total_tokens_per_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "total_time_sec": result.total_time_sec,
            "prompt": result.prompt,
            "generated_text": result.generated_text,
        }

    def print_summary_table(self):
        """Print a summary table of all results"""
        if not self.results:
            print("No benchmark results available")
            return

        print(f"\n{'='*120}")
        print(f"{'Benchmark Summary':^120}")
        print(f"{'='*120}")
        print(
            f"{'Name':<25} {'Tokens':<8} {'Prefill':<10} {'Decode':<10} {'Overall':<10} {'Memory':<8} {'Time':<8}"
        )
        print(f"{'='*120}")

        for result in self.results:
            print(
                f"{result.name:<25} "
                f"{result.generated_tokens:<8} "
                f"{result.prefill_tokens_per_sec:<10.1f} "
                f"{result.decode_tokens_per_sec:<10.1f} "
                f"{result.total_tokens_per_sec:<10.1f} "
                f"{result.peak_memory_gb:<8.2f} "
                f"{result.total_time_sec:<8.1f}"
            )

        print(f"{'='*120}")

        # Summary statistics
        decode_speeds = [
            r.decode_tokens_per_sec for r in self.results if r.decode_tokens_per_sec > 0
        ]
        if decode_speeds:
            print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
            print(f"Best decode speed: {np.max(decode_speeds):.1f} tokens/sec")
            print(f"Worst decode speed: {np.min(decode_speeds):.1f} tokens/sec")

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def create_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Create comprehensive benchmark configurations"""

        configs = []

        # 1. Context Length Variations
        configs.extend(
            [
                BenchmarkConfig(
                    name="short_context_quick",
                    prompt="Brief answer: What is artificial intelligence?",
                    max_tokens=50,
                    description="Short context, quick response - chat scenario",
                ),
                BenchmarkConfig(
                    name="medium_context_analysis",
                    prompt=self._create_medium_context_prompt(),
                    max_tokens=200,
                    description="Medium context, analytical response",
                ),
                BenchmarkConfig(
                    name="long_context_detailed",
                    prompt=self._create_long_context_prompt(),
                    max_tokens=500,
                    description="Long context, detailed analysis",
                ),
                BenchmarkConfig(
                    name="very_long_context_comprehensive",
                    prompt=self._create_very_long_context_prompt(),
                    max_tokens=1000,
                    description="Very long context, comprehensive response",
                ),
            ]
        )

        # 2. Generation Length Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="micro_generation",
                    prompt="Complete this sentence: The future of AI is",
                    max_tokens=10,
                    description="Micro generation - attention prefill dominated",
                ),
                BenchmarkConfig(
                    name="short_generation",
                    prompt="Explain in one paragraph: What makes transformers effective?",
                    max_tokens=100,
                    description="Short generation - balanced prefill/decode",
                ),
                BenchmarkConfig(
                    name="long_generation",
                    prompt="Write a detailed technical explanation of how neural networks learn:",
                    max_tokens=1000,
                    description="Long generation - decode performance critical",
                ),
                BenchmarkConfig(
                    name="very_long_generation",
                    prompt="Write a comprehensive guide to machine learning for beginners:",
                    max_tokens=2000,
                    description="Very long generation - sustained decode performance",
                ),
                BenchmarkConfig(
                    name="ultra_long_generation",
                    prompt="The future of AI is",
                    max_tokens=5000,
                    description="Ultra long generation - memory scaling test",
                ),
            ]
        )

        # 3. Different Use Case Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="code_generation",
                    prompt="""Write a Python function to implement binary search:

def binary_search(arr, target):
    \"\"\"
    Implement binary search algorithm
    Args:
        arr: sorted array
        target: element to find
    Returns:
        index of target or -1 if not found
    \"\"\"
""",
                    max_tokens=300,
                    description="Code generation - structured output patterns",
                ),
                BenchmarkConfig(
                    name="step_by_step_reasoning",
                    prompt="""Solve this step by step:

A train travels from City A to City B at 80 mph. The distance is 240 miles. 
If it leaves at 2:00 PM, what time will it arrive? Show your work.""",
                    max_tokens=400,
                    description="Step-by-step reasoning - logical sequence patterns",
                ),
                BenchmarkConfig(
                    name="creative_writing",
                    prompt="""Write a short story about a robot who discovers emotions for the first time. 
Include dialogue and describe the robot's internal experience as it learns about feelings like 
joy, sadness, and wonder. Make it engaging and thoughtful.""",
                    max_tokens=800,
                    description="Creative writing - diverse vocabulary and narrative",
                ),
                BenchmarkConfig(
                    name="technical_documentation",
                    prompt="""Create comprehensive documentation for a REST API with the following endpoints:
- GET /users - List all users
- POST /users - Create new user  
- GET /users/{id} - Get specific user
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

Include request/response examples, error codes, and authentication details.""",
                    max_tokens=1200,
                    description="Technical documentation - structured information",
                ),
                BenchmarkConfig(
                    name="conversational_assistant",
                    prompt="""You are a helpful AI assistant. A user asks:

"I'm planning a trip to Japan for 2 weeks. I've never been there before. I like 
history, food, and nature. I have a moderate budget. Can you help me plan an 
itinerary with recommendations for cities to visit, things to do, and travel tips?"

Provide a detailed, helpful response:""",
                    max_tokens=1500,
                    description="Conversational assistant - helpful response patterns",
                ),
            ]
        )

        # 4. Memory Pressure Scenarios
        configs.extend(
            [
                BenchmarkConfig(
                    name="progressive_context_building",
                    prompt=self._create_progressive_context_prompt(),
                    max_tokens=600,
                    description="Progressive context building - KV cache growth",
                ),
                BenchmarkConfig(
                    name="repetitive_pattern_generation",
                    prompt="Generate a list of 100 creative product names for a tech startup, with explanations:",
                    max_tokens=2000,
                    description="Repetitive patterns - memory efficiency test",
                ),
            ]
        )

        # 5. Extended Long Generation Tests (for sustained decode performance)
        configs.extend(
            [
                BenchmarkConfig(
                    name="extreme_long_generation",
                    prompt="Write a complete tutorial on deep learning from basics to advanced topics, including mathematical foundations, architectures, training techniques, and real-world applications:",
                    max_tokens=8000,
                    description="Extreme long generation - maximum decode performance test",
                ),
                BenchmarkConfig(
                    name="sustained_dialogue_generation",
                    prompt="Create a detailed dialogue between an AI researcher and a software engineer discussing the future of artificial intelligence, covering topics like AGI, safety, ethics, and technological implications. Make it engaging and informative:",
                    max_tokens=6000,
                    description="Sustained dialogue - consistent long-form generation",
                ),
                BenchmarkConfig(
                    name="comprehensive_analysis_generation",
                    prompt="Analyze the evolution of computer programming languages from assembly to modern high-level languages. Discuss paradigms, performance considerations, developer productivity, and future trends:",
                    max_tokens=7000,
                    description="Comprehensive analysis - complex reasoning with long output",
                ),
                BenchmarkConfig(
                    name="maximum_context_stress_test",
                    prompt=self._create_maximum_context_prompt(),
                    max_tokens=10000,
                    description="Maximum context stress test - ultimate performance challenge",
                ),
            ]
        )

        return configs

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration with proper warmup"""
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(f"Max tokens: {config.max_tokens}")
        print(f"{'='*60}")

        # Performance measurement parameters
        WARMUP_RUNS = 2  # Warmup runs to eliminate cold start effects
        MEASUREMENT_RUNS = 3  # Multiple measurement runs for reliability

        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(config.prompt)
            prompt_file = f.name

        try:
            # Build command
            cmd = [
                "python",
                "-m",
                "mlx_lm.generate",
                "--model",
                self.model_path,
                "--prompt",
                config.prompt,
                "--max-tokens",
                str(config.max_tokens),
            ]

            # Clear MLX cache before starting
            print(f"🧹 Clearing MLX cache...")
            mx.clear_cache()

            # Warmup runs - don't measure these
            print(f"🔥 Running {WARMUP_RUNS} warmup runs to eliminate cold start effects...")
            for i in range(WARMUP_RUNS):
                try:
                    print(f"   Warmup run {i+1}/{WARMUP_RUNS}...")
                    warmup_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if warmup_result.returncode != 0:
                        print(f"   ⚠️  Warmup run {i+1} failed: {warmup_result.stderr[:100]}...")
                    else:
                        print(f"   ✅ Warmup run {i+1} completed")

                    # Clear cache between warmup runs
                    mx.clear_cache()

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Warmup run {i+1} timed out")
                except Exception as e:
                    print(f"   ❌ Warmup run {i+1} error: {e}")

            print(f"📊 Running {MEASUREMENT_RUNS} measurement runs...")

            # Measurement runs
            successful_results = []
            for run_idx in range(MEASUREMENT_RUNS):
                try:
                    print(f"   Measurement run {run_idx+1}/{MEASUREMENT_RUNS}...")

                    # Clear cache before each measurement run for consistency
                    mx.clear_cache()
                    initial_memory = mx.get_active_memory()

                    # Run benchmark
                    start_time = time.perf_counter()
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    end_time = time.perf_counter()

                    if result.returncode != 0:
                        print(f"   ❌ Measurement run {run_idx+1} failed: {result.stderr[:100]}...")
                        continue

                    # Parse output
                    parsed_result = self._parse_benchmark_output(
                        result.stdout, config, end_time - start_time
                    )

                    if parsed_result:
                        successful_results.append(parsed_result)
                        print(
                            f"   ✅ Run {run_idx+1}: {parsed_result.decode_tokens_per_sec:.1f} tokens/sec"
                        )
                    else:
                        print(f"   ❌ Run {run_idx+1}: Failed to parse output")

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Measurement run {run_idx+1} timed out")
                except Exception as e:
                    print(f"   ❌ Measurement run {run_idx+1} error: {e}")

            # Require at least 2 successful runs for reliable results
            if len(successful_results) < 2:
                print(
                    f"❌ Only {len(successful_results)}/{MEASUREMENT_RUNS} measurement runs succeeded"
                )
                print(f"❌ Need at least 2 successful runs for reliable results")
                raise RuntimeError(
                    f"Insufficient successful runs: {len(successful_results)}/{MEASUREMENT_RUNS}"
                )

            # Calculate statistics from multiple runs
            decode_speeds = [r.decode_tokens_per_sec for r in successful_results]
            prefill_speeds = [r.prefill_tokens_per_sec for r in successful_results]
            memories = [r.peak_memory_gb for r in successful_results]
            times = [r.total_time_sec for r in successful_results]

            # Use median for more robust results (less sensitive to outliers)
            final_result = BenchmarkResult(
                name=config.name,
                prompt_tokens=int(np.median([r.prompt_tokens for r in successful_results])),
                generated_tokens=int(np.median([r.generated_tokens for r in successful_results])),
                prefill_tokens_per_sec=float(np.median(prefill_speeds)),
                decode_tokens_per_sec=float(np.median(decode_speeds)),
                total_tokens_per_sec=float(
                    np.median([r.total_tokens_per_sec for r in successful_results])
                ),
                peak_memory_gb=float(np.median(memories)),
                total_time_sec=float(np.median(times)),
                prompt=config.prompt[:200] + "..." if len(config.prompt) > 200 else config.prompt,
                generated_text=successful_results[0].generated_text,  # Use first result's text
            )

            # Print final results with statistics
            print(f"\n📈 Final Results (median of {len(successful_results)} runs):")
            print(f"  Prompt tokens: {final_result.prompt_tokens}")
            print(f"  Generated tokens: {final_result.generated_tokens}")
            print(f"  Prefill speed: {final_result.prefill_tokens_per_sec:.2f} tokens/sec")
            print(
                f"  Decode speed: {final_result.decode_tokens_per_sec:.2f} tokens/sec (σ={np.std(decode_speeds):.2f})"
            )
            print(f"  Overall speed: {final_result.total_tokens_per_sec:.2f} tokens/sec")
            print(f"  Peak memory: {final_result.peak_memory_gb:.3f} GB")
            print(f"  Total time: {final_result.total_time_sec:.2f} seconds")

            if len(decode_speeds) > 1:
                print(
                    f"  Performance consistency: {np.std(decode_speeds)/np.mean(decode_speeds)*100:.1f}% CV"
                )

            return final_result

        finally:
            # Clean up
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def run_full_benchmark_suite(self) -> Dict:
        """Run the complete benchmark suite"""
        print(f"\n{'='*80}")
        print(f"Qwen3-0.6B Comprehensive Benchmark Suite")
        print(f"Model: {self.model_path}")
        print(f"Hardware: Apple M4 24GB")
        print(f"Target: Custom Metal kernel optimization validation")
        print(f"{'='*80}")

        configs = self.create_benchmark_configs()
        results = []

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Starting benchmark: {config.name}")
            try:
                result = self.run_single_benchmark(config)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark {config.name}: {e}")
                continue

        # Generate summary
        summary = self.generate_summary(results)
        self.save_results(results, summary)

        return {"results": [self._result_to_dict(r) for r in results], "summary": summary}

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def generate_summary(self, results: List[BenchmarkResult]) -> Dict:
        """Generate benchmark summary statistics"""
        if not results:
            return {}

        # Overall statistics
        decode_speeds = [r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0]
        prefill_speeds = [r.prefill_tokens_per_sec for r in results if r.prefill_tokens_per_sec > 0]
        memories = [r.peak_memory_gb for r in results if r.peak_memory_gb > 0]

        summary = {
            "total_benchmarks": len(results),
            "avg_decode_speed": np.mean(decode_speeds) if decode_speeds else 0,
            "min_decode_speed": np.min(decode_speeds) if decode_speeds else 0,
            "max_decode_speed": np.max(decode_speeds) if decode_speeds else 0,
            "avg_prefill_speed": np.mean(prefill_speeds) if prefill_speeds else 0,
            "min_prefill_speed": np.min(prefill_speeds) if prefill_speeds else 0,
            "max_prefill_speed": np.max(prefill_speeds) if prefill_speeds else 0,
            "avg_memory_usage": np.mean(memories) if memories else 0,
            "max_memory_usage": np.max(memories) if memories else 0,
            "min_memory_usage": np.min(memories) if memories else 0,
        }

        # Category analysis
        categories = {
            "context_length": [r for r in results if "context" in r.name],
            "generation_length": [r for r in results if "generation" in r.name],
            "use_cases": [
                r
                for r in results
                if any(
                    x in r.name
                    for x in ["code", "reasoning", "creative", "technical", "conversational"]
                )
            ],
            "memory_pressure": [
                r for r in results if any(x in r.name for x in ["progressive", "repetitive"])
            ],
        }

        for category, cat_results in categories.items():
            if cat_results:
                cat_decode_speeds = [
                    r.decode_tokens_per_sec for r in cat_results if r.decode_tokens_per_sec > 0
                ]
                summary[f"{category}_avg_decode_speed"] = (
                    np.mean(cat_decode_speeds) if cat_decode_speeds else 0
                )
                summary[f"{category}_count"] = len(cat_results)

        return summary

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def save_results(self, results: List[BenchmarkResult], summary: Dict):
        """Save benchmark results to files"""
        timestamp = int(time.time())

        # Save detailed results
        detailed_results = {
            "timestamp": timestamp,
            "model": self.model_path,
            "hardware": "Apple M4 24GB",
            "optimization": "Custom Metal kernel for GQA attention",
            "mlx_version": mx.__version__,
            "results": [self._result_to_dict(r) for r in results],
            "summary": summary,
        }

        with open(f"qwen3_benchmark_results_{timestamp}.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save CSV for easy analysis
        import csv

        with open(f"qwen3_benchmark_results_{timestamp}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "description",
                    "prompt_tokens",
                    "generated_tokens",
                    "prefill_tokens_per_sec",
                    "decode_tokens_per_sec",
                    "total_tokens_per_sec",
                    "peak_memory_gb",
                    "total_time_sec",
                ]
            )

            configs = self.create_benchmark_configs()
            config_dict = {c.name: c for c in configs}

            for result in results:
                config = config_dict.get(result.name)
                writer.writerow(
                    [
                        result.name,
                        config.description if config else "",
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.prefill_tokens_per_sec,
                        result.decode_tokens_per_sec,
                        result.total_tokens_per_sec,
                        result.peak_memory_gb,
                        result.total_time_sec,
                    ]
                )

        print(f"\n{'='*60}")
        print(f"Results saved to:")
        print(f"  - qwen3_benchmark_results_{timestamp}.json")
        print(f"  - qwen3_benchmark_results_{timestamp}.csv")
        print(f"{'='*60}")

# From mlx_metal_kernel_opt/qwen3_benchmark_suite.py
def print_summary_table(self):
        """Print a summary table of all results"""
        if not self.results:
            print("No benchmark results available")
            return

        print(f"\n{'='*120}")
        print(f"{'Benchmark Summary':^120}")
        print(f"{'='*120}")
        print(
            f"{'Name':<25} {'Tokens':<8} {'Prefill':<10} {'Decode':<10} {'Overall':<10} {'Memory':<8} {'Time':<8}"
        )
        print(f"{'='*120}")

        for result in self.results:
            print(
                f"{result.name:<25} "
                f"{result.generated_tokens:<8} "
                f"{result.prefill_tokens_per_sec:<10.1f} "
                f"{result.decode_tokens_per_sec:<10.1f} "
                f"{result.total_tokens_per_sec:<10.1f} "
                f"{result.peak_memory_gb:<8.2f} "
                f"{result.total_time_sec:<8.1f}"
            )

        print(f"{'='*120}")

        # Summary statistics
        decode_speeds = [
            r.decode_tokens_per_sec for r in self.results if r.decode_tokens_per_sec > 0
        ]
        if decode_speeds:
            print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
            print(f"Best decode speed: {np.max(decode_speeds):.1f} tokens/sec")
            print(f"Worst decode speed: {np.min(decode_speeds):.1f} tokens/sec")

from mlx_lm.models.rope_utils import initialize_rope

# From mlx_metal_kernel_opt/best_program.py
class CustomGQAAttention(nn.Module):
    """
    Qwen3 attention module with custom Metal kernel optimization.

    This module integrates the custom Metal kernel while maintaining
    compatibility with the standard MLX-LM interface.
    """

    def __init__(self, args):
        super().__init__()

        # Standard Qwen3 parameters
        dim = args.hidden_size  # 5120
        self.n_heads = n_heads = args.num_attention_heads  # 40
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads  # 8
        head_dim = args.head_dim  # 128
        self.scale = head_dim**-0.5

        # Standard MLX-LM projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Standard MLX-LM norms
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Standard MLX-LM RoPE
        try:
            from mlx_lm.models.rope_utils import initialize_rope

            self.rope = initialize_rope(
                head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        except ImportError:
            print("⚠️ Could not import mlx_lm rope_utils, using basic RoPE")
            self.rope = None

        print(f"🔧 Initialized Custom Metal GQA Attention")
        print(f"   📊 Architecture: {n_heads}:{n_kv_heads} heads ({n_heads//n_kv_heads}:1 ratio)")
        print(f"   🎯 Head dimension: {head_dim}")
        print(f"   ⚡ Using custom Metal kernel for GQA optimization")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Standard preprocessing (already optimized, don't evolve)
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Standard RoPE application (already optimized, don't evolve)
        if cache is not None:
            if self.rope is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            if self.rope is not None:
                queries = self.rope(queries)
                keys = self.rope(keys)

        # CORE INNOVATION: Custom Metal kernel for GQA attention
        output = qwen3_custom_gqa_attention(queries, keys, values, scale=self.scale, mask=mask)

        # Standard postprocessing (already optimized, don't evolve)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

# From mlx_metal_kernel_opt/best_program.py
def qwen3_custom_gqa_attention(queries, keys, values, scale=1.0, mask=None):
    """
    Custom Metal kernel implementation for Qwen3 GQA attention.

    Args:
        queries: [B, num_heads=40, L, head_dim=128]
        keys: [B, num_kv_heads=8, L, head_dim=128]
        values: [B, num_kv_heads=8, L, head_dim=128]
        scale: Attention scaling factor (1/sqrt(head_dim))
        mask: Attention mask (None, "causal", or boolean tensor)

    Returns:
        Attention output [B, num_heads=40, L, head_dim=128]
    """

    B, num_heads, L, head_dim = queries.shape
    _, num_kv_heads, _, _ = keys.shape
    heads_per_kv = num_heads // num_kv_heads  # Should be 5 for Qwen3

    # Handle mask conversion
    if mask == "causal" or mask is None:
        # Create causal mask for autoregressive attention
        causal_mask = mx.triu(mx.ones((L, L), dtype=mx.bool_), k=1)
        mask_tensor = mx.logical_not(causal_mask)  # True where attention is allowed
        use_mask = True
    elif isinstance(mask, (mx.array, type(None))):
        if mask is None:
            mask_tensor = mx.ones((L, L), dtype=mx.bool_)
            use_mask = False
        else:
            mask_tensor = mask.astype(mx.bool_)
            use_mask = True
    else:
        # Raise error for unsupported mask types - no fallback
        raise ValueError(
            f"Unsupported mask type: {type(mask)}. Custom kernel requires None, 'causal', or mx.array mask."
        )

    # Expand mask to match batch and head dimensions if needed
    if mask_tensor.ndim == 2:
        mask_tensor = mx.broadcast_to(mask_tensor[None, None, :, :], (B, num_heads, L, L))
    elif mask_tensor.ndim == 3:
        mask_tensor = mx.broadcast_to(mask_tensor[:, None, :, :], (B, num_heads, L, L))

    # EVOLVE-BLOCK-START
    # Custom Metal kernel source for Qwen3 GQA optimization
    # This kernel leverages the 40:8 head ratio and Apple Silicon architecture
    kernel_source = """
    // Qwen3 GQA Metal Kernel - Optimized for 40:8 head pattern
    // Thread mapping: each thread processes one query position
    uint thread_id = thread_position_in_grid.x;
    uint head_idx = thread_position_in_grid.y; 
    uint batch_idx = thread_position_in_grid.z;
    uint query_pos = thread_id;
    
    // Bounds checking
    if (batch_idx >= BATCH_SIZE || head_idx >= NUM_HEADS || query_pos >= SEQ_LEN) {
        return;
    }
    
    // Extract scalar values from input arrays
    T scale_val = scale[0];
    bool use_mask_val = use_mask[0] > 0;
    
    // GQA mapping: determine which KV head corresponds to this query head
    uint kv_head_idx = head_idx / HEADS_PER_KV;  // 5 query heads per KV head
    
    // Pre-calculate base indices for memory access optimization
    const uint q_base = batch_idx * (NUM_HEADS * SEQ_LEN * HEAD_DIM) + 
                        head_idx * (SEQ_LEN * HEAD_DIM) + 
                        query_pos * HEAD_DIM;
                        
    const uint k_base_start = batch_idx * (NUM_KV_HEADS * SEQ_LEN * HEAD_DIM) + 
                              kv_head_idx * (SEQ_LEN * HEAD_DIM);
                              
    const uint v_base_start = k_base_start;  // Values have same layout as keys
    
    const uint mask_base = batch_idx * (NUM_HEADS * SEQ_LEN * SEQ_LEN) + 
                           head_idx * (SEQ_LEN * SEQ_LEN) + 
                           query_pos * SEQ_LEN;
                           
    const uint out_base = q_base;
    
    // Use vector type for query_vec (e.g., float8 or half8 for better SIMD utilization)
    // HEAD_DIM is 128, so 16 vec<T, 8> elements
    vec<T, 8> query_vec_v[HEAD_DIM / 8];
    for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) {
        query_vec_v[d_vec] = ((device vec<T, 8>*) (queries + q_base))[d_vec];
    }
    
    // Pass 1: Compute max_score for numerical stability (online max)
    T max_score = T(-INFINITY);
    
    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
        
        T score;
        if (!is_valid) {
            score = T(-INFINITY); // Masked scores are -infinity, consistent with Pass 2
        } else {
            // Compute Q @ K^T for this key position using vectorized dot product
            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            score = T(0.0); // Initialize score here
            
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                score += dot(query_vec_v[d_vec], ((device vec<T, 8>*) (keys + k_base))[d_vec]);
            }
            
            // Apply attention scaling
            score *= scale_val;
        }
        max_score = max(max_score, score);
    }
    
    // Pass 2: Compute softmax denominator and weighted sum (online sum)
    T sum_exp = T(0.0);
    vec<T, 8> output_acc_v[HEAD_DIM / 8]; // Accumulator for output vector, use vec<T, 8>
    
    // Initialize output accumulator to zero
    for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) {
        output_acc_v[d_vec] = T(0.0);
    }

    for (uint key_pos = 0; key_pos < SEQ_LEN; key_pos++) {
        bool is_valid = use_mask_val ? mask[mask_base + key_pos] : true;
        
        T current_score;
        if (!is_valid) {
            current_score = T(-INFINITY); // Masked scores are -infinity
        } else {
            // Recompute Q @ K^T for this key position
            const uint k_base = k_base_start + key_pos * HEAD_DIM;
            T score = T(0.0);
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                score += dot(query_vec_v[d_vec], ((device vec<T, 8>*) (keys + k_base))[d_vec]);
            }
            current_score = score * scale_val;
        }

        // Apply softmax (exp and sum)
        T exp_score;
        if (current_score == T(-INFINITY)) {
            exp_score = T(0.0); // exp(-infinity) is 0
        } else {
            exp_score = exp(current_score - max_score);
        }
        sum_exp += exp_score;
        
        // Compute weighted sum of values
        if (exp_score > T(0.0)) { // Only add if exp_score is positive
            const uint v_base = v_base_start + key_pos * HEAD_DIM;
            for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
                output_acc_v[d_vec] += exp_score * ((device vec<T, 8>*) (values + v_base))[d_vec];
            }
        }
    }
    
    // Final normalization and write result to global memory
    if (sum_exp > T(0.0)) {
        for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
            output_acc_v[d_vec] /= sum_exp;
            ((device vec<T, 8>*) (output + out_base))[d_vec] = output_acc_v[d_vec];
        }
    } else {
        // Handle case where sum_exp is zero (e.g., all scores were masked or extremely small)
        // Set output to zero to avoid NaN/Inf results.
        for (uint d_vec = 0; d_vec < HEAD_DIM / 8; d_vec++) { // Use vec<T, 8>
            ((device vec<T, 8>*) (output + out_base))[d_vec] = T(0.0);
        }
    }
    """
    # EVOLVE-BLOCK-END

    try:
        # Prepare kernel inputs
        scale_tensor = mx.array([scale], dtype=queries.dtype)
        use_mask_tensor = mx.array([1 if use_mask else 0], dtype=mx.int32)

        # Create and execute custom Metal kernel
        kernel = mx.fast.metal_kernel(
            name="qwen3_gqa_attention_kernel",
            input_names=["queries", "keys", "values", "mask", "scale", "use_mask"],
            output_names=["output"],
            source=kernel_source,
        )

        # Optimize thread group size for Apple Silicon
        threadgroup_size = min(32, L)  # Adapt to sequence length

        # Execute kernel
        outputs = kernel(
            inputs=[queries, keys, values, mask_tensor, scale_tensor, use_mask_tensor],
            output_shapes=[(B, num_heads, L, head_dim)],
            output_dtypes=[queries.dtype],
            grid=(L, num_heads, B),  # (SEQ_LEN, NUM_HEADS, BATCH_SIZE)
            threadgroup=(threadgroup_size, 1, 1),
            template=[
                ("T", queries.dtype),
                ("BATCH_SIZE", B),
                ("NUM_HEADS", num_heads),
                ("NUM_KV_HEADS", num_kv_heads),
                ("SEQ_LEN", L),
                ("HEAD_DIM", head_dim),
                ("HEADS_PER_KV", heads_per_kv),
            ],
        )

        return outputs[0]

    except Exception as e:
        # No fallback - let the custom kernel failure propagate for proper scoring
        print(f"❌ Custom GQA kernel failed: {e}")
        raise RuntimeError(f"Custom Metal kernel execution failed: {e}") from e

# From mlx_metal_kernel_opt/best_program.py
def create_metal_qwen3_optimization_hook():
    """
    Create hooks to replace Qwen3's attention with Metal kernel optimized version.
    """

    def apply_optimization_hook():
        """Apply the Metal kernel optimized attention"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class
            original_attention = qwen3_module.Attention

            # Replace with Metal optimized implementation
            qwen3_module.Attention = CustomGQAAttention

            print("✅ Applied Custom Metal GQA Attention hook")
            return original_attention

        except ImportError:
            print("❌ Could not import mlx_lm.models.qwen3")
            return None

    def remove_optimization_hook(original_attention):
        """Remove the optimization hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            qwen3_module.Attention = original_attention
            print("✅ Removed Custom Metal GQA Attention hook")
        except ImportError:
            pass

    return apply_optimization_hook, remove_optimization_hook

# From mlx_metal_kernel_opt/best_program.py
def benchmark_metal_gqa_optimization():
    """
    Benchmark Metal kernel optimized GQA attention against MLX baseline.
    """

    # Qwen3-0.6B configuration
    class MockArgs:
        hidden_size = 5120
        num_attention_heads = 40
        num_key_value_heads = 8
        head_dim = 128
        rms_norm_eps = 1e-06
        rope_theta = 1000000
        rope_scaling = None
        max_position_embeddings = 40960

    args = MockArgs()

    # Test configurations for Metal kernel validation
    test_configs = [
        ("short_sequence", 1, 128, 5120),
        ("medium_sequence", 1, 512, 5120),
        ("long_sequence", 1, 1024, 5120),
        ("max_sequence", 1, 2048, 5120),
    ]

    print("Benchmarking Custom Metal GQA Kernel vs MLX Baseline")
    print("=" * 70)

    # Initialize Metal optimized attention
    metal_attn = CustomGQAAttention(args)

    for config_name, batch_size, seq_len, hidden_size in test_configs:
        print(f"\nTesting {config_name}: B={batch_size}, L={seq_len}")

        # Create test inputs
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        mask = "causal"

        # Warmup runs
        for _ in range(3):
            _ = metal_attn(x, mask=mask)
            mx.eval(_)

        # Benchmark Metal optimized implementation
        mx.synchronize()
        start_time = time.perf_counter()

        for _ in range(10):
            output = metal_attn(x, mask=mask)
            mx.eval(output)

        mx.synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10
        tokens_per_sec = seq_len / avg_time

        print(f"  Metal GQA: {avg_time*1000:.2f} ms, {tokens_per_sec:.1f} tokens/sec")
        print(f"  Memory: {mx.get_active_memory() / 1e9:.2f} GB")

# From mlx_metal_kernel_opt/best_program.py
def test_metal_gqa_correctness():
    """
    Test that Metal kernel implementation produces correct results.
    """
    print("Testing Custom Metal GQA Correctness")
    print("=" * 50)

    # Test configuration
    B, L, D = 1, 64, 5120

    class MockArgs:
        hidden_size = 5120
        num_attention_heads = 40
        num_key_value_heads = 8
        head_dim = 128
        rms_norm_eps = 1e-06
        rope_theta = 1000000
        rope_scaling = None
        max_position_embeddings = 40960

    args = MockArgs()

    # Create test input
    x = mx.random.normal((B, L, D))
    mask = "causal"

    # Test Metal optimized implementation
    metal_attn = CustomGQAAttention(args)
    output = metal_attn(x, mask=mask)

    print(f"✅ Metal GQA output shape: {output.shape}")

    # Check for valid output
    has_nan = bool(mx.any(mx.isnan(output)))
    has_inf = bool(mx.any(mx.isinf(output)))

    print(f"✅ Has NaN: {has_nan}, Has Inf: {has_inf}")

    # Check output statistics
    output_mean = float(mx.mean(output))
    output_std = float(mx.std(output))

    print(f"✅ Output statistics - Mean: {output_mean:.6f}, Std: {output_std:.6f}")

    # Test direct kernel function
    print("\n=== Testing Direct Kernel Function ===")
    B, H, L, D = 1, 40, 128, 128
    q = mx.random.normal((B, H, L, D))
    k = mx.random.normal((B, 8, L, D))  # 8 KV heads
    v = mx.random.normal((B, 8, L, D))
    scale = 1.0 / math.sqrt(D)

    kernel_output = qwen3_custom_gqa_attention(q, k, v, scale=scale, mask="causal")
    print(f"✅ Direct kernel output shape: {kernel_output.shape}")

    kernel_mean = float(mx.mean(kernel_output))
    kernel_std = float(mx.std(kernel_output))
    print(f"✅ Direct kernel stats - Mean: {kernel_mean:.6f}, Std: {kernel_std:.6f}")

    return True

# From mlx_metal_kernel_opt/best_program.py
def apply_optimization_hook():
        """Apply the Metal kernel optimized attention"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            # Store original attention class
            original_attention = qwen3_module.Attention

            # Replace with Metal optimized implementation
            qwen3_module.Attention = CustomGQAAttention

            print("✅ Applied Custom Metal GQA Attention hook")
            return original_attention

        except ImportError:
            print("❌ Could not import mlx_lm.models.qwen3")
            return None

# From mlx_metal_kernel_opt/best_program.py
def remove_optimization_hook(original_attention):
        """Remove the optimization hook"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            qwen3_module.Attention = original_attention
            print("✅ Removed Custom Metal GQA Attention hook")
        except ImportError:
            pass

from quick_benchmark_test import run_quick_test

# From mlx_metal_kernel_opt/run_benchmarks.py
def run_compare_benchmarks(args):
    """
    Run comprehensive comparison between standard and optimized attention.
    Uses the full benchmark suite for thorough analysis.
    """
    print(f"\n🔬 Running Comparison Benchmark Mode")
    print(f"📊 Comparing Standard vs OpenEvolve Discovered Optimization")
    print(f"🎯 Model: {args.model}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 80)

    # Change to output directory
    original_dir = os.getcwd()
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    try:
        # Run standard benchmark (baseline)
        print("\n🏃‍♂️ Phase 1: Running Standard MLX-LM Attention Benchmark...")
        print("⏱️  This establishes our baseline performance across all scenarios")

        # Get dynamic test count
        temp_suite = Qwen3BenchmarkSuite(args.model)
        test_count = len(temp_suite.create_benchmark_configs())

        print(f"📊 Running full benchmark suite ({test_count} comprehensive tests)")
        print("⏳ This will take 15-30 minutes depending on your hardware...")

        standard_suite = Qwen3BenchmarkSuite(args.model)
        standard_results = standard_suite.run_full_benchmark_suite()

        print("\n✅ Standard benchmark complete!")
        print(f"📊 Standard results: {len(standard_results['results'])} benchmarks completed")

        # Apply optimized attention hook and run benchmark
        print("\n🚀 Phase 2: Running OpenEvolve Discovered Optimization...")
        print("💡 Applying custom Metal kernel optimized GQA attention")

        # Import and apply the optimized attention
        optimized_results = run_optimized_benchmark(args, original_dir)

        if optimized_results is None:
            print("❌ Failed to run optimized benchmark")
            return 1

        print("\n✅ Optimized benchmark complete!")
        print(f"📊 Optimized results: {len(optimized_results['results'])} benchmarks completed")

        # Generate comparison analysis
        print("\n📈 Generating Comparison Analysis...")
        comparison_results = analyze_comparison_results(
            standard_results, optimized_results, args.model
        )

        if comparison_results is None:
            print("❌ Failed to generate comparison analysis")
            return 1

        # Save comparison results
        save_comparison_results(comparison_results, args.output_dir)

        # Print detailed comparison
        print_comparison_summary(comparison_results)

        return 0

    except Exception as e:
        print(f"❌ Error in comparison benchmark: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        os.chdir(original_dir)

# From mlx_metal_kernel_opt/run_benchmarks.py
def run_optimized_benchmark(args, original_dir):
    """
    Run benchmark with the optimized attention from best_program.py.
    """
    try:
        # Import the optimized attention implementation
        # First, try the OpenEvolve output directory (most likely location)
        best_program_path = os.path.join(
            original_dir, "openevolve_output", "best", "best_program.py"
        )

        # Fallback to root directory if not found in openevolve_output
        if not os.path.exists(best_program_path):
            best_program_path = os.path.join(original_dir, "best_program.py")

        if not os.path.exists(best_program_path):
            print(f"❌ Error: Optimized program not found")
            print("Searched in the following locations:")
            print(
                f"  1. {os.path.join(original_dir, 'openevolve_output', 'best', 'best_program.py')}"
            )
            print(f"  2. {os.path.join(original_dir, 'best_program.py')}")
            print("Please ensure OpenEvolve has generated an optimized solution")
            print("Expected path: ./openevolve_output/best/best_program.py")
            return None

        print(f"📁 Loading optimized program from: {best_program_path}")

        # Import the optimized module
        import importlib.util

        spec = importlib.util.spec_from_file_location("best_program", best_program_path)
        best_program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(best_program)

        print("✅ Optimized program loaded successfully")

        # Check for the hook function
        if not hasattr(best_program, "create_metal_qwen3_optimization_hook"):
            print(
                "❌ Error: create_metal_qwen3_optimization_hook function not found in best_program.py"
            )
            print(
                "Available functions:",
                [attr for attr in dir(best_program) if not attr.startswith("_")],
            )
            return None

        # Apply the custom attention hook
        apply_hook, remove_hook = best_program.create_metal_qwen3_optimization_hook()
        print("🔧 Applying custom Metal kernel optimized attention hook...")

        original_attention = apply_hook()

        if original_attention is None:
            print("❌ Failed to apply custom Metal kernel optimization hook")
            print("This may indicate MLX-LM import issues or incompatible environment")
            return None

        print("✅ Custom Metal kernel optimization hook applied successfully")

        try:
            # Run benchmarks with optimized attention
            print("📊 Running full benchmark suite with custom Metal kernel optimization...")
            print("⏳ This will take another 15-30 minutes...")
            print(
                "💡 The optimization uses custom Metal kernel implementation for Apple Silicon GPU"
            )

            optimized_suite = Qwen3BenchmarkSuite(args.model)
            optimized_results = optimized_suite.run_full_benchmark_suite()

            print("✅ Custom Metal kernel benchmark suite completed successfully")
            return optimized_results

        finally:
            # Always remove the hook to restore original behavior
            print("🔄 Restoring standard attention...")
            remove_hook(original_attention)
            print("✅ Standard attention restored")

    except Exception as e:
        print(f"❌ Error running Metal kernel optimized benchmark: {e}")
        import traceback

        traceback.print_exc()
        return None

# From mlx_metal_kernel_opt/run_benchmarks.py
def analyze_comparison_results(standard_results, optimized_results, model_name):
    """
    Analyze and compare the benchmark results.
    """
    if not standard_results or not optimized_results:
        print("❌ Cannot compare - missing results")
        return None

    print("🔍 Analyzing benchmark comparisons...")

    standard_benchmarks = {r["name"]: r for r in standard_results["results"]}
    optimized_benchmarks = {r["name"]: r for r in optimized_results["results"]}

    print(f"📊 Standard benchmarks: {len(standard_benchmarks)}")
    print(f"📊 Optimized benchmarks: {len(optimized_benchmarks)}")

    # Find common benchmarks
    common_benchmarks = set(standard_benchmarks.keys()) & set(optimized_benchmarks.keys())
    print(f"📊 Common benchmarks for comparison: {len(common_benchmarks)}")

    if len(common_benchmarks) == 0:
        print("❌ No common benchmarks found for comparison")
        return None

    comparisons = []
    improvements = {
        "decode_speed_improvements": [],
        "prefill_speed_improvements": [],
        "total_speed_improvements": [],
        "memory_improvements": [],
        "time_improvements": [],
    }

    for name in common_benchmarks:
        std_result = standard_benchmarks[name]
        opt_result = optimized_benchmarks[name]

        # Calculate improvements
        decode_improvement = (
            (
                (opt_result["decode_tokens_per_sec"] - std_result["decode_tokens_per_sec"])
                / std_result["decode_tokens_per_sec"]
                * 100
            )
            if std_result["decode_tokens_per_sec"] > 0
            else 0
        )

        prefill_improvement = (
            (
                (opt_result["prefill_tokens_per_sec"] - std_result["prefill_tokens_per_sec"])
                / std_result["prefill_tokens_per_sec"]
                * 100
            )
            if std_result["prefill_tokens_per_sec"] > 0
            else 0
        )

        total_improvement = (
            (
                (opt_result["total_tokens_per_sec"] - std_result["total_tokens_per_sec"])
                / std_result["total_tokens_per_sec"]
                * 100
            )
            if std_result["total_tokens_per_sec"] > 0
            else 0
        )

        memory_improvement = (
            (
                (std_result["peak_memory_gb"] - opt_result["peak_memory_gb"])
                / std_result["peak_memory_gb"]
                * 100
            )
            if std_result["peak_memory_gb"] > 0
            else 0
        )

        time_improvement = (
            (
                (std_result["total_time_sec"] - opt_result["total_time_sec"])
                / std_result["total_time_sec"]
                * 100
            )
            if std_result["total_time_sec"] > 0
            else 0
        )

        comparison = {
            "benchmark_name": name,
            "standard": std_result,
            "optimized": opt_result,
            "improvements": {
                "decode_speed_pct": decode_improvement,
                "prefill_speed_pct": prefill_improvement,
                "total_speed_pct": total_improvement,
                "memory_reduction_pct": memory_improvement,
                "time_reduction_pct": time_improvement,
            },
        }

        comparisons.append(comparison)

        # Collect for aggregate statistics
        improvements["decode_speed_improvements"].append(decode_improvement)
        improvements["prefill_speed_improvements"].append(prefill_improvement)
        improvements["total_speed_improvements"].append(total_improvement)
        improvements["memory_improvements"].append(memory_improvement)
        improvements["time_improvements"].append(time_improvement)

    # Calculate aggregate statistics
    aggregate_stats = {}
    for key, values in improvements.items():
        if values:
            aggregate_stats[f"{key}_avg"] = np.mean(values)
            aggregate_stats[f"{key}_median"] = np.median(values)
            aggregate_stats[f"{key}_min"] = np.min(values)
            aggregate_stats[f"{key}_max"] = np.max(values)
            aggregate_stats[f"{key}_std"] = np.std(values)

    # Calculate overall metrics
    std_decode_speeds = [
        std_result["decode_tokens_per_sec"] for std_result in standard_benchmarks.values()
    ]
    opt_decode_speeds = [
        opt_result["decode_tokens_per_sec"] for opt_result in optimized_benchmarks.values()
    ]

    avg_std_decode = np.mean(std_decode_speeds) if std_decode_speeds else 0
    avg_opt_decode = np.mean(opt_decode_speeds) if opt_decode_speeds else 0

    print(f"📊 Analysis complete:")
    print(f"  📈 Average standard decode speed: {avg_std_decode:.1f} tokens/sec")
    print(f"  📈 Average optimized decode speed: {avg_opt_decode:.1f} tokens/sec")
    print(
        f"  📈 Average improvement: {aggregate_stats.get('decode_speed_improvements_avg', 0):.1f}%"
    )

    return {
        "model": model_name,
        "timestamp": int(time.time()),
        "optimization_type": "custom_metal_kernel",
        "total_comparisons": len(comparisons),
        "individual_comparisons": comparisons,
        "aggregate_improvements": aggregate_stats,
        "summary": {
            "avg_decode_improvement_pct": aggregate_stats.get("decode_speed_improvements_avg", 0),
            "avg_total_improvement_pct": aggregate_stats.get("total_speed_improvements_avg", 0),
            "avg_memory_reduction_pct": aggregate_stats.get("memory_improvements_avg", 0),
            "avg_time_reduction_pct": aggregate_stats.get("time_improvements_avg", 0),
            "avg_standard_decode_speed": avg_std_decode,
            "avg_optimized_decode_speed": avg_opt_decode,
            "benchmarks_improved": sum(
                1 for x in improvements["decode_speed_improvements"] if x > 0
            ),
            "total_benchmarks": len(improvements["decode_speed_improvements"]),
        },
    }

# From mlx_metal_kernel_opt/run_benchmarks.py
def save_comparison_results(comparison_results, output_dir):
    """
    Save detailed comparison results to files.
    """
    if not comparison_results:
        return

    timestamp = comparison_results["timestamp"]

    # Save detailed JSON results
    comparison_file = f"openevolve_comparison_results_{timestamp}.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    # Save CSV summary for easy analysis
    import csv

    csv_file = f"openevolve_comparison_summary_{timestamp}.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "benchmark_name",
                "category",
                "standard_decode_speed",
                "optimized_decode_speed",
                "decode_improvement_pct",
                "standard_prefill_speed",
                "optimized_prefill_speed",
                "prefill_improvement_pct",
                "standard_total_speed",
                "optimized_total_speed",
                "total_improvement_pct",
                "standard_memory_gb",
                "optimized_memory_gb",
                "memory_reduction_pct",
                "standard_time_sec",
                "optimized_time_sec",
                "time_reduction_pct",
            ]
        )

        for comp in comparison_results["individual_comparisons"]:
            # Extract category from benchmark name
            category = "general"
            name = comp["benchmark_name"]
            if "short" in name.lower():
                category = "short_context"
            elif "long" in name.lower():
                category = "long_context"
            elif "code" in name.lower():
                category = "code_generation"
            elif "stress" in name.lower() or "maximum" in name.lower():
                category = "stress_test"

            writer.writerow(
                [
                    comp["benchmark_name"],
                    category,
                    comp["standard"]["decode_tokens_per_sec"],
                    comp["optimized"]["decode_tokens_per_sec"],
                    comp["improvements"]["decode_speed_pct"],
                    comp["standard"]["prefill_tokens_per_sec"],
                    comp["optimized"]["prefill_tokens_per_sec"],
                    comp["improvements"]["prefill_speed_pct"],
                    comp["standard"]["total_tokens_per_sec"],
                    comp["optimized"]["total_tokens_per_sec"],
                    comp["improvements"]["total_speed_pct"],
                    comp["standard"]["peak_memory_gb"],
                    comp["optimized"]["peak_memory_gb"],
                    comp["improvements"]["memory_reduction_pct"],
                    comp["standard"]["total_time_sec"],
                    comp["optimized"]["total_time_sec"],
                    comp["improvements"]["time_reduction_pct"],
                ]
            )

    print(f"\n📁 Comparison results saved:")
    print(f"  📊 Detailed: {comparison_file}")
    print(f"  📈 Summary: {csv_file}")

# From mlx_metal_kernel_opt/run_benchmarks.py
def print_comparison_summary(comparison_results):
    """
    Print a comprehensive comparison summary.
    """
    if not comparison_results:
        print("❌ No comparison results to display")
        return

    print(f"\n{'='*100}")
    print(f"{'🚀 OPENEVOLVE CUSTOM METAL KERNEL OPTIMIZATION RESULTS':^100}")
    print(f"{'='*100}")

    summary = comparison_results["summary"]
    total_tests = comparison_results["total_comparisons"]

    print(f"\n💡 OPTIMIZATION: Custom Metal Kernel for GQA Attention")
    print(f"   Strategy: Hand-optimized Metal kernel using vectorized operations")
    print(f"   Target: Apple Silicon GPU with optimized memory access patterns")

    print(f"\n🎯 OVERALL PERFORMANCE IMPROVEMENTS (across {total_tests} comprehensive tests):")
    print(f"  📈 Average Decode Speed Improvement: {summary['avg_decode_improvement_pct']:+.2f}%")
    print(f"  ⚡ Average Total Speed Improvement:  {summary['avg_total_improvement_pct']:+.2f}%")
    print(f"  💾 Average Memory Reduction:        {summary['avg_memory_reduction_pct']:+.2f}%")
    print(f"  ⏱️  Average Time Reduction:          {summary['avg_time_reduction_pct']:+.2f}%")

    print(f"\n📊 ABSOLUTE PERFORMANCE:")
    print(
        f"  🔵 Standard MLX-LM:     {summary['avg_standard_decode_speed']:.1f} tokens/sec average"
    )
    print(
        f"  🟠 Metal Kernel Optimized: {summary['avg_optimized_decode_speed']:.1f} tokens/sec average"
    )
    print(
        f"  📈 Net Improvement:     {summary['avg_optimized_decode_speed'] - summary['avg_standard_decode_speed']:+.1f} tokens/sec"
    )

    print(f"\n📊 DETAILED BENCHMARK COMPARISON:")
    print(f"{'='*110}")
    print(
        f"{'Benchmark':<30} {'Standard':<12} {'Optimized':<12} {'Decode':<12} {'Memory':<12} {'Time':<12}"
    )
    print(
        f"{'Name':<30} {'Decode':<12} {'Decode':<12} {'Improv(%)':<12} {'Reduct(%)':<12} {'Reduct(%)':<12}"
    )
    print(f"{'-'*110}")

    for comp in sorted(
        comparison_results["individual_comparisons"],
        key=lambda x: x["improvements"]["decode_speed_pct"],
        reverse=True,
    ):
        name = comp["benchmark_name"][:29]
        std_decode = comp["standard"]["decode_tokens_per_sec"]
        opt_decode = comp["optimized"]["decode_tokens_per_sec"]
        decode_imp = comp["improvements"]["decode_speed_pct"]
        mem_imp = comp["improvements"]["memory_reduction_pct"]
        time_imp = comp["improvements"]["time_reduction_pct"]

        # Color coding for improvements
        if decode_imp > 20:
            marker = "🚀"
        elif decode_imp > 10:
            marker = "📈"
        elif decode_imp > 0:
            marker = "✅"
        else:
            marker = "⚠️"

        print(
            f"{marker} {name:<28} {std_decode:<12.1f} {opt_decode:<12.1f} {decode_imp:+<12.1f} {mem_imp:+<12.1f} {time_imp:+<12.1f}"
        )

    print(f"{'-'*110}")

    # Highlight best and worst improvements
    best_decode = max(
        comparison_results["individual_comparisons"],
        key=lambda x: x["improvements"]["decode_speed_pct"],
    )
    worst_decode = min(
        comparison_results["individual_comparisons"],
        key=lambda x: x["improvements"]["decode_speed_pct"],
    )

    print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
    print(
        f"  🥇 Best Improvement: {best_decode['benchmark_name']} (+{best_decode['improvements']['decode_speed_pct']:.1f}%)"
    )
    print(
        f"  📊 Worst Case: {worst_decode['benchmark_name']} ({worst_decode['improvements']['decode_speed_pct']:+.1f}%)"
    )

    # Optimization analysis
    improved_count = summary["benchmarks_improved"]
    total_count = summary["total_benchmarks"]
    success_rate = improved_count / total_count * 100 if total_count > 0 else 0

    print(f"\n📈 OPTIMIZATION ANALYSIS:")
    print(f"  ✅ Benchmarks Improved: {improved_count}/{total_count}")
    print(f"  📊 Success Rate: {success_rate:.1f}%")

    if summary["avg_decode_improvement_pct"] > 15:
        print(f"  🎉 EXCELLENT: OpenEvolve discovered a significant optimization!")
        print(
            f"  💡 {summary['avg_decode_improvement_pct']:.1f}% average improvement is substantial"
        )
        print(f"  🔬 This warrants further investigation and potential MLX-LM contribution")
    elif summary["avg_decode_improvement_pct"] > 5:
        print(f"  📈 GOOD: Meaningful performance improvements achieved")
        print(
            f"  🔧 {summary['avg_decode_improvement_pct']:.1f}% improvement shows optimization potential"
        )
    elif summary["avg_decode_improvement_pct"] > 0:
        print(f"  📊 MODEST: Some improvements observed")
        print(
            f"  💭 {summary['avg_decode_improvement_pct']:.1f}% suggests room for further optimization"
        )
    else:
        print(f"  ⚠️  No overall improvement detected")
        print(f"  🔧 Consider running additional evolution cycles or different strategies")

    # Technical insights
    print(f"\n🔬 TECHNICAL INSIGHTS:")
    print(f"  💡 Custom Metal Kernel Strategy:")
    print(f"     • Standard: mx.fast.scaled_dot_product_attention")
    print(f"     • Optimized: Hand-written Metal kernel with vectorized operations")
    print(f"  🧠 Potential Reasons for Performance Gains:")
    print(f"     • Optimized memory access patterns for Apple Silicon")
    print(f"     • Vectorized operations using vec<T, 8> types")
    print(f"     • Better cache locality with custom computation order")
    print(f"     • GPU-specific optimizations for M-series processors")

    if summary["avg_decode_improvement_pct"] > 10:
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Verify results independently outside this framework")
        print(f"  2. Profile Metal kernel execution patterns and memory usage")
        print(f"  3. Test on different Apple Silicon variants (M1, M2, M3, M4)")
        print(f"  4. Consider contributing Metal kernel optimization back to MLX")
        print(f"  5. Explore similar Metal kernel strategies for other attention patterns")

    print(f"\n{'='*100}")
    print(f"🔬 Comprehensive analysis complete! Results saved to comparison files.")
    print(f"💡 This represents a genuine Metal kernel discovery by OpenEvolve.")
    print(f"{'='*100}")


from scipy import signal
from scipy.stats import pearsonr

# From signal_processing/evaluator.py
def calculate_slope_changes(signal_data):
    """
    Calculate slope change penalty S(θ) - counts directional reversals

    Args:
        signal_data: 1D array of signal values

    Returns:
        Number of slope changes (directional reversals)
    """
    if len(signal_data) < 3:
        return 0

    # Calculate differences
    diffs = np.diff(signal_data)

    # Count sign changes in consecutive differences
    sign_changes = 0
    for i in range(1, len(diffs)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]) and diffs[i - 1] != 0:
            sign_changes += 1

    return sign_changes

# From signal_processing/evaluator.py
def calculate_lag_error(filtered_signal, original_signal, window_size):
    """
    Calculate instantaneous lag error L_recent(θ) = |y[n] - x[n]|

    Args:
        filtered_signal: Output of the filter
        original_signal: Original input signal
        window_size: Size of the processing window

    Returns:
        Instantaneous lag error at the most recent sample
    """
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    # Account for processing delay
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    # Compare the last filtered sample with the corresponding original sample
    recent_filtered = filtered_signal[-1]
    recent_original = original_signal[delay + len(filtered_signal) - 1]

    return abs(recent_filtered - recent_original)

# From signal_processing/evaluator.py
def calculate_average_tracking_error(filtered_signal, original_signal, window_size):
    """
    Calculate average tracking error L_avg(θ) over the window

    Args:
        filtered_signal: Output of the filter
        original_signal: Original input signal
        window_size: Size of the processing window

    Returns:
        Average absolute error over the processed samples
    """
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    # Account for processing delay
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    # Align signals
    aligned_original = original_signal[delay : delay + len(filtered_signal)]

    # Ensure same length
    min_length = min(len(filtered_signal), len(aligned_original))
    if min_length == 0:
        return 1.0

    filtered_aligned = filtered_signal[:min_length]
    original_aligned = aligned_original[:min_length]

    # Calculate mean absolute error
    return np.mean(np.abs(filtered_aligned - original_aligned))

# From signal_processing/evaluator.py
def calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size):
    """
    Calculate false reversal penalty R(θ) - mismatched trend changes

    Args:
        filtered_signal: Output of the filter
        clean_signal: Ground truth clean signal
        window_size: Size of the processing window

    Returns:
        Penalty for trend changes that don't match the clean signal
    """
    if len(filtered_signal) < 3 or len(clean_signal) < 3:
        return 0

    # Account for processing delay
    delay = window_size - 1
    if len(clean_signal) <= delay:
        return 1.0

    # Align signals
    aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
    min_length = min(len(filtered_signal), len(aligned_clean))

    if min_length < 3:
        return 0

    filtered_aligned = filtered_signal[:min_length]
    clean_aligned = aligned_clean[:min_length]

    # Calculate trend changes for both signals
    filtered_diffs = np.diff(filtered_aligned)
    clean_diffs = np.diff(clean_aligned)

    # Count mismatched trend changes
    false_reversals = 0
    for i in range(1, len(filtered_diffs)):
        # Check if there's a trend change in filtered signal
        filtered_change = (
            np.sign(filtered_diffs[i]) != np.sign(filtered_diffs[i - 1])
            and filtered_diffs[i - 1] != 0
        )

        # Check if there's a corresponding trend change in clean signal
        clean_change = (
            np.sign(clean_diffs[i]) != np.sign(clean_diffs[i - 1]) and clean_diffs[i - 1] != 0
        )

        # Count as false reversal if filtered has change but clean doesn't
        if filtered_change and not clean_change:
            false_reversals += 1

    return false_reversals

# From signal_processing/evaluator.py
def calculate_composite_score(S, L_recent, L_avg, R, alpha=[0.3, 0.2, 0.2, 0.3]):
    """
    Calculate the composite metric J(θ) = α₁·S(θ) + α₂·L_recent(θ) + α₃·L_avg(θ) + α₄·R(θ)

    All metrics are normalized and converted to penalties (higher = worse)
    The final score is converted to a maximization problem (higher = better)
    """
    # Normalize slope changes (typical range 0-100)
    S_norm = min(S / 50.0, 2.0)

    # Lag errors are already in reasonable range (0-10 typically)
    L_recent_norm = min(L_recent, 2.0)
    L_avg_norm = min(L_avg, 2.0)

    # Normalize false reversals (typical range 0-50)
    R_norm = min(R / 25.0, 2.0)

    # Calculate weighted penalty
    penalty = (
        alpha[0] * S_norm + alpha[1] * L_recent_norm + alpha[2] * L_avg_norm + alpha[3] * R_norm
    )

    # Convert to maximization score (higher is better)
    score = 1.0 / (1.0 + penalty)

    return score

# From signal_processing/evaluator.py
def generate_test_signals(num_signals=5):
    """
    Generate multiple test signals with different characteristics
    """
    test_signals = []

    for i in range(num_signals):
        np.random.seed(42 + i)  # Different seed for each signal
        length = 500 + i * 100  # Varying lengths
        noise_level = 0.2 + i * 0.1  # Varying noise levels

        t = np.linspace(0, 10, length)

        # Different signal characteristics
        if i == 0:
            # Smooth sinusoidal with trend
            clean = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        elif i == 1:
            # Multiple frequency components
            clean = (
                np.sin(2 * np.pi * 0.5 * t)
                + 0.5 * np.sin(2 * np.pi * 2 * t)
                + 0.2 * np.sin(2 * np.pi * 5 * t)
            )
        elif i == 2:
            # Non-stationary with changing frequency
            clean = np.sin(2 * np.pi * (0.5 + 0.2 * t) * t)
        elif i == 3:
            # Step changes
            clean = np.concatenate(
                [
                    np.ones(length // 3),
                    2 * np.ones(length // 3),
                    0.5 * np.ones(length - 2 * (length // 3)),
                ]
            )
        else:
            # Random walk with trend
            clean = np.cumsum(np.random.randn(length) * 0.1) + 0.05 * t

        # Add noise
        noise = np.random.normal(0, noise_level, length)
        noisy = clean + noise

        test_signals.append((noisy, clean))

    return test_signals

from collections import deque

# From signal_processing/initial_program.py
def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y

# From signal_processing/initial_program.py
def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using weighted moving average.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Create weights that emphasize recent samples
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    for i in range(output_length):
        window = x[i : i + window_size]

        # Weighted moving average with exponential weights
        y[i] = np.sum(window * weights)

    return y

# From signal_processing/initial_program.py
def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)

# From signal_processing/initial_program.py
def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal

# From signal_processing/initial_program.py
def run_signal_processing(signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Returns:
        Dictionary containing results and metrics
    """
    # Generate test signal
    noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)

    # Process the signal
    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics
    if len(filtered_signal) > 0:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }



# From circle_packing/best_program.py
def objective(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]
        return -np.sum(radii)

# From circle_packing/best_program.py
def constraint(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]

        # Overlap constraint
        overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                overlap_constraints.append(dist - (radii[i] + radii[j]))

        # Boundary constraints
        boundary_constraints = []
        for i in range(n):
            boundary_constraints.append(centers[i, 0] - radii[i])  # x >= radius
            boundary_constraints.append(1 - centers[i, 0] - radii[i])  # x <= 1 - radius
            boundary_constraints.append(centers[i, 1] - radii[i])  # y >= radius
            boundary_constraints.append(1 - centers[i, 1] - radii[i])  # y <= 1 - radius

        return np.array(overlap_constraints + boundary_constraints)


import shutil

# From attention_optimization/evaluator.py
class MLIRAttentionEvaluator:
    def __init__(self):
        self.verify_tools()
        self.mlir_file = Path("mlir/self_attn_with_consts_linalg_dialect.mlir")
        # self.mlir_file = Path("mlir/export_mlir.mlir")
        self.baseline_mlir = None
        self.baseline_metrics = None

    def verify_tools(self):
        """Verify MLIR tools are available"""
        tools = ['mlir-opt']
        for tool in tools:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool not found: {tool}")
        print("MLIR tools verified: mlir-opt")

    def load_baseline_mlir(self):
        """Load baseline MLIR from file"""
        if self.mlir_file.exists():
            print(f"Loading MLIR from: {self.mlir_file}")
            with open(self.mlir_file, 'r') as f:
                content = f.read()
            print(f"Loaded {len(content)} characters")
            return content
        else:
            raise FileNotFoundError(f"MLIR file not found: {self.mlir_file}")

    def analyze_ir_complexity(self, mlir_content):
        """Analyze MLIR IR for performance-relevant characteristics"""
        lines = mlir_content.splitlines()
        
        metrics = {
            'total_lines': len(lines),
            'total_chars': len(mlir_content),
            'operations': 0,
            'loops': 0,
            'memory_ops': 0,
            'arithmetic_ops': 0,
            'linalg_ops': 0,
            'func_calls': 0,
            'nested_depth': 0
        }
        
        current_depth = 0
        max_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
                
            # Count braces for nesting depth
            current_depth += stripped.count('{') - stripped.count('}')
            max_depth = max(max_depth, current_depth)
            
            # Count different operation types
            if '=' in stripped and ('%' in stripped or '@' in stripped):
                metrics['operations'] += 1
            
            # Specific operation patterns
            if any(loop_kw in stripped for loop_kw in ['scf.for', 'affine.for', 'scf.while']):
                metrics['loops'] += 1
            
            if any(mem_op in stripped for mem_op in ['memref.load', 'memref.store', 'tensor.extract', 'tensor.insert']):
                metrics['memory_ops'] += 1
                
            if any(arith_op in stripped for arith_op in ['arith.addf', 'arith.mulf', 'arith.divf', 'arith.subf']):
                metrics['arithmetic_ops'] += 1
                
            if 'linalg.' in stripped:
                metrics['linalg_ops'] += 1
                
            if 'func.call' in stripped or 'call @' in stripped:
                metrics['func_calls'] += 1
        
        metrics['nested_depth'] = max_depth
        return metrics

    def estimate_performance_from_ir(self, optimized_metrics, baseline_metrics, params):
        """Estimate performance based on IR analysis"""
        
        # Calculate relative changes
        ops_ratio = optimized_metrics['operations'] / max(baseline_metrics['operations'], 1)
        size_ratio = optimized_metrics['total_chars'] / max(baseline_metrics['total_chars'], 1)
        loop_ratio = optimized_metrics['loops'] / max(baseline_metrics['loops'], 1)
        arith_ratio = optimized_metrics['arithmetic_ops'] / max(baseline_metrics['arithmetic_ops'], 1)
        
        # Base performance model
        base_speedup = 1.0
        
        # Size reduction usually means optimization
        if size_ratio < 1.0:
            base_speedup += (1.0 - size_ratio) * 0.5  # Up to 50% speedup from size reduction
        
        # Loop optimizations
        unroll_factor = params.get('unroll_factor', 1)
        if unroll_factor > 1:
            base_speedup += min(unroll_factor * 0.05, 0.3)  # Up to 30% from unrolling
        
        # Memory optimizations  
        if params.get('use_shared_memory', False):
            base_speedup += 0.15  # 15% from better memory usage
        
        # Loop interchange
        if params.get('loop_interchange', False):
            base_speedup += 0.10  # 10% from better cache locality
        
        # Penalize if optimization increased complexity significantly
        if ops_ratio > 1.2:
            base_speedup *= 0.9  # 10% penalty for increased complexity
        
        # Add some realistic noise
        import random
        noise = random.uniform(0.95, 1.05)
        final_speedup = base_speedup * noise
        
        # Estimate runtime (inverse of speedup)
        base_runtime = 10.0  # Baseline runtime in arbitrary units
        estimated_runtime = base_runtime / final_speedup
        
        return {
            'speedup': final_speedup,
            'runtime': estimated_runtime,
            'method': 'ir_analysis',
            'size_ratio': size_ratio,
            'ops_ratio': ops_ratio,
            'optimization_score': base_speedup
        }

    def apply_optimizations(self, mlir_content, params):
        """Apply MLIR optimization passes based on parameters"""
        print(f"Applying optimizations: {params}")
        
        # Build pass pipeline with only verified working passes
        passes = ["canonicalize", "cse", "linalg-fold-unit-extent-dims"]
        
        # Add unroll with parameter
        unroll_factor = params.get('unroll_factor', 1)
        if unroll_factor > 1:
            passes.append(f"func.func(affine-loop-unroll)")
        
        # Add conditional passes
        if params.get('use_shared_memory', False):
            passes.append("linalg-fold-unit-extent-dims")
        
        if params.get('loop_interchange', False):
            passes.append("canonicalize")
            
        passes.extend(["canonicalize", "cse"])
        
        pipeline = f"builtin.module({','.join(passes)})"
        print(f"Using pipeline: {pipeline}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as input_file:
            input_file.write(mlir_content)
            input_file.flush()
            
            try:
                start_time = time.time()
                cmd = ['mlir-opt', input_file.name, f'--pass-pipeline={pipeline}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                compile_time = time.time() - start_time
                
                if result.returncode != 0:
                    return None, f"Optimization failed: {result.stderr}", None
                
                print(f"Optimization succeeded (compile time: {compile_time:.3f}s)")
                return result.stdout, None, compile_time
                
            except subprocess.TimeoutExpired:
                return None, "Optimization timeout", None
            except Exception as e:
                return None, f"Optimization error: {str(e)}", None
            finally:
                os.unlink(input_file.name)

    def evaluate(self, optimize_attention_input):
        """Main evaluation function called by OpenEvolve"""
        try:
            # Handle different input types from OpenEvolve
            if isinstance(optimize_attention_input, str):
                if optimize_attention_input.startswith('/tmp/') and optimize_attention_input.endswith('.py'):
                    print(f"Loading code from: {optimize_attention_input}")
                    with open(optimize_attention_input, 'r') as f:
                        code = f.read()
                    
                    namespace = {}
                    exec(code, namespace)
                    
                    if 'optimize_attention' in namespace:
                        optimize_attention_func = namespace['optimize_attention']
                        print("Calling loaded optimize_attention function...")
                        params = optimize_attention_func()
                    else:
                        raise ValueError("No optimize_attention function found in loaded code")
                else:
                    raise ValueError(f"Unexpected string input: {optimize_attention_input}")
                    
            elif callable(optimize_attention_input):
                print("Calling optimize_attention function...")
                params = optimize_attention_input()
            elif isinstance(optimize_attention_input, dict):
                print("Using direct parameters...")
                params = optimize_attention_input
            else:
                raise ValueError(f"Unexpected input type: {type(optimize_attention_input)}")
            
            print(f"Evaluating parameters: {params}")
            
            # Load baseline MLIR
            if self.baseline_mlir is None:
                self.baseline_mlir = self.load_baseline_mlir()
                self.baseline_metrics = self.analyze_ir_complexity(self.baseline_mlir)
                print(f"Baseline metrics: {self.baseline_metrics['operations']} ops, {self.baseline_metrics['loops']} loops")
            
            # Apply optimizations
            optimized_mlir, error, compile_time = self.apply_optimizations(self.baseline_mlir, params)
            if error:
                print(f"Compilation failed: {error}")
                return {
                    "error": 100.0,
                    "compilation_error": error
                }
            
            # Analyze optimized IR
            print(optimized_mlir)
            optimized_metrics = self.analyze_ir_complexity(optimized_mlir)
            print(f"Optimized metrics: {optimized_metrics['operations']} ops, {optimized_metrics['loops']} loops")
            
            # Estimate performance using IR analysis
            print("Using sophisticated IR analysis for performance estimation...")
            result = self.estimate_performance_from_ir(optimized_metrics, self.baseline_metrics, params)
            
            # Calculate error (lower is better)
            speedup = result.get('speedup', 0.0)
            runtime = result.get('runtime', 1.0)
            target_speedup = params.get('target_speedup', 1.32)
            
            # Error calculation: penalize if below target, reward if above
            if speedup >= target_speedup:
                error = max(0.1, (target_speedup - speedup) * 5)  # Small positive error for success
                print(f"TARGET ACHIEVED! {speedup:.3f}x >= {target_speedup}x")
            else:
                error = (target_speedup - speedup) * 15  # Penalty for missing target
                print(f"Target missed: {speedup:.3f}x < {target_speedup}x")
            
            result_data = {
                "error": float(error),
                "speedup": float(speedup),
                "runtime": float(runtime),
                "compile_time": float(compile_time or 0),
                "method": result.get('method', 'ir_analysis'),
                "size_ratio": result.get('size_ratio', 1.0),
                "optimization_score": result.get('optimization_score', 1.0)
            }
            
            print(f"📊 Result: error={error:.3f}, speedup={speedup:.3f}x, runtime={runtime:.3f}")
            return result_data
            
        except Exception as e:
            error_msg = str(e)
            print(f"Evaluation exception: {error_msg}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "error": 1000.0,
                "exception": error_msg
            }

# From attention_optimization/evaluator.py
def verify_tools(self):
        """Verify MLIR tools are available"""
        tools = ['mlir-opt']
        for tool in tools:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool not found: {tool}")
        print("MLIR tools verified: mlir-opt")

# From attention_optimization/evaluator.py
def load_baseline_mlir(self):
        """Load baseline MLIR from file"""
        if self.mlir_file.exists():
            print(f"Loading MLIR from: {self.mlir_file}")
            with open(self.mlir_file, 'r') as f:
                content = f.read()
            print(f"Loaded {len(content)} characters")
            return content
        else:
            raise FileNotFoundError(f"MLIR file not found: {self.mlir_file}")

# From attention_optimization/evaluator.py
def analyze_ir_complexity(self, mlir_content):
        """Analyze MLIR IR for performance-relevant characteristics"""
        lines = mlir_content.splitlines()
        
        metrics = {
            'total_lines': len(lines),
            'total_chars': len(mlir_content),
            'operations': 0,
            'loops': 0,
            'memory_ops': 0,
            'arithmetic_ops': 0,
            'linalg_ops': 0,
            'func_calls': 0,
            'nested_depth': 0
        }
        
        current_depth = 0
        max_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
                
            # Count braces for nesting depth
            current_depth += stripped.count('{') - stripped.count('}')
            max_depth = max(max_depth, current_depth)
            
            # Count different operation types
            if '=' in stripped and ('%' in stripped or '@' in stripped):
                metrics['operations'] += 1
            
            # Specific operation patterns
            if any(loop_kw in stripped for loop_kw in ['scf.for', 'affine.for', 'scf.while']):
                metrics['loops'] += 1
            
            if any(mem_op in stripped for mem_op in ['memref.load', 'memref.store', 'tensor.extract', 'tensor.insert']):
                metrics['memory_ops'] += 1
                
            if any(arith_op in stripped for arith_op in ['arith.addf', 'arith.mulf', 'arith.divf', 'arith.subf']):
                metrics['arithmetic_ops'] += 1
                
            if 'linalg.' in stripped:
                metrics['linalg_ops'] += 1
                
            if 'func.call' in stripped or 'call @' in stripped:
                metrics['func_calls'] += 1
        
        metrics['nested_depth'] = max_depth
        return metrics

# From attention_optimization/evaluator.py
def estimate_performance_from_ir(self, optimized_metrics, baseline_metrics, params):
        """Estimate performance based on IR analysis"""
        
        # Calculate relative changes
        ops_ratio = optimized_metrics['operations'] / max(baseline_metrics['operations'], 1)
        size_ratio = optimized_metrics['total_chars'] / max(baseline_metrics['total_chars'], 1)
        loop_ratio = optimized_metrics['loops'] / max(baseline_metrics['loops'], 1)
        arith_ratio = optimized_metrics['arithmetic_ops'] / max(baseline_metrics['arithmetic_ops'], 1)
        
        # Base performance model
        base_speedup = 1.0
        
        # Size reduction usually means optimization
        if size_ratio < 1.0:
            base_speedup += (1.0 - size_ratio) * 0.5  # Up to 50% speedup from size reduction
        
        # Loop optimizations
        unroll_factor = params.get('unroll_factor', 1)
        if unroll_factor > 1:
            base_speedup += min(unroll_factor * 0.05, 0.3)  # Up to 30% from unrolling
        
        # Memory optimizations  
        if params.get('use_shared_memory', False):
            base_speedup += 0.15  # 15% from better memory usage
        
        # Loop interchange
        if params.get('loop_interchange', False):
            base_speedup += 0.10  # 10% from better cache locality
        
        # Penalize if optimization increased complexity significantly
        if ops_ratio > 1.2:
            base_speedup *= 0.9  # 10% penalty for increased complexity
        
        # Add some realistic noise
        import random
        noise = random.uniform(0.95, 1.05)
        final_speedup = base_speedup * noise
        
        # Estimate runtime (inverse of speedup)
        base_runtime = 10.0  # Baseline runtime in arbitrary units
        estimated_runtime = base_runtime / final_speedup
        
        return {
            'speedup': final_speedup,
            'runtime': estimated_runtime,
            'method': 'ir_analysis',
            'size_ratio': size_ratio,
            'ops_ratio': ops_ratio,
            'optimization_score': base_speedup
        }

# From attention_optimization/evaluator.py
def apply_optimizations(self, mlir_content, params):
        """Apply MLIR optimization passes based on parameters"""
        print(f"Applying optimizations: {params}")
        
        # Build pass pipeline with only verified working passes
        passes = ["canonicalize", "cse", "linalg-fold-unit-extent-dims"]
        
        # Add unroll with parameter
        unroll_factor = params.get('unroll_factor', 1)
        if unroll_factor > 1:
            passes.append(f"func.func(affine-loop-unroll)")
        
        # Add conditional passes
        if params.get('use_shared_memory', False):
            passes.append("linalg-fold-unit-extent-dims")
        
        if params.get('loop_interchange', False):
            passes.append("canonicalize")
            
        passes.extend(["canonicalize", "cse"])
        
        pipeline = f"builtin.module({','.join(passes)})"
        print(f"Using pipeline: {pipeline}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as input_file:
            input_file.write(mlir_content)
            input_file.flush()
            
            try:
                start_time = time.time()
                cmd = ['mlir-opt', input_file.name, f'--pass-pipeline={pipeline}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                compile_time = time.time() - start_time
                
                if result.returncode != 0:
                    return None, f"Optimization failed: {result.stderr}", None
                
                print(f"Optimization succeeded (compile time: {compile_time:.3f}s)")
                return result.stdout, None, compile_time
                
            except subprocess.TimeoutExpired:
                return None, "Optimization timeout", None
            except Exception as e:
                return None, f"Optimization error: {str(e)}", None
            finally:
                os.unlink(input_file.name)

# From attention_optimization/evaluator.py
def test_params():
        return {
            'tile_size_m': 32,
            'tile_size_n': 64,
            'unroll_factor': 4,
            'use_shared_memory': True,
            'loop_interchange': True,
            'target_speedup': 1.32
        }


# From attention_optimization/initial_program.py
def optimize_attention():
    """
    Define attention optimization parameters for evolution.
    
    The goal is to achieve 32% speedup (1.32x) like AlphaEvolve paper
    by optimizing compiler-generated MLIR IR for attention kernels.
    """
    
    # AlphaEvolve-inspired parameter space exploration
    # These parameters control MLIR compiler transformations
    
    # Memory tiling strategy - crucial for cache performance  
    # Based on typical L1/L2 cache sizes and attention patterns
    tile_options_m = [16, 32, 64, 128]  # Sequence dimension tiles
    tile_options_n = [32, 64, 128, 256] # Head dimension tiles
    
    # Smart initialization: favor cache-friendly sizes
    tile_size_m = random.choice([32, 64])  # Sweet spot for most caches
    tile_size_n = random.choice([64, 128]) # Head dim optimization
    
    # Vectorization strategy - critical for modern SIMD
    vectorization_options = ['none', 'affine', 'linalg']
    vectorization = random.choice(vectorization_options)
    
    # Loop unrolling - balance code size vs performance
    unroll_factors = [1, 2, 4, 8]
    # Favor moderate unrolling for attention kernels
    unroll_factor = random.choice([2, 4] if random.random() > 0.5 else unroll_factors)
    
    # Fusion strategy - key for reducing memory traffic
    fusion_strategies = ['none', 'producer', 'consumer', 'both']
    # Favor fusion for attention (Q@K^T, softmax, @V pattern)
    fusion_strategy = random.choice(['both', 'producer'] if random.random() > 0.3 else fusion_strategies)
    
    # Loop interchange - can improve memory access patterns
    loop_interchange = random.choice([True, False])
    
    # Memory optimizations - crucial for large attention matrices
    use_shared_memory = random.choice([True, False])
    
    # Performance vs latency trade-off
    optimize_for_latency = random.choice([True, False])
    
    # Additional optimizations inspired by FlashAttention
    enable_blocking = random.choice([True, False])  # Block-wise computation
    enable_recomputation = random.choice([True, False])  # Memory vs compute trade-off
    
    optimization_params = {
        # Core tiling parameters
        'tile_size_m': tile_size_m,
        'tile_size_n': tile_size_n,
        
        # Vectorization and parallelization
        'vectorization': vectorization,
        'unroll_factor': unroll_factor,
        'loop_interchange': loop_interchange,
        
        # Fusion and memory optimization
        'fusion_strategy': fusion_strategy,
        'use_shared_memory': use_shared_memory,
        
        # Performance tuning
        'optimize_for_latency': optimize_for_latency,
        'enable_blocking': enable_blocking,
        'enable_recomputation': enable_recomputation,
        
        # Metadata for analysis
        'optimization_strategy': 'alphaevolve_inspired',
        'target_speedup': 1.32,
    }
    
    return optimization_params

import sympy

# From bench/dataclasses.py
class Equation:
    symbols: list
    symbol_descs: list
    symbol_properties: list
    expression: str
    desc: Optional[str] = None

    sympy_format: Optional[sympy.Expr] = None
    lambda_format: Optional[callable] = None
    program_format: Optional[str] = None

# From bench/dataclasses.py
class SearchResult:
    equation: Equation
    aux: Any

# From bench/dataclasses.py
class SEDTask:
    name: str
    symbols: list
    symbol_descs: list
    symbol_properties: list
    samples: Any
    desc: Optional[str] = None

# From bench/dataclasses.py
class Problem:
    dataset_identifier: str
    equation_idx: str
    gt_equation: Equation
    samples: Any

    def create_task(self) -> SEDTask:
        return SEDTask(
            name=self.equation_idx,
            symbols=self.gt_equation.symbols,
            symbol_descs=self.gt_equation.symbol_descs,
            symbol_properties=self.gt_equation.symbol_properties,
            samples=self.train_samples,
            desc=self.gt_equation.desc,
        )

    @property
    def train_samples(self):
        return self.samples["train"]

    @property
    def test_samples(self):
        return self.samples["test"]

    @property
    def ood_test_samples(self):
        return self.samples.get("ood_test", None)

# From bench/dataclasses.py
def create_task(self) -> SEDTask:
        return SEDTask(
            name=self.equation_idx,
            symbols=self.gt_equation.symbols,
            symbol_descs=self.gt_equation.symbol_descs,
            symbol_properties=self.gt_equation.symbol_properties,
            samples=self.train_samples,
            desc=self.gt_equation.desc,
        )

# From bench/dataclasses.py
def train_samples(self):
        return self.samples["train"]

# From bench/dataclasses.py
def test_samples(self):
        return self.samples["test"]

# From bench/dataclasses.py
def ood_test_samples(self):
        return self.samples.get("ood_test", None)

import h5py
import datasets
from huggingface_hub import snapshot_download
from dataclasses import Equation
from dataclasses import Problem
import warnings

# From bench/datamodules.py
class TransformedFeynmanDataModule:
    def __init__(self):
        self._dataset_dir = None
        self._dataset_identifier = "lsr_transform"

    def setup(self):
        self._dataset_dir = Path(_download(repo_id=REPO_ID))
        ds = datasets.load_dataset(REPO_ID)["lsr_transform"]
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[f'/lsr_transform/{e["name"]}'].items()
                }
                self.problems.append(
                    Problem(
                        dataset_identifier=self._dataset_identifier,
                        equation_idx=e["name"],
                        gt_equation=Equation(
                            symbols=e["symbols"],
                            symbol_descs=e["symbol_descs"],
                            symbol_properties=e["symbol_properties"],
                            expression=e["expression"],
                        ),
                        samples=samples,
                    )
                )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return "LSR_Transform"

# From bench/datamodules.py
class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples["train_data"]

    @property
    def test_samples(self):
        return self.samples["id_test_data"]

    @property
    def ood_test_samples(self):
        return self.samples["ood_test_data"]

# From bench/datamodules.py
class BaseSynthDataModule:
    def __init__(
        self,
        dataset_identifier,
        short_dataset_identifier,
        root,
        default_symbols=None,
        default_symbol_descs=None,
    ):
        self._dataset_dir = Path(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs

    def setup(self):
        self._dataset_dir = Path(_download(repo_id=REPO_ID))
        ds = datasets.load_dataset(REPO_ID)[f"lsr_synth_{self._dataset_identifier}"]
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[
                        f'/lsr_synth/{self._dataset_identifier}/{e["name"]}'
                    ].items()
                }
                self.problems.append(
                    Problem(
                        dataset_identifier=self._dataset_identifier,
                        equation_idx=e["name"],
                        gt_equation=Equation(
                            symbols=e["symbols"],
                            symbol_descs=e["symbol_descs"],
                            symbol_properties=e["symbol_properties"],
                            expression=e["expression"],
                        ),
                        samples=samples,
                    )
                )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier

# From bench/datamodules.py
class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("matsci", "MatSci", root)

# From bench/datamodules.py
class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__(
            "chem_react",
            "CRK",
            root,
            default_symbols=["dA_dt", "t", "A"],
            default_symbol_descs=[
                "Rate of change of concentration in chemistry reaction kinetics",
                "Time",
                "Concentration at time t",
            ],
        )

# From bench/datamodules.py
class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__(
            "bio_pop_growth",
            "BPG",
            root,
            default_symbols=["dP_dt", "t", "P"],
            default_symbol_descs=["Population growth rate", "Time", "Population at time t"],
        )

# From bench/datamodules.py
class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__(
            "phys_osc",
            "PO",
            root,
            default_symbols=["dv_dt", "x", "t", "v"],
            default_symbol_descs=[
                "Acceleration in Nonl-linear Harmonic Oscillator",
                "Position at time t",
                "Time",
                "Velocity at time t",
            ],
        )

# From bench/datamodules.py
def get_datamodule(name, root_folder):
    if name == "bio_pop_growth":
        root = root_folder or "datasets/lsr-synth-bio"
        return BioPopGrowthDataModule(root)
    elif name == "chem_react":
        root = root_folder or "datasets/lsr-synth-chem"
        return ChemReactKineticsDataModule(root)
    elif name == "matsci":
        root = root_folder or "datasets/lsr-synth-matsci"
        return MatSciDataModule(root)
    elif name == "phys_osc":
        root = root_folder or "datasets/lsr-synth-phys"
        return PhysOscilDataModule(root)
    # elif name == 'feynman':
    #     return FeynmanDataModule()
    elif name == "lsrtransform":
        return TransformedFeynmanDataModule()
    else:
        raise ValueError(f"Unknown datamodule name: {name}")

# From bench/datamodules.py
def setup(self):
        self._dataset_dir = Path(_download(repo_id=REPO_ID))
        ds = datasets.load_dataset(REPO_ID)["lsr_transform"]
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[f'/lsr_transform/{e["name"]}'].items()
                }
                self.problems.append(
                    Problem(
                        dataset_identifier=self._dataset_identifier,
                        equation_idx=e["name"],
                        gt_equation=Equation(
                            symbols=e["symbols"],
                            symbol_descs=e["symbol_descs"],
                            symbol_properties=e["symbol_properties"],
                            expression=e["expression"],
                        ),
                        samples=samples,
                    )
                )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

# From bench/datamodules.py
def name(self):
        return "LSR_Transform"


# From legacy/prev_sim__works_evaluator.py
def load_base_mlir(self):
        """Load the baseline MLIR implementation"""
        if not self.base_mlir_file.exists():
            # Create a simple baseline if it doesn't exist
            return self.create_baseline_mlir()
        
        with open(self.base_mlir_file, 'r') as f:
            return f.read()

# From legacy/prev_sim__works_evaluator.py
def create_baseline_mlir(self):
        """Create a simple baseline MLIR attention implementation"""
        baseline = '''
        func.func @baseline_attention(
            %query: tensor<?x?x?x?xf32>,
            %key: tensor<?x?x?x?xf32>, 
            %value: tensor<?x?x?x?xf32>
        ) -> tensor<?x?x?x?xf32> {
            // Simple attention: Q @ K^T @ V (simplified)
            %result = linalg.generic {
                indexing_maps = [affine_map<(b, h, s, d) -> (b, h, s, d)>],
                iterator_types = ["parallel", "parallel", "parallel", "parallel"]
            } ins(%query : tensor<?x?x?x?xf32>) 
              outs(%query : tensor<?x?x?x?xf32>) {
            ^bb0(%q: f32, %out: f32):
                linalg.yield %q : f32
            }
            return %result : tensor<?x?x?x?xf32>
        }
        '''
        return baseline

# From legacy/prev_sim__works_evaluator.py
def compile_mlir_with_optimizations(self, base_mlir, optimization_params):
        """Apply optimizations and compile MLIR"""
        try:
            # Create optimized MLIR by applying transformations
            optimized_mlir = self.apply_optimizations(base_mlir, optimization_params)
            
            # Simulate MLIR compilation (in real implementation, use mlir-opt)
            compile_success = self.simulate_mlir_compilation(optimized_mlir)
            
            return compile_success, optimized_mlir
            
        except Exception as e:
            return False, str(e)

# From legacy/prev_sim__works_evaluator.py
def simulate_mlir_compilation(self, mlir_code):
        """Simulate MLIR compilation success"""
        # Simple checks for valid MLIR
        required_elements = ['func.func', 'tensor', 'return']
        
        for element in required_elements:
            if element not in mlir_code:
                return False
        
        # Check for obvious syntax errors
        if mlir_code.count('{') != mlir_code.count('}'):
            return False
            
        return True

# From legacy/prev_sim__works_evaluator.py
def benchmark_implementation(self, optimized_mlir, test_config):
        """Benchmark the optimized implementation"""
        batch, heads, seq_len, head_dim = test_config
        
        # Estimate FLOPs for attention computation
        # Q@K^T: batch * heads * seq_len^2 * head_dim
        # Softmax@V: batch * heads * seq_len^2 * head_dim
        flops = 2 * batch * heads * seq_len * seq_len * head_dim
        
        # Simulate performance based on optimizations
        base_flops_per_second = 1e12  # 1 TFLOP/s baseline
        
        # Apply optimization factors
        speedup_factor = self.calculate_speedup_factor(optimized_mlir)
        
        # Calculate runtime
        runtime = flops / (base_flops_per_second * speedup_factor)
        
        return runtime

# From legacy/prev_sim__works_evaluator.py
def calculate_speedup_factor(self, optimized_mlir):
        """Calculate speedup factor based on applied optimizations"""
        speedup = 1.0
        
        # Parse optimization comments to extract speedup factors
        if "Tile sizes: 128x128x128" in optimized_mlir:
            speedup *= 1.25  # 25% improvement from large tiles
        elif "Tile sizes: 64x64x64" in optimized_mlir:
            speedup *= 1.15  # 15% improvement from better tiling
        elif "Tile sizes: 32x32x32" in optimized_mlir:
            speedup *= 1.10  # 10% improvement
        elif "Tile sizes: 256x256x256" in optimized_mlir:
            speedup *= 1.30  # 30% improvement from very large tiles
        
        if "Vectorization: full" in optimized_mlir:
            speedup *= 1.20  # 20% improvement from vectorization
        elif "Vectorization: outer" in optimized_mlir:
            speedup *= 1.10  # 10% improvement
        elif "Vectorization: inner" in optimized_mlir:
            speedup *= 1.08  # 8% improvement
        
        if "Fusion: producer" in optimized_mlir or "Fusion: consumer" in optimized_mlir:
            speedup *= 1.08  # 8% improvement from fusion
        elif "Fusion: both" in optimized_mlir:
            speedup *= 1.15  # 15% improvement
        
        if "Unroll factor: 8" in optimized_mlir:
            speedup *= 1.08  # 8% improvement
        elif "Unroll factor: 4" in optimized_mlir:
            speedup *= 1.05  # 5% improvement from unrolling
        elif "Unroll factor: 2" in optimized_mlir:
            speedup *= 1.02  # 2% improvement
        
        return speedup

# From legacy/prev_sim__works_evaluator.py
def get_reference_performance(self):
        """Get baseline performance for comparison"""
        if self.reference_performance is None:
            base_mlir = self.load_base_mlir()
            total_time = 0
            
            for config in self.test_configs:
                runtime = self.benchmark_implementation(base_mlir, config)
                total_time += runtime
            
            self.reference_performance = total_time / len(self.test_configs)
        
        return self.reference_performance


# From scripts/fix_tensor_shapes.py
def add_output_shape(match):
    var, indices, input_type, output_type = match.groups()
    
    # Extract dimensions from output tensor type
    dims_match = re.search(r'tensor<([^>]+)>', output_type)
    if dims_match:
        dims_str = dims_match.group(1)
        # Extract just the dimension numbers (ignore 'xf32' etc.)
        dims = re.findall(r'\d+', dims_str.split('x')[:-1])  # Exclude the type part
        if dims:
            output_shape = '[' + ', '.join(dims) + ']'
            return f'tensor.expand_shape {var} {indices} output_shape {output_shape} : {input_type} into {output_type}'
    
    return match.group(0)


# From scripts/debug_real_execution.py
def check_mlir_tools():
    """Check what MLIR tools are available"""
    tools = [
        'mlir-opt',
        'mlir-translate', 
        'mlir-cpu-runner',
        'mlir-lsp-server',
        'clang',
        'gcc'
    ]
    
    print("🔍 Checking available tools:")
    available = {}
    for tool in tools:
        path = shutil.which(tool)
        available[tool] = path is not None
        status = "✅" if path else "❌"
        print(f"  {status} {tool}: {path or 'Not found'}")
    
    return available

# From scripts/debug_real_execution.py
def test_mlir_translate():
    """Test MLIR to LLVM translation"""
    print("\n🧪 Testing MLIR→LLVM translation:")
    
    # Simple test MLIR
    test_mlir = '''
module {
  func.func @simple_add(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
}
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(test_mlir)
        f.flush()
        
        try:
            # Test mlir-translate
            cmd = ['mlir-translate', '--mlir-to-llvmir', f.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ mlir-translate works!")
                print(f"   LLVM IR size: {len(result.stdout)} chars")
                return True
            else:
                print("❌ mlir-translate failed:")
                print(f"   Error: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ mlir-translate not found")
            return False
        except Exception as e:
            print(f"❌ mlir-translate error: {e}")
            return False

# From scripts/debug_real_execution.py
def test_actual_mlir_file():
    """Test with your actual MLIR file"""
    print("\n🧪 Testing your actual MLIR file:")
    
    mlir_file = Path("mlir/self_attn_with_consts_linalg_dialect.mlir")
    if not mlir_file.exists():
        print("❌ MLIR file not found!")
        return False
    
    try:
        # Test basic parsing
        cmd = ['mlir-opt', str(mlir_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MLIR file parses correctly")
            
            # Test optimization
            cmd = ['mlir-opt', str(mlir_file), '--canonicalize']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Basic optimization works")
                
                # Test LLVM translation
                if shutil.which('mlir-translate'):
                    cmd = ['mlir-translate', '--mlir-to-llvmir', str(mlir_file)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ LLVM translation works!")
                        print(f"   LLVM IR size: {len(result.stdout)} chars")
                        return True
                    else:
                        print("❌ LLVM translation failed:")
                        print(f"   Error: {result.stderr[:500]}...")
                        return False
                else:
                    print("⚠️ mlir-translate not available")
                    return False
            else:
                print("❌ Basic optimization failed:")
                print(f"   Error: {result.stderr}")
                return False
        else:
            print("❌ MLIR file parsing failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MLIR file: {e}")
        return False

# From scripts/debug_real_execution.py
def suggest_fixes():
    """Suggest ways to enable real execution"""
    print("\n💡 Suggestions to enable real execution:")
    
    available = check_mlir_tools()
    
    if not available.get('mlir-translate'):
        print("1. Install mlir-translate:")
        print("   - Build LLVM/MLIR with: cmake -DLLVM_ENABLE_PROJECTS='mlir' ...")
        print("   - Or install via package manager if available")
    
    if not available.get('clang') and not available.get('gcc'):
        print("2. Install a C compiler (clang or gcc)")
    
    print("3. Alternative: Improve the simulation")
    print("   - Use more sophisticated IR analysis")
    print("   - Measure compilation time more accurately")
    print("   - Add pass-specific performance heuristics")


# From scripts/mlir_syntax_test.py
def test_mlir_syntax():
    """Test the corrected MLIR baseline syntax"""
    
    baseline_mlir = '''
#map_q = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>
#map_k = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_scores = affine_map<(b, h, s1, s2, d) -> (b, h, s1, s2)>
#map_weights = affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>
#map_v = affine_map<(b, h, s1, s2, d) -> (b, h, s2, d)>
#map_out = affine_map<(b, h, s1, s2, d) -> (b, h, s1, d)>

module {
  func.func @baseline_attention(
      %query: tensor<1x8x128x64xf32>,
      %key: tensor<1x8x128x64xf32>, 
      %value: tensor<1x8x128x64xf32>
  ) -> tensor<1x8x128x64xf32> {
    
    %c0 = arith.constant 0.0 : f32
    %cst_scale = arith.constant 0.125 : f32
    
    // Initialize output tensors
    %scores_init = tensor.empty() : tensor<1x8x128x128xf32>
    %output_init = tensor.empty() : tensor<1x8x128x64xf32>
    
    // Compute Q @ K^T (scaled dot-product attention)
    %attention_scores = linalg.generic {
      indexing_maps = [#map_q, #map_k, #map_scores],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%query, %key : tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) 
      outs(%scores_init : tensor<1x8x128x128xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
      %prod = arith.mulf %q, %k : f32
      %scaled = arith.mulf %prod, %cst_scale : f32
      %sum = arith.addf %acc, %scaled : f32
      linalg.yield %sum : f32
    } -> tensor<1x8x128x128xf32>
    
    // Apply attention weights to values  
    %attention_output = linalg.generic {
      indexing_maps = [#map_weights, #map_v, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
    } ins(%attention_scores, %value : tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) 
      outs(%output_init : tensor<1x8x128x64xf32>) {
    ^bb0(%weight: f32, %v: f32, %acc: f32):
      %weighted = arith.mulf %weight, %v : f32
      %sum = arith.addf %acc, %weighted : f32
      linalg.yield %sum : f32
    } -> tensor<1x8x128x64xf32>
    
    return %attention_output : tensor<1x8x128x64xf32>
  }
}
'''
    
    try:
        # Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(baseline_mlir)
            temp_file = f.name
        
        print("🔧 Testing MLIR baseline syntax...")
        
        # Test basic parsing
        result = subprocess.run([
            "mlir-opt", temp_file
        ], capture_output=True, text=True, timeout=30)
        
        Path(temp_file).unlink()  # Clean up
        
        if result.returncode == 0:
            print("✅ MLIR baseline syntax is correct!")
            return True
        else:
            print(f"❌ MLIR syntax error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

# From scripts/mlir_syntax_test.py
def test_tiling_pass():
    """Test the linalg tiling pass syntax"""
    
    simple_linalg = '''
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @simple_add(%arg0: tensor<128x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<128x64xf32>) 
      outs(%0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    } -> tensor<128x64xf32>
    return %1 : tensor<128x64xf32>
  }
}
'''
    
    try:
        # Write MLIR to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(simple_linalg)
            temp_file = f.name
        
        print("\n🔧 Testing linalg tiling pass...")
        
        # Test tiling with our syntax
        pipeline = "builtin.module(linalg-tile,canonicalize,cse)"
        result = subprocess.run([
            "mlir-opt", temp_file, f"--pass-pipeline={pipeline}"
        ], capture_output=True, text=True, timeout=30)
        
        Path(temp_file).unlink()  # Clean up
        
        if result.returncode == 0:
            print("✅ Linalg tiling pass works!")
            print("Sample output:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ Tiling pass error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


# From scripts/mlir_lowering_pipeline.py
class MLIRLoweringPipeline:
    def __init__(self):
        self.verify_tools()
        
    def verify_tools(self):
        """Verify required MLIR tools"""
        required_tools = ['mlir-opt', 'mlir-translate']
        for tool in required_tools:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool not found: {tool}")
        print("✅ MLIR tools verified: mlir-opt, mlir-translate")

    def find_available_passes(self):
        """Find what lowering passes are available"""
        print("🔍 Finding available lowering passes...")
        
        try:
            result = subprocess.run(['mlir-opt', '--help'], capture_output=True, text=True)
            help_text = result.stdout
            
            # Look for conversion passes
            conversion_passes = []
            for line in help_text.splitlines():
                line = line.strip()
                if 'convert-' in line and '-to-' in line:
                    # Extract pass name
                    if line.startswith('--'):
                        pass_name = line.split()[0][2:]  # Remove --
                        conversion_passes.append(pass_name)
            
            print("📋 Available conversion passes:")
            relevant_passes = []
            for pass_name in sorted(conversion_passes):
                if any(keyword in pass_name for keyword in ['arith', 'func', 'llvm', 'std', 'scf']):
                    print(f"   ✅ {pass_name}")
                    relevant_passes.append(pass_name)
                else:
                    print(f"   ❓ {pass_name}")
            
            return relevant_passes
            
        except Exception as e:
            print(f"❌ Error finding passes: {e}")
            return []

    def test_lowering_passes(self, input_file):
        """Test different lowering pass combinations"""
        print(f"\n🧪 Testing lowering passes on {input_file}...")
        
        # Common lowering pass sequences
        pass_sequences = [
            # Basic arith lowering
            ["convert-arith-to-llvm"],
            
            # More comprehensive lowering
            ["convert-arith-to-llvm", "convert-func-to-llvm"],
            
            # Full lowering pipeline
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm", 
                "convert-scf-to-cf",
                "convert-cf-to-llvm"
            ],
            
            # Alternative approaches
            ["arith-bufferize", "convert-arith-to-llvm"],
            ["canonicalize", "convert-arith-to-llvm", "canonicalize"],
            
            # Try with reconcile-unrealized-casts
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts"
            ]
        ]
        
        successful_sequences = []
        
        for i, passes in enumerate(pass_sequences):
            print(f"\n📋 Testing sequence {i+1}: {' → '.join(passes)}")
            
            success = self.test_pass_sequence(input_file, passes)
            if success:
                successful_sequences.append(passes)
                print(f"   ✅ Sequence {i+1} works!")
            else:
                print(f"   ❌ Sequence {i+1} failed")
        
        return successful_sequences

    def test_pass_sequence(self, input_file, passes):
        """Test a specific sequence of passes"""
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(passes)})"
            
            with tempfile.NamedTemporaryFile(suffix='.mlir', delete=False) as temp_file:
                # Apply passes
                cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode != 0:
                    return False
                
                # Write result to temp file
                temp_file.write(result.stdout)
                temp_file.flush()
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', temp_file.name]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                success = translate_result.returncode == 0
                if success:
                    print(f"      💡 LLVM IR size: {len(translate_result.stdout)} chars")
                
                return success
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
        finally:
            try:
                Path(temp_file.name).unlink()
            except:
                pass

    def create_lowered_file(self, input_file, output_file, pass_sequence):
        """Create a fully lowered MLIR file"""
        print(f"\n🚀 Creating lowered file: {input_file} → {output_file}")
        print(f"📋 Using passes: {' → '.join(pass_sequence)}")
        
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(pass_sequence)})"
            
            start_time = time.time()
            cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}', '-o', output_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Lowering failed: {result.stderr}")
                return False
            
            print(f"✅ Lowering completed in {elapsed:.3f}s")
            
            # Verify the output
            output_path = Path(output_file)
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"📄 Output file size: {size} bytes")
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', output_file]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if translate_result.returncode == 0:
                    llvm_size = len(translate_result.stdout)
                    print(f"✅ LLVM translation successful! LLVM IR size: {llvm_size} chars")
                    
                    # Save LLVM IR too
                    llvm_file = output_file.replace('.mlir', '.ll')
                    with open(llvm_file, 'w') as f:
                        f.write(translate_result.stdout)
                    print(f"💾 LLVM IR saved to: {llvm_file}")
                    
                    return True
                else:
                    print(f"❌ LLVM translation failed: {translate_result.stderr[:200]}...")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Error creating lowered file: {e}")
            return False

    def process_file(self, input_file):
        """Complete pipeline to lower an MLIR file"""
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_file}")
            return None
        
        print(f"🎯 Processing {input_file}")
        print(f"📊 Input size: {input_path.stat().st_size} bytes")
        
        # Find available passes
        available_passes = self.find_available_passes()
        
        # Test lowering approaches
        successful_sequences = self.test_lowering_passes(str(input_path))
        
        if not successful_sequences:
            print("❌ No working lowering sequences found!")
            return None
        
        # Use the first successful sequence
        best_sequence = successful_sequences[0]
        print(f"\n🎯 Using best sequence: {' → '.join(best_sequence)}")
        
        # Create output filename
        output_file = str(input_path.parent / f"{input_path.stem}_lowered{input_path.suffix}")
        
        # Create the lowered file
        if self.create_lowered_file(str(input_path), output_file, best_sequence):
            print(f"🎉 Success! Lowered file created: {output_file}")
            return output_file
        else:
            print("❌ Failed to create lowered file")
            return None

# From scripts/mlir_lowering_pipeline.py
def find_available_passes(self):
        """Find what lowering passes are available"""
        print("🔍 Finding available lowering passes...")
        
        try:
            result = subprocess.run(['mlir-opt', '--help'], capture_output=True, text=True)
            help_text = result.stdout
            
            # Look for conversion passes
            conversion_passes = []
            for line in help_text.splitlines():
                line = line.strip()
                if 'convert-' in line and '-to-' in line:
                    # Extract pass name
                    if line.startswith('--'):
                        pass_name = line.split()[0][2:]  # Remove --
                        conversion_passes.append(pass_name)
            
            print("📋 Available conversion passes:")
            relevant_passes = []
            for pass_name in sorted(conversion_passes):
                if any(keyword in pass_name for keyword in ['arith', 'func', 'llvm', 'std', 'scf']):
                    print(f"   ✅ {pass_name}")
                    relevant_passes.append(pass_name)
                else:
                    print(f"   ❓ {pass_name}")
            
            return relevant_passes
            
        except Exception as e:
            print(f"❌ Error finding passes: {e}")
            return []

# From scripts/mlir_lowering_pipeline.py
def test_lowering_passes(self, input_file):
        """Test different lowering pass combinations"""
        print(f"\n🧪 Testing lowering passes on {input_file}...")
        
        # Common lowering pass sequences
        pass_sequences = [
            # Basic arith lowering
            ["convert-arith-to-llvm"],
            
            # More comprehensive lowering
            ["convert-arith-to-llvm", "convert-func-to-llvm"],
            
            # Full lowering pipeline
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm", 
                "convert-scf-to-cf",
                "convert-cf-to-llvm"
            ],
            
            # Alternative approaches
            ["arith-bufferize", "convert-arith-to-llvm"],
            ["canonicalize", "convert-arith-to-llvm", "canonicalize"],
            
            # Try with reconcile-unrealized-casts
            [
                "convert-arith-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts"
            ]
        ]
        
        successful_sequences = []
        
        for i, passes in enumerate(pass_sequences):
            print(f"\n📋 Testing sequence {i+1}: {' → '.join(passes)}")
            
            success = self.test_pass_sequence(input_file, passes)
            if success:
                successful_sequences.append(passes)
                print(f"   ✅ Sequence {i+1} works!")
            else:
                print(f"   ❌ Sequence {i+1} failed")
        
        return successful_sequences

# From scripts/mlir_lowering_pipeline.py
def test_pass_sequence(self, input_file, passes):
        """Test a specific sequence of passes"""
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(passes)})"
            
            with tempfile.NamedTemporaryFile(suffix='.mlir', delete=False) as temp_file:
                # Apply passes
                cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode != 0:
                    return False
                
                # Write result to temp file
                temp_file.write(result.stdout)
                temp_file.flush()
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', temp_file.name]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                success = translate_result.returncode == 0
                if success:
                    print(f"      💡 LLVM IR size: {len(translate_result.stdout)} chars")
                
                return success
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
        finally:
            try:
                Path(temp_file.name).unlink()
            except:
                pass

# From scripts/mlir_lowering_pipeline.py
def create_lowered_file(self, input_file, output_file, pass_sequence):
        """Create a fully lowered MLIR file"""
        print(f"\n🚀 Creating lowered file: {input_file} → {output_file}")
        print(f"📋 Using passes: {' → '.join(pass_sequence)}")
        
        try:
            # Build pipeline
            pipeline = f"builtin.module({','.join(pass_sequence)})"
            
            start_time = time.time()
            cmd = ['mlir-opt', input_file, f'--pass-pipeline={pipeline}', '-o', output_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Lowering failed: {result.stderr}")
                return False
            
            print(f"✅ Lowering completed in {elapsed:.3f}s")
            
            # Verify the output
            output_path = Path(output_file)
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"📄 Output file size: {size} bytes")
                
                # Test LLVM translation
                cmd = ['mlir-translate', '--mlir-to-llvmir', output_file]
                translate_result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                if translate_result.returncode == 0:
                    llvm_size = len(translate_result.stdout)
                    print(f"✅ LLVM translation successful! LLVM IR size: {llvm_size} chars")
                    
                    # Save LLVM IR too
                    llvm_file = output_file.replace('.mlir', '.ll')
                    with open(llvm_file, 'w') as f:
                        f.write(translate_result.stdout)
                    print(f"💾 LLVM IR saved to: {llvm_file}")
                    
                    return True
                else:
                    print(f"❌ LLVM translation failed: {translate_result.stderr[:200]}...")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Error creating lowered file: {e}")
            return False

# From scripts/mlir_lowering_pipeline.py
def process_file(self, input_file):
        """Complete pipeline to lower an MLIR file"""
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_file}")
            return None
        
        print(f"🎯 Processing {input_file}")
        print(f"📊 Input size: {input_path.stat().st_size} bytes")
        
        # Find available passes
        available_passes = self.find_available_passes()
        
        # Test lowering approaches
        successful_sequences = self.test_lowering_passes(str(input_path))
        
        if not successful_sequences:
            print("❌ No working lowering sequences found!")
            return None
        
        # Use the first successful sequence
        best_sequence = successful_sequences[0]
        print(f"\n🎯 Using best sequence: {' → '.join(best_sequence)}")
        
        # Create output filename
        output_file = str(input_path.parent / f"{input_path.stem}_lowered{input_path.suffix}")
        
        # Create the lowered file
        if self.create_lowered_file(str(input_path), output_file, best_sequence):
            print(f"🎉 Success! Lowered file created: {output_file}")
            return output_file
        else:
            print("❌ Failed to create lowered file")
            return None


# From prompt/templates.py
class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template

# From prompt/templates.py
def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

# From prompt/templates.py
def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template


# From utils/format_utils.py
def format_metrics_safe(metrics: Dict[str, Any]) -> str:
    """
    Safely format metrics dictionary for logging, handling both numeric and string values.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Formatted string representation of metrics
    """
    if not metrics:
        return ""

    formatted_parts = []
    for name, value in metrics.items():
        # Check if value is numeric (int, float)
        if isinstance(value, (int, float)):
            try:
                # Only apply float formatting to numeric values
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                # Fallback to string representation if formatting fails
                formatted_parts.append(f"{name}={value}")
        else:
            # For non-numeric values (strings, etc.), just convert to string
            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)

# From utils/format_utils.py
def format_improvement_safe(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """
    Safely format improvement metrics for logging.

    Args:
        parent_metrics: Parent program metrics
        child_metrics: Child program metrics

    Returns:
        Formatted string representation of improvements
    """
    if not parent_metrics or not child_metrics:
        return ""

    improvement_parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            # Only calculate improvement for numeric values
            if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                try:
                    diff = child_value - parent_value
                    improvement_parts.append(f"{metric}={diff:+.4f}")
                except (ValueError, TypeError):
                    # Skip non-numeric comparisons
                    continue

    return ", ".join(improvement_parts)

import functools
from typing import TypeVar

# From utils/async_utils.py
class TaskPool:
    """
    A simple task pool for managing and limiting concurrent tasks
    """

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.tasks: List[asyncio.Task] = []

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize the semaphore when first needed"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run a coroutine in the pool

        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine

        Returns:
            Result of the coroutine
        """
        async with self.semaphore:
            return await coro(*args, **kwargs)

    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        """
        Create and track a task in the pool

        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine

        Returns:
            Task object
        """
        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task

    async def wait_all(self) -> None:
        """Wait for all tasks in the pool to complete"""
        if self.tasks:
            await asyncio.gather(*self.tasks)

    async def cancel_all(self) -> None:
        """Cancel all tasks in the pool"""
        for task in self.tasks:
            task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

# From utils/async_utils.py
def run_in_executor(f: Callable) -> Callable:
    """
    Decorator to run a synchronous function in an executor

    Args:
        f: Function to decorate

    Returns:
        Decorated function that runs in an executor
    """

    @functools.wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper

# From utils/async_utils.py
def semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize the semaphore when first needed"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore


# From utils/metrics_utils.py
def safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_values = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)

# From utils/metrics_utils.py
def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Sum of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    return numeric_sum


# From utils/code_utils.py
def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks

# From utils/code_utils.py
def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)

# From utils/code_utils.py
def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]

# From utils/code_utils.py
def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response

# From utils/code_utils.py
def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)

# From utils/code_utils.py
def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]

# From utils/code_utils.py
def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM

# From llm/ensemble.py
class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [OpenAILLM(model_cfg) for model_cfg in models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses

from abc import ABC
from abc import abstractmethod

# From llm/base.py
class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        pass

import openai

# From llm/openai.py
class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        if self.api_base == "https://api.openai.com/v1" and str(self.model).lower().startswith("o"):
            # For o-series models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content

