from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.config import (
    ADV_CTGAN_OUTPUT_PATH,
    ADV_CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH,
    ADV_CTGAN_SCRIPT,
    CLASSIFIER_EVAL_SCRIPT,
    CLASSIFIER_OUTPUT_PATH,
    CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH,
    FIN_FRAUD_ROOT,
    LABEL_SCRIPT_PATH,
    LABEL_WORKDIR,
    MODELS_DIR,
    OUTPUT_ADV_CTGAN_PATH,
    OUTPUT_ADV_CTGAN_SYNTHETIC_NOVEL_PATH,
    OUTPUT_CLASSIFIER_RESULTS_PATH,
    OUTPUT_CTGAN_PATH,
    OUTPUT_CTGAN_SYNTHETIC_NOVEL_PATH,
    OUTPUT_DIR,
    OUTPUT_MEMORY_PATH,
    OUTPUT_MODELS_DIR,
    OUTPUT_PREPARED_DATASET_PATH,
    OUTPUT_ROBUSTNESS_RESULTS_PATH,
    ORIGINAL_DATASET_PATH,
    ROBUSTNESS_EVAL_SCRIPT,
    ROBUSTNESS_OUTPUT_PATH,
    RUN_SUMMARY_PATH,
    SCRAPED_COMMENTS_PATH,
    SCRAPED_LABELED_CSV_PATH,
    SCRAPED_LABELED_JSON_PATH,
    SCRAPED_POSTS_PATH,
    SCRAPER_SCRIPT_PATH,
    SCRAPER_WORKDIR,
    STANDARD_CTGAN_SCRIPT,
    TOOL_CALL_OUTPUTS_PATH,
    CTGAN_OUTPUT_PATH,
)
from agent.dataset_prep import prepare_main_dataset, sync_scraped_annotations_into_main_csv
from agent.memory.store import AgentMemory
from agent.tool.pipeline_tools import FraudIntelTool, OllamaAdvisorTool, PythonScriptTool


@dataclass(frozen=True)
class PipelineStep:
    name: str
    description: str
    script_path: Path
    expected_output: Path | None = None


class FraudPipelineAgent:
    def __init__(self) -> None:
        self.memory = AgentMemory()
        self.intel_tool = FraudIntelTool()
        self.ollama_tool = OllamaAdvisorTool()
        self.script_tool = PythonScriptTool()
        self.tool_call_outputs: list[dict] = []
        self.current_run_context: dict = {}
        self.tools = [
            self.intel_tool.spec.as_dict(),
            self.ollama_tool.spec.as_dict(),
            self.script_tool.spec.as_dict(),
        ]
        self.pipeline_steps = [
            PipelineStep(
                name="standard_ctgan",
                description="Generate synthetic data with standard CTGAN",
                script_path=STANDARD_CTGAN_SCRIPT,
                expected_output=CTGAN_OUTPUT_PATH,
            ),
            PipelineStep(
                name="adv_ctgan",
                description="Generate synthetic data with adversarial CTGAN",
                script_path=ADV_CTGAN_SCRIPT,
                expected_output=ADV_CTGAN_OUTPUT_PATH,
            ),
            PipelineStep(
                name="classifier_eval",
                description="Train and compare classifiers",
                script_path=CLASSIFIER_EVAL_SCRIPT,
                expected_output=CLASSIFIER_OUTPUT_PATH,
            ),
            PipelineStep(
                name="robustness_eval",
                description="Run robustness evaluation",
                script_path=ROBUSTNESS_EVAL_SCRIPT,
                expected_output=ROBUSTNESS_OUTPUT_PATH,
            ),
        ]

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _created_utc_to_iso(self, value: float | int | None) -> str:
        if value in (None, 0, 0.0):
            return ""
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

    def _latest_timestamp_for_artifact(self, artifact_name: str, path: Path) -> datetime | None:
        artifact = self.memory.get_artifact(artifact_name)
        candidates: list[datetime] = []
        if artifact:
            for key in ("updated_at",):
                parsed = self._parse_timestamp(artifact.get(key))
                if parsed is not None:
                    candidates.append(parsed)
        if path.exists():
            candidates.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
        return max(candidates) if candidates else None

    def _artifact_is_fresh(self, artifact_name: str, path: Path, max_age_days: int) -> tuple[bool, str]:
        if not path.exists():
            return False, "missing"
        latest = self._latest_timestamp_for_artifact(artifact_name, path)
        if latest is None:
            return False, "unknown_age"
        age = datetime.now(timezone.utc) - latest
        if age <= timedelta(days=max_age_days):
            return True, f"fresh:{age.days}d"
        return False, f"stale:{age.days}d"

    def _run_python_command(
        self,
        stage_name: str,
        script_path: Path,
        cwd: Path,
        args: list[str] | None = None,
        expected_outputs: list[Path] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(script_path), *(args or [])]
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        self.memory.record_run(
            {
                "stage": stage_name,
                "script": str(script_path),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage_name}' failed.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        for output in expected_outputs or []:
            self.memory.upsert_artifact(
                f"{stage_name}:{output.name}",
                str(output),
                {"exists": output.exists(), "stage": stage_name},
            )
        return result

    def _register_tools(self) -> None:
        self.memory.register_tools(self.tools)

    def _log(self, message: str) -> None:
        print(f"[agent] {message}", flush=True)

    def _record_tool_call(self, tool_name: str, inputs: dict, output: dict | list) -> None:
        self.tool_call_outputs.append(
            {
                "tool": tool_name,
                "inputs": inputs,
                "output": output,
            }
        )

    def _prepare_dataset(self, collection_summary: dict) -> dict:
        self._log("Preparing dataset from Data labeling/outputs/main.csv")
        prepared = prepare_main_dataset()
        snapshot = {
            "dataset_name": "main.csv_normalized_for_finfraud",
            "source_path": str(Path("Data labeling/outputs/main.csv")),
            "prepared_path": str(prepared.output_path),
            "row_count": prepared.row_count,
            "class_counts": prepared.class_counts,
            "column_count": len(prepared.columns),
            "new_scam_row_count": prepared.new_scam_row_count,
            "new_scam_seed_path": str(prepared.new_scam_seed_path),
            "new_scam_report_path": str(prepared.new_scam_report_path),
            "base_row_count": prepared.base_row_count,
            "labeled_scraped_row_count": prepared.labeled_scraped_row_count,
            "labeled_scraped_class_counts": prepared.labeled_scraped_class_counts,
            "labeled_scraped_usable_count": prepared.labeled_scraped_usable_count,
            "new_scam_signatures": prepared.new_scam_signatures,
            "collection_summary": collection_summary,
        }
        self.memory.add_dataset_snapshot(snapshot)
        self.memory.upsert_artifact("prepared_dataset", str(prepared.output_path), snapshot)
        self.memory.update_pipeline_state("data_collection", collection_summary.get("collection", {}))
        self.memory.update_pipeline_state("labeling", collection_summary.get("labeling", {}))
        self.memory.update_pipeline_state("main_dataset_sync", collection_summary.get("main_dataset_sync", {}))
        self.memory.update_pipeline_state(
            "novel_fraud_detection",
            {
                "new_scam_row_count": prepared.new_scam_row_count,
                "new_scam_signatures": prepared.new_scam_signatures,
                "new_scam_seed_path": str(prepared.new_scam_seed_path),
                "new_scam_report_path": str(prepared.new_scam_report_path),
            },
        )
        self._record_tool_call(
            "prepare_main_dataset",
            {"source_path": str(Path("Data labeling/outputs/main.csv"))},
            snapshot,
        )
        self.current_run_context["dataset"] = prepared.to_memory_dict()
        self.current_run_context["collection_flow"] = collection_summary
        self._log(
            "Prepared dataset saved with "
            f"{prepared.row_count} rows, class counts {prepared.class_counts}, "
            f"and {prepared.new_scam_row_count} novel scam rows from the scraper"
        )
        return snapshot

    def _refresh_reddit_collection(self, scrape_max_age_days: int, label_max_age_days: int) -> dict:
        scrape_fresh, scrape_reason = self._artifact_is_fresh("reddit_scrape", SCRAPED_POSTS_PATH, scrape_max_age_days)
        comments_ready = SCRAPED_COMMENTS_PATH.exists()
        should_scrape = not (scrape_fresh and comments_ready)

        if should_scrape:
            self._log("Refreshing Reddit scrape inputs")
            scrape_result = self._run_python_command(
                "reddit_scrape",
                SCRAPER_SCRIPT_PATH,
                SCRAPER_WORKDIR,
                expected_outputs=[SCRAPED_POSTS_PATH, SCRAPED_COMMENTS_PATH],
            )
            self.memory.upsert_artifact(
                "reddit_scrape",
                str(SCRAPED_POSTS_PATH),
                {
                    "posts_path": str(SCRAPED_POSTS_PATH),
                    "comments_path": str(SCRAPED_COMMENTS_PATH),
                    "reason": scrape_reason,
                    "stdout_tail": scrape_result.stdout[-1000:],
                },
            )
            scrape_status = "completed"
        else:
            self._log(f"Skipping Reddit scrape ({scrape_reason})")
            scrape_status = "reused"

        collection = {
            "status": scrape_status,
            "reason": scrape_reason,
            "posts_count": self._count_csv_rows(SCRAPED_POSTS_PATH),
            "comments_count": self._count_csv_rows(SCRAPED_COMMENTS_PATH),
            "posts_path": str(SCRAPED_POSTS_PATH),
            "comments_path": str(SCRAPED_COMMENTS_PATH),
            "available": SCRAPED_POSTS_PATH.exists() and SCRAPED_COMMENTS_PATH.exists(),
            "max_age_days": scrape_max_age_days,
        }

        label_fresh, label_reason = self._artifact_is_fresh(
            "reddit_label",
            SCRAPED_LABELED_CSV_PATH,
            label_max_age_days,
        )
        should_label = collection["available"] and (should_scrape or not label_fresh or not SCRAPED_LABELED_JSON_PATH.exists())

        if should_label:
            self._log("Refreshing scraped-label artifacts")
            label_result = self._run_python_command(
                "label_scraped_reddit",
                LABEL_SCRIPT_PATH,
                LABEL_WORKDIR,
                args=[
                    "--mode",
                    "full",
                    "--posts-file",
                    str(SCRAPED_POSTS_PATH),
                    "--comments-file",
                    str(SCRAPED_COMMENTS_PATH),
                    "--output-csv",
                    str(SCRAPED_LABELED_CSV_PATH),
                    "--output-json",
                    str(SCRAPED_LABELED_JSON_PATH),
                ],
                expected_outputs=[SCRAPED_LABELED_CSV_PATH, SCRAPED_LABELED_JSON_PATH],
            )
            self.memory.upsert_artifact(
                "reddit_label",
                str(SCRAPED_LABELED_CSV_PATH),
                {
                    "output_csv": str(SCRAPED_LABELED_CSV_PATH),
                    "output_json": str(SCRAPED_LABELED_JSON_PATH),
                    "reason": label_reason,
                    "stdout_tail": label_result.stdout[-1000:],
                },
            )
            label_status = "completed"
        elif collection["available"]:
            self._log(f"Skipping scraped labeling ({label_reason})")
            label_status = "reused"
        else:
            label_status = "skipped"

        labeling = self._summarize_labeled_csv(SCRAPED_LABELED_CSV_PATH)
        labeling.update(
            {
                "status": label_status,
                "reason": label_reason,
                "output_csv": str(SCRAPED_LABELED_CSV_PATH),
                "output_json": str(SCRAPED_LABELED_JSON_PATH),
                "max_age_days": label_max_age_days,
            }
        )

        main_sync = {
            "status": "skipped",
            "new_rows_added": 0,
            "base_row_count": 0,
            "final_row_count": 0,
            "min_created_utc_exclusive": 0.0,
            "latest_added_created_utc": 0.0,
            "latest_added_created_at": "",
            "target_path": str(Path("Data labeling/outputs/main.csv")),
        }
        if SCRAPED_LABELED_CSV_PATH.exists():
            previous_sync_state = self.memory.get_pipeline_state("main_dataset_sync") or {}
            min_created_utc_exclusive = float(previous_sync_state.get("latest_added_created_utc", 0) or 0)
            sync_summary = sync_scraped_annotations_into_main_csv(
                min_created_utc_exclusive=min_created_utc_exclusive,
            )
            main_sync.update(sync_summary)
            main_sync["min_created_utc_exclusive"] = min_created_utc_exclusive
            main_sync["latest_added_created_at"] = self._created_utc_to_iso(
                float(sync_summary.get("latest_added_created_utc", 0) or 0)
            )
            main_sync["status"] = "completed" if sync_summary["new_rows_added"] else "reused"
            self.memory.upsert_artifact("main_dataset", str(Path("Data labeling/outputs/main.csv")), main_sync)
            self._log(
                "Main dataset sync "
                f"added {sync_summary['new_rows_added']} new rows "
                f"(final rows: {sync_summary['final_row_count']})"
            )

        summary = {
            "collection": collection,
            "labeling": labeling,
            "main_dataset_sync": main_sync,
        }
        self.current_run_context["collection_flow"] = summary
        return summary

    def _count_csv_rows(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)

    def _summarize_labeled_csv(self, path: Path) -> dict:
        if not path.exists():
            return {
                "labeled_row_count": 0,
                "class_counts": {},
                "usable_for_training_count": 0,
                "high_confidence_count": 0,
            }

        class_counts: dict[str, int] = {}
        usable_for_training_count = 0
        high_confidence_count = 0
        labeled_row_count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labeled_row_count += 1
                label = str(row.get("annotation.is_fraud", "")).strip()
                if label:
                    class_counts[label] = class_counts.get(label, 0) + 1
                usable = str(row.get("annotation.label_quality.usable_for_training", "")).strip().lower()
                if usable in {"true", "1", "yes"}:
                    usable_for_training_count += 1
                bucket = str(row.get("annotation.label_quality.confidence_bucket", "")).strip().lower()
                if bucket == "high":
                    high_confidence_count += 1
        return {
            "labeled_row_count": labeled_row_count,
            "class_counts": class_counts,
            "usable_for_training_count": usable_for_training_count,
            "high_confidence_count": high_confidence_count,
        }

    def _summarize_classifier_results(self) -> dict:
        if not CLASSIFIER_OUTPUT_PATH.exists():
            return {}
        best_row = None
        with CLASSIFIER_OUTPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if best_row is None or float(row.get("F1-Score", 0) or 0) > float(best_row.get("F1-Score", 0) or 0):
                    best_row = row
        return {
            "best_dataset": best_row.get("Dataset", "") if best_row else "",
            "best_classifier": best_row.get("Classifier", "") if best_row else "",
            "best_f1_score": float(best_row.get("F1-Score", 0) or 0) if best_row else 0.0,
            "results_path": str(CLASSIFIER_OUTPUT_PATH),
        }

    def _summarize_selected_model(self, classifier_summary: dict) -> dict:
        if not classifier_summary:
            return {}
        dataset_name = classifier_summary.get("best_dataset", "")
        classifier_name = classifier_summary.get("best_classifier", "")
        dataset_slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
        classifier_slug = (
            classifier_name.lower()
            .replace(" ", "_")
            .replace("_(mlp)", "")
            .replace("(", "")
            .replace(")", "")
        )
        model_path = MODELS_DIR / f"{classifier_slug}_{dataset_slug}.pkl"
        return {
            "selected_dataset": dataset_name,
            "selected_classifier": classifier_name,
            "selection_metric": "best_f1_score",
            "selected_model_path": str(model_path) if model_path.exists() else "",
            "selection_available": model_path.exists(),
            "selection_note": (
                "All datasets are trained; this selects the best-performing saved model after evaluation."
            ),
        }

    def _summarize_robustness_results(self) -> dict:
        if not ROBUSTNESS_OUTPUT_PATH.exists():
            return {}
        best_row = None
        with ROBUSTNESS_OUTPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if best_row is None or float(row.get("FGSM Accuracy", 0) or 0) > float(best_row.get("FGSM Accuracy", 0) or 0):
                    best_row = row
        return {
            "best_dataset": best_row.get("Dataset", "") if best_row else "",
            "fgsm_accuracy": float(best_row.get("FGSM Accuracy", 0) or 0) if best_row else 0.0,
            "pgd_accuracy": float(best_row.get("PGD Accuracy", 0) or 0) if best_row else 0.0,
            "robustness_score": float(best_row.get("Robustness Score", 0) or 0) if best_row else 0.0,
            "results_path": str(ROBUSTNESS_OUTPUT_PATH),
        }

    def _refresh_intel(self, enabled: bool) -> list[dict]:
        if not enabled:
            self._log("Skipping fraud-intel refresh")
            return []
        self._log("Refreshing latest fraud and cybercrime signals")
        try:
            signals = self.intel_tool.run()
        except Exception as exc:
            signals = [
                {
                    "title": "Fraud intel refresh failed",
                    "link": "",
                    "published_at": "",
                    "error": str(exc),
                }
            ]
        self.memory.set_latest_fraud_signals(signals)
        self._record_tool_call(
            self.intel_tool.spec.name,
            {"enabled": enabled, "max_items": 8},
            signals,
        )
        self._log(f"Stored {len(signals)} fraud signal entries in memory")
        return signals

    def _run_stage(self, stage_name: str, script_path: Path, expected_output: Path | None = None) -> dict:
        self._log(f"Starting stage: {stage_name} ({script_path.name})")
        result = self.script_tool.run(script_path=script_path, cwd=FIN_FRAUD_ROOT, stream=True)
        run_record = {
            "stage": stage_name,
            "script": result["script"],
            "returncode": result["returncode"],
            "stdout_tail": result["stdout"][-4000:],
            "stderr_tail": result["stderr"][-4000:],
        }
        self.memory.record_run(run_record)
        if result["returncode"] != 0:
            raise RuntimeError(
                f"Stage '{stage_name}' failed.\nSTDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}"
            )
        if expected_output is not None:
            metadata = {"exists": expected_output.exists()}
            if expected_output.exists():
                metadata["size_bytes"] = expected_output.stat().st_size
            self.memory.upsert_artifact(stage_name, str(expected_output), metadata)
            self._record_tool_call(
                self.script_tool.spec.name,
                {"stage_name": stage_name, "script_path": str(script_path)},
                {
                    "returncode": result["returncode"],
                    "expected_output": str(expected_output),
                    "output_exists": expected_output.exists(),
                },
            )
            self._log(f"Completed stage: {stage_name} -> {expected_output}")
        else:
            self._record_tool_call(
                self.script_tool.spec.name,
                {"stage_name": stage_name, "script_path": str(script_path)},
                {"returncode": result["returncode"]},
            )
            self._log(f"Completed stage: {stage_name}")
        return result

    def _execute_pipeline_steps(self) -> None:
        for step in self.pipeline_steps:
            self._log(f"Pipeline step: {step.description}")
            self._run_stage(step.name, step.script_path, step.expected_output)

    def _run_llm_assessment(self, dataset_snapshot: dict, fraud_signals: list[dict]) -> dict:
        self._log("Running Ollama assessment with qwen3.5:0.8b")
        assessment = self.ollama_tool.run(
            dataset_snapshot=dataset_snapshot,
            fraud_signals=fraud_signals,
        )
        payload = {
            "stage": "ollama_assessment",
            "dataset_snapshot": dataset_snapshot,
            "fraud_signal_count": len(fraud_signals),
            **assessment,
        }
        self.memory.add_llm_assessment(payload)
        self._record_tool_call(
            self.ollama_tool.spec.name,
            {
                "model": self.ollama_tool.model,
                "fraud_signal_count": len(fraud_signals),
            },
            assessment,
        )
        self._log(f"Ollama assessment status: {assessment['status']}")
        self._log("Ollama assessment output:")
        print(json.dumps(assessment, indent=2, ensure_ascii=True), flush=True)
        return payload

    def _copy_file(self, source: Path, destination: Path) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def _copy_file_if_exists(self, source: Path, destination: Path) -> Path | None:
        if not source.exists():
            return None
        return self._copy_file(source, destination)

    def _copy_tree(self, source: Path, destination: Path) -> Path:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return destination

    def _collect_agent_outputs(self) -> dict:
        self._log("Copying final artifacts into agent/output")
        prepared_dataset_path = self._copy_file(ORIGINAL_DATASET_PATH, OUTPUT_PREPARED_DATASET_PATH)
        ctgan_output_path = self._copy_file(CTGAN_OUTPUT_PATH, OUTPUT_CTGAN_PATH)
        ctgan_synthetic_novel_path = self._copy_file_if_exists(
            CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH, OUTPUT_CTGAN_SYNTHETIC_NOVEL_PATH
        )
        adv_ctgan_output_path = self._copy_file(ADV_CTGAN_OUTPUT_PATH, OUTPUT_ADV_CTGAN_PATH)
        adv_ctgan_synthetic_novel_path = self._copy_file_if_exists(
            ADV_CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH, OUTPUT_ADV_CTGAN_SYNTHETIC_NOVEL_PATH
        )
        classifier_output_path = self._copy_file(CLASSIFIER_OUTPUT_PATH, OUTPUT_CLASSIFIER_RESULTS_PATH)
        robustness_output_path = self._copy_file(ROBUSTNESS_OUTPUT_PATH, OUTPUT_ROBUSTNESS_RESULTS_PATH)
        models_dir = self._copy_tree(MODELS_DIR, OUTPUT_MODELS_DIR)
        memory_path = self._copy_file(self.memory.path, OUTPUT_MEMORY_PATH)
        outputs = {
            "prepared_dataset_path": str(prepared_dataset_path),
            "ctgan_output_path": str(ctgan_output_path),
            "ctgan_synthetic_novel_path": str(ctgan_synthetic_novel_path) if ctgan_synthetic_novel_path else "",
            "adv_ctgan_output_path": str(adv_ctgan_output_path),
            "adv_ctgan_synthetic_novel_path": str(adv_ctgan_synthetic_novel_path) if adv_ctgan_synthetic_novel_path else "",
            "classifier_output_path": str(classifier_output_path),
            "robustness_output_path": str(robustness_output_path),
            "models_dir": str(models_dir),
            "memory_path": str(memory_path),
        }
        self.memory.update_pipeline_state(
            "augmentation",
            {
                "ctgan_balanced_output": outputs["ctgan_output_path"],
                "ctgan_synthetic_novel_output": outputs["ctgan_synthetic_novel_path"],
                "adv_ctgan_balanced_output": outputs["adv_ctgan_output_path"],
                "adv_ctgan_synthetic_novel_output": outputs["adv_ctgan_synthetic_novel_path"],
                "novel_fraud_mode_used": bool(ctgan_synthetic_novel_path or adv_ctgan_synthetic_novel_path),
            },
        )
        return outputs

    def _write_output_summary(self, summary: dict) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        self.memory.upsert_artifact("run_summary", str(RUN_SUMMARY_PATH), {"exists": True})
        self._log(f"Wrote run summary to {RUN_SUMMARY_PATH}")

    def _write_tool_call_outputs(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with TOOL_CALL_OUTPUTS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(self.tool_call_outputs, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        self.memory.upsert_artifact("tool_call_outputs", str(TOOL_CALL_OUTPUTS_PATH), {"exists": True})
        self._log(f"Wrote tool call outputs to {TOOL_CALL_OUTPUTS_PATH}")

    def run(self, refresh_intel: bool = True, scrape_max_age_days: int = 3, label_max_age_days: int = 3) -> dict:
        self._log("Starting fraud pipeline agent")
        self._register_tools()
        collection_summary = self._refresh_reddit_collection(scrape_max_age_days, label_max_age_days)
        dataset_snapshot = self._prepare_dataset(collection_summary)
        signals = self._refresh_intel(enabled=refresh_intel)
        llm_assessment = self._run_llm_assessment(dataset_snapshot, signals)
        self._execute_pipeline_steps()
        self.memory.upsert_artifact("models_dir", str(MODELS_DIR), {"exists": MODELS_DIR.exists()})
        agent_outputs = self._collect_agent_outputs()

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_snapshot,
            "collection_flow": collection_summary,
            "fraud_signal_count": len(signals),
            "ollama_model": llm_assessment["model"],
            "ollama_status": llm_assessment["status"],
            **agent_outputs,
        }
        classifier_summary = self._summarize_classifier_results()
        robustness_summary = self._summarize_robustness_results()
        selected_model_summary = self._summarize_selected_model(classifier_summary)
        pipeline_run_summary = {
            "collection": self.current_run_context.get("collection_flow", {}).get("collection", {}),
            "labeling": self.current_run_context.get("collection_flow", {}).get("labeling", {}),
            "main_dataset_sync": self.current_run_context.get("collection_flow", {}).get("main_dataset_sync", {}),
            "dataset": self.current_run_context.get("dataset", {}),
            "augmentation": {
                "ctgan_synthetic_novel_path": agent_outputs.get("ctgan_synthetic_novel_path", ""),
                "adv_ctgan_synthetic_novel_path": agent_outputs.get("adv_ctgan_synthetic_novel_path", ""),
                "novel_fraud_mode_used": bool(
                    agent_outputs.get("ctgan_synthetic_novel_path") or agent_outputs.get("adv_ctgan_synthetic_novel_path")
                ),
            },
            "classifier": classifier_summary,
            "robustness": robustness_summary,
            "selected_model": selected_model_summary,
        }
        self.memory.update_pipeline_state("classifier", classifier_summary)
        self.memory.update_pipeline_state("robustness", robustness_summary)
        self.memory.update_pipeline_state("selected_model", selected_model_summary)
        self.memory.add_pipeline_run_summary(pipeline_run_summary)
        self._write_tool_call_outputs()
        self._write_output_summary(summary)
        self._log("Pipeline completed")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic controller for the Fin-Fraud AI pipeline.")
    parser.add_argument(
        "--skip-intel",
        action="store_true",
        help="Skip refreshing the latest fraud attack/scam signal memory before the training loop.",
    )
    parser.add_argument(
        "--scrape-max-age-days",
        type=int,
        default=3,
        help="Reuse scraped Reddit posts/comments if they are this many days old or newer.",
    )
    parser.add_argument(
        "--label-max-age-days",
        type=int,
        default=3,
        help="Reuse scraped label outputs if they are this many days old or newer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = FraudPipelineAgent()
    summary = agent.run(
        refresh_intel=not args.skip_intel,
        scrape_max_age_days=args.scrape_max_age_days,
        label_max_age_days=args.label_max_age_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
