from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: list[str]
    outputs: list[str]

    @property
    def signature(self) -> str:
        payload = "|".join([self.name, self.description, ",".join(self.inputs), ",".join(self.outputs)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "signature": self.signature,
        }


class FraudIntelTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="latest_fraud_intel",
            description="Pulls recent fraud and scam attack signals from current news/RSS search feeds.",
            inputs=["search queries", "max_items"],
            outputs=["dated fraud signal list with titles and source links"],
        )
        self.queries = [
            "financial fraud scam phishing bank impersonation",
            "gift card scam fraud alert",
            "crypto investment scam fraud alert",
        ]

    def _rss_url(self, query: str) -> str:
        return (
            "https://news.google.com/rss/search?q="
            f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

    def _parse_feed(self, query: str) -> list[dict[str, Any]]:
        request = Request(
            self._rss_url(query),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=20) as response:
            content = response.read()

        root = ET.fromstring(content)
        signals = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            if not title or not link:
                continue
            published_at = pub_date
            if pub_date:
                try:
                    published_at = parsedate_to_datetime(pub_date).isoformat()
                except (TypeError, ValueError, IndexError):
                    published_at = pub_date
            signals.append(
                {
                    "query": query,
                    "title": title,
                    "link": link,
                    "published_at": published_at,
                }
            )
        return signals

    def run(self, max_items: int = 8) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        seen_links: set[str] = set()
        for query in self.queries:
            for signal in self._parse_feed(query):
                if signal["link"] in seen_links:
                    continue
                seen_links.add(signal["link"])
                signals.append(signal)

        signals.sort(key=lambda item: item.get("published_at", ""), reverse=True)
        return signals[:max_items]


class PythonScriptTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="python_pipeline_runner",
            description="Runs one Python stage in the Fin-Fraud AI pipeline and captures stdout/stderr.",
            inputs=["script_path"],
            outputs=["returncode", "stdout", "stderr"],
        )

    def run(self, script_path: Path, cwd: Path, stream: bool = True) -> dict[str, Any]:
        if not stream:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
            return {
                "script": str(script_path),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout_chunks: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stdout_chunks.append(line)
        process.wait()
        return {
            "script": str(script_path),
            "returncode": process.returncode,
            "stdout": "".join(stdout_chunks),
            "stderr": "",
        }


class OllamaAdvisorTool:
    def __init__(self, model: str = "qwen3:0.6b") -> None:
        self.model = model
        self.spec = ToolSpec(
            name="ollama_classifier_advisor",
            description="Uses a local Ollama model to assess new fraud patterns and suggest classifier updates.",
            inputs=["dataset snapshot", "latest fraud signals"],
            outputs=["llm assessment with retraining and feature recommendations"],
        )

    def run(self, dataset_snapshot: dict[str, Any], fraud_signals: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from ollama import chat
        except ImportError as exc:
            return {
                "model": self.model,
                "status": "unavailable",
                "error": f"ollama package missing: {exc}",
                "assessment": "",
            }

        signals_block = fraud_signals[:5]
        prompt = (
            "/think and explain the steps\n"
            "You are assisting a fraud-detection agent.\n"
            "Given the current training dataset summary and recent fraud/cybercrime signals, "
            "decide whether the classifier pipeline should be retrained and what should change.\n\n"
            f"Dataset snapshot:\n{dataset_snapshot}\n\n"
            f"Recent fraud signals:\n{signals_block}\n\n"
            "Return concise JSON with keys: risk_level, retrain_now, new_attack_types, "
            "feature_updates, classifier_updates, data_collection_updates, summary."
        )

        try:
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            return {
                "model": self.model,
                "status": "error",
                "error": str(exc),
                "assessment": "",
            }

        message = getattr(response, "message", None)
        content = getattr(message, "content", "")
        return {
            "model": self.model,
            "status": "ok",
            "assessment": content,
        }
