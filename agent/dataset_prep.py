from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from agent.config import (
    MAIN_CSV_PATH,
    NOVEL_SCAM_REPORT_PATH,
    NOVEL_SCAM_SEEDS_PATH,
    ORIGINAL_DATASET_DIR,
    ORIGINAL_DATASET_PATH,
    SCRAPED_LABELED_CSV_PATH,
    SCRAPED_POSTS_PATH,
)


TRAINING_COLUMNS = [
    "title",
    "body",
    "is_fraud",
    "amount_numeric",
    "post_metadata.subreddit",
    "post_metadata.num_comments",
    "annotation.fraud_confidence",
    "annotation.fraud_type",
    "annotation.fraud_labels.transaction_upi_fraud",
    "annotation.fraud_labels.transaction_card_fraud",
    "annotation.fraud_labels.transaction_bank_transfer",
    "annotation.fraud_labels.transaction_nondelivery",
    "annotation.fraud_labels.transaction_fake_seller",
    "annotation.fraud_labels.commerce_nondelivery",
    "annotation.fraud_labels.commerce_fake_seller",
    "annotation.fraud_labels.credential_phishing",
    "annotation.fraud_labels.social_authority_scam",
    "annotation.fraud_labels.social_urgency_scam",
    "annotation.fraud_labels.meta_victim_story",
    "annotation.fraud_labels.meta_fraud_question",
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.victim_action",
    "annotation.key_features.request_type",
    "annotation.key_features.impersonated_entity",
    "annotation.key_features.currency",
    "annotation.key_features.urgency_level",
    "annotation.psychological_tactics.urgency",
    "annotation.psychological_tactics.fear",
    "annotation.psychological_tactics.authority",
    "annotation.psychological_tactics.reward",
    "annotation.community_signals.num_comments",
    "annotation.community_signals.scam_confirmations",
    "annotation.community_signals.not_scam_claims",
    "annotation.community_signals.advice_requests",
    "annotation.label_quality.confidence_bucket",
    "annotation.label_quality.usable_for_training",
    "annotation.gan_quality.suitable_for_gan",
    "annotation.gan_quality.quality_score",
    "post_metadata.body_length",
    "post_metadata.body_language",
    "post_metadata.scam_confirmations",
    "post_metadata.not_scam_claims",
    "post_metadata.advice_requests",
]

FRAUD_SIGNATURE_COLUMNS = [
    "annotation.fraud_type",
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.request_type",
    "annotation.key_features.impersonated_entity",
]


@dataclass
class PreparedDataset:
    output_path: Path
    row_count: int
    class_counts: dict[str, int]
    columns: list[str]
    new_scam_row_count: int
    new_scam_seed_path: Path
    new_scam_report_path: Path
    base_row_count: int
    labeled_scraped_row_count: int
    labeled_scraped_class_counts: dict[str, int]
    labeled_scraped_usable_count: int
    new_scam_signatures: list[dict]

    def to_memory_dict(self) -> dict:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        payload["new_scam_seed_path"] = str(self.new_scam_seed_path)
        payload["new_scam_report_path"] = str(self.new_scam_report_path)
        return payload


def _first_available(df: pd.DataFrame, candidates: Iterable[str], default=None):
    for column in candidates:
        if column in df.columns:
            series = df[column]
            if series.notna().any():
                return series
    return default


def _to_boolish(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        if lowered.isin(mapping.keys()).any():
            return lowered.map(mapping).where(~series.isna(), other=pd.NA)
    return series


def _extract_amount(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _normalize_base_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["title"] = _first_available(
        df,
        ["title", "post_metadata.title"],
        default=pd.Series([pd.NA] * len(df)),
    )
    df["body"] = _first_available(
        df,
        ["body", "post_metadata.body"],
        default=pd.Series([pd.NA] * len(df)),
    )
    df["is_fraud"] = pd.to_numeric(
        _first_available(df, ["is_fraud", "annotation.is_fraud"]),
        errors="coerce",
    )
    df["amount_numeric"] = _extract_amount(
        _first_available(
            df,
            [
                "amount_numeric",
                "annotation.key_features.amount_mentioned",
                "key_features.amount_mentioned",
                "annotation.amount_mentioned",
            ],
            default=pd.Series([pd.NA] * len(df)),
        )
    )
    return df


def _build_known_signatures(base_df: pd.DataFrame) -> set[tuple[str, ...]]:
    known: set[tuple[str, ...]] = set()
    fraud_rows = base_df[base_df["is_fraud"] == 1].copy()
    for _, row in fraud_rows.iterrows():
        signature = []
        for column in FRAUD_SIGNATURE_COLUMNS:
            value = row.get(column, "unknown")
            if pd.isna(value):
                value = "unknown"
            signature.append(str(value).strip().lower() or "unknown")
        known.add(tuple(signature))
    return known


def _build_scraped_candidates(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if not SCRAPED_LABELED_CSV_PATH.exists():
        return pd.DataFrame(columns=base_df.columns), []

    labeled_df = _normalize_base_annotations(pd.read_csv(SCRAPED_LABELED_CSV_PATH))
    if labeled_df.empty:
        return pd.DataFrame(columns=base_df.columns), []

    known_post_ids = {
        str(value)
        for value in _first_available(
            base_df,
            ["post_metadata.post_id", "post_id"],
            default=pd.Series(dtype=str),
        ).dropna()
    }
    known_signatures = _build_known_signatures(base_df)

    candidate_rows: list[dict] = []
    report_rows: list[dict] = []

    for _, post in labeled_df.iterrows():
        post_id = str(post.get("post_metadata.post_id", "")).strip()
        if not post_id or post_id in known_post_ids:
            continue

        is_fraud = pd.to_numeric(post.get("annotation.is_fraud"), errors="coerce")
        if pd.isna(is_fraud) or int(is_fraud) != 1:
            continue
        confidence_bucket = str(post.get("annotation.label_quality.confidence_bucket", "") or "").strip().lower()
        usable = post.get("annotation.label_quality.usable_for_training", "")
        if confidence_bucket != "high" or not _is_truthy(usable):
            continue

        fraud_type = str(post.get("annotation.fraud_type", "unknown") or "unknown")
        payment_method = str(post.get("annotation.key_features.payment_method", "unknown") or "unknown")
        fraud_channel = str(post.get("annotation.key_features.fraud_channel", "unknown") or "unknown")
        request_type = str(post.get("annotation.key_features.request_type", "unknown") or "unknown")
        impersonated_entity = str(post.get("annotation.key_features.impersonated_entity", "unknown") or "unknown")

        signature = (
            fraud_type,
            payment_method,
            fraud_channel,
            request_type,
            impersonated_entity,
        )
        normalized_signature = tuple(value.strip().lower() or "unknown" for value in signature)
        is_new_signature = normalized_signature not in known_signatures
        if not is_new_signature:
            continue

        row = {column: post[column] if column in post.index else pd.NA for column in base_df.columns}
        candidate_rows.append(row)
        report_rows.append(
            {
                "post_id": post_id,
                "subreddit": post.get("post_metadata.subreddit", ""),
                "is_fraud": 1,
                "fraud_type": fraud_type,
                "payment_method": payment_method,
                "fraud_channel": fraud_channel,
                "request_type": request_type,
                "impersonated_entity": impersonated_entity,
                "fraud_confidence": post.get("annotation.fraud_confidence", ""),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=base_df.columns)
    return candidate_df, report_rows


def _write_new_scam_artifacts(new_scam_df: pd.DataFrame, report_rows: list[dict]) -> None:
    ORIGINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if new_scam_df.empty:
        pd.DataFrame(columns=TRAINING_COLUMNS).to_csv(NOVEL_SCAM_SEEDS_PATH, index=False)
        report = {
            "new_scam_row_count": 0,
            "new_scam_signatures": [],
        }
    else:
        available_columns = [column for column in TRAINING_COLUMNS if column in new_scam_df.columns]
        ordered_columns = [
            "title",
            "body",
            "is_fraud",
            "amount_numeric",
            *[column for column in available_columns if column not in {"title", "body", "is_fraud", "amount_numeric"}],
        ]
        seeds = new_scam_df[ordered_columns].copy()
        seeds.to_csv(NOVEL_SCAM_SEEDS_PATH, index=False)
        report = {
            "new_scam_row_count": int(len(new_scam_df)),
            "new_scam_signatures": report_rows,
        }

    with NOVEL_SCAM_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def sync_scraped_annotations_into_main_csv(
    main_path: Path = MAIN_CSV_PATH,
    scraped_path: Path = SCRAPED_LABELED_CSV_PATH,
    posts_path: Path = SCRAPED_POSTS_PATH,
    min_created_utc_exclusive: float | None = None,
) -> dict[str, int | float]:
    if not main_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {main_path}")

    if not scraped_path.exists():
        return {
            "base_row_count": 0,
            "scraped_row_count": 0,
            "new_rows_added": 0,
            "final_row_count": 0,
            "latest_added_created_utc": 0.0,
        }

    base_df = pd.read_csv(main_path)
    scraped_df = pd.read_csv(scraped_path)
    base_row_count = int(len(base_df))
    scraped_row_count = int(len(scraped_df))

    if scraped_df.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
        }

    base_ids = (
        base_df.get("post_metadata.post_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    scraped_ids = (
        scraped_df.get("post_metadata.post_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    known_ids = set(base_ids[base_ids != ""])
    new_rows = scraped_df.loc[(scraped_ids != "") & (~scraped_ids.isin(known_ids))].copy()
    latest_added_created_utc = 0.0

    if not new_rows.empty and posts_path.exists():
        posts_df = pd.read_csv(posts_path, usecols=["post_id", "created_utc"])
        posts_df["post_id"] = posts_df["post_id"].fillna("").astype(str).str.strip()
        posts_df["created_utc"] = pd.to_numeric(posts_df["created_utc"], errors="coerce")
        created_lookup = posts_df.dropna(subset=["created_utc"]).drop_duplicates(subset=["post_id"], keep="last")
        created_map = created_lookup.set_index("post_id")["created_utc"]
        new_rows["post_metadata.created_utc"] = new_rows["post_metadata.post_id"].map(created_map)

        if min_created_utc_exclusive is not None:
            new_rows = new_rows.loc[
                pd.to_numeric(new_rows["post_metadata.created_utc"], errors="coerce") > float(min_created_utc_exclusive)
            ].copy()

    if new_rows.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
        }

    latest_added_created_utc = float(
        pd.to_numeric(new_rows.get("post_metadata.created_utc"), errors="coerce").dropna().max()
        if "post_metadata.created_utc" in new_rows.columns
        else 0.0
    )

    merged_columns = list(base_df.columns)
    for column in new_rows.columns:
        if column not in merged_columns:
            merged_columns.append(column)

    merged_df = pd.concat(
        [
            base_df.reindex(columns=merged_columns),
            new_rows.reindex(columns=merged_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    merged_df.to_csv(main_path, index=False)
    return {
        "base_row_count": base_row_count,
        "scraped_row_count": scraped_row_count,
        "new_rows_added": int(len(new_rows)),
        "final_row_count": int(len(merged_df)),
        "latest_added_created_utc": latest_added_created_utc,
    }


def prepare_main_dataset(
    input_path: Path = MAIN_CSV_PATH,
    output_path: Path = ORIGINAL_DATASET_PATH,
) -> PreparedDataset:
    base_df = _normalize_base_annotations(pd.read_csv(input_path))
    labeled_scraped_df = (
        _normalize_base_annotations(pd.read_csv(SCRAPED_LABELED_CSV_PATH))
        if SCRAPED_LABELED_CSV_PATH.exists()
        else pd.DataFrame()
    )
    new_scam_df, report_rows = _build_scraped_candidates(base_df)
    combined_df = pd.concat([base_df, new_scam_df], ignore_index=True, sort=False)

    available_columns = [column for column in TRAINING_COLUMNS if column in combined_df.columns]
    ordered_columns = [
        "title",
        "body",
        "is_fraud",
        "amount_numeric",
        *[column for column in available_columns if column not in {"title", "body", "is_fraud", "amount_numeric"}],
    ]
    prepared = combined_df[ordered_columns].copy()

    for column in prepared.columns:
        prepared[column] = _to_boolish(prepared[column])

    prepared = prepared.dropna(subset=["is_fraud"])
    prepared["is_fraud"] = prepared["is_fraud"].astype(int)

    ORIGINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    _write_new_scam_artifacts(new_scam_df, report_rows)

    class_counts = {
        str(label): int(count)
        for label, count in prepared["is_fraud"].value_counts(dropna=False).sort_index().items()
    }
    labeled_scraped_class_counts: dict[str, int] = {}
    labeled_scraped_usable_count = 0
    if not labeled_scraped_df.empty and "is_fraud" in labeled_scraped_df.columns:
        labeled_scraped_clean = labeled_scraped_df.dropna(subset=["is_fraud"]).copy()
        labeled_scraped_clean["is_fraud"] = labeled_scraped_clean["is_fraud"].astype(int)
        labeled_scraped_class_counts = {
            str(label): int(count)
            for label, count in labeled_scraped_clean["is_fraud"].value_counts(dropna=False).sort_index().items()
        }
        if "annotation.label_quality.usable_for_training" in labeled_scraped_clean.columns:
            labeled_scraped_usable_count = int(
                labeled_scraped_clean["annotation.label_quality.usable_for_training"].map(_is_truthy).sum()
            )
    return PreparedDataset(
        output_path=output_path,
        row_count=int(len(prepared)),
        class_counts=class_counts,
        columns=list(prepared.columns),
        new_scam_row_count=int(len(new_scam_df)),
        new_scam_seed_path=NOVEL_SCAM_SEEDS_PATH,
        new_scam_report_path=NOVEL_SCAM_REPORT_PATH,
        base_row_count=int(len(base_df)),
        labeled_scraped_row_count=int(len(labeled_scraped_df)),
        labeled_scraped_class_counts=labeled_scraped_class_counts,
        labeled_scraped_usable_count=labeled_scraped_usable_count,
        new_scam_signatures=report_rows,
    )
