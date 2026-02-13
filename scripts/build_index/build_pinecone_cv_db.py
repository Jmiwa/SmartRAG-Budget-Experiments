#!/usr/bin/env python3
"""
Build a 5-fold CV Pinecone database from HIE_DB JSON files.

Embedding text uses:
  - consultation_text
  - financial_data (formatted as human-readable text)

Metadata (payload) stores:
  - case_id
  - expert_advice_text
  - financial_data (JSON string)
  - expert_reduction (JSON string)
  - retrieval_attributes (JSON string)
  - retrieval_* fields for future filtering
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from langchain_openai import OpenAIEmbeddings  # type: ignore

# Avoid clashing with the installed `pinecone` package
_CURRENT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(_CURRENT_DIR))
except ValueError:
    pass
sys.path.append(str(_CURRENT_DIR))

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None


try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except ImportError:
    print("Error: pinecone-client is not installed.")
    print("Install with: pip install pinecone-client")
    sys.exit(1)


INDEX_NAME_DEFAULT = "your-pinecone-index-name"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


@dataclass(frozen=True)
class CaseRecord:
    case_id: int
    source_file: str
    embedding_text: str
    metadata: Dict[str, object]


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"{var_name} environment variable is not set.")
    return value


def iter_json_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path


def format_financial_data(financial_data: dict) -> str:
    def fmt_items(title: str, items: list) -> List[str]:
        if not items:
            return []
        lines = [f"[{title}]"]
        for item in items:
            label = item.get("label", "")
            amount = item.get("amount_yen")
            if amount is None:
                lines.append(f"- {label}")
            else:
                lines.append(f"- {label}: {amount} yen")
        return lines + [""]

    lines: List[str] = []
    lines += fmt_items("income", financial_data.get("income", []))
    lines += fmt_items("expenses", financial_data.get("expenses", []))
    lines += fmt_items("savings", financial_data.get("savings", []))
    lines += fmt_items("investments", financial_data.get("investments", []))

    totals = financial_data.get("totals", {})
    if totals:
        lines.append("[totals]")
        key_map = {
            "income_monthly_yen": "income_monthly_yen",
            "expense_monthly_yen": "expense_monthly_yen",
            "savings_total_yen": "savings_total_yen",
            "investments_total_yen": "investments_total_yen",
            "savings_monthly_yen": "savings_monthly_yen",
            "investments_monthly_yen": "investments_monthly_yen",
        }
        for key, label in key_map.items():
            if key in totals:
                lines.append(f"- {label}: {totals.get(key)}")
        lines.append("")

    return "\n".join(lines).strip()


def build_embedding_text(consultation_text: str, financial_data: dict) -> str:
    formatted_financial = format_financial_data(financial_data)
    return (
        "[consultation_text]\n"
        f"{consultation_text.strip()}\n\n"
        "[financial_data]\n"
        f"{formatted_financial}"
    ).strip()


def prune_none(metadata: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in metadata.items() if v is not None}


def build_metadata(payload: dict, source_file: str) -> Dict[str, object]:
    retrieval = payload.get("retrieval_attributes") or {}
    metadata = {
        "case_id": payload.get("case_id"),
        "source_file": source_file,
        "expert_advice_text": payload.get("expert_advice_text"),
        "financial_data_json": json.dumps(
            payload.get("financial_data"), ensure_ascii=False
        ),
        "expert_reduction_json": json.dumps(
            payload.get("expert_reduction"), ensure_ascii=False
        ),
        "retrieval_attributes_json": json.dumps(retrieval, ensure_ascii=False),
        # Flattened fields for future metadata filtering
        "retrieval_expense_categories": retrieval.get("expense_categories"),
        "retrieval_income_monthly_yen": retrieval.get("income_monthly_yen"),
        "retrieval_household_size": retrieval.get("household_size"),
    }
    return prune_none(metadata)


def load_case_records(data_dir: Path, limit: Optional[int]) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    all_files = list(iter_json_files(data_dir))
    if limit is not None:
        all_files = all_files[:limit]

    for json_path in all_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        case_id = payload.get("case_id")
        if case_id is None:
            print(f"Warning: {json_path.name} has no case_id. Skipping.")
            continue

        consultation_text = payload.get("consultation_text")
        if not consultation_text:
            print(f"Warning: {json_path.name} has no consultation_text. Skipping.")
            continue

        financial_data = payload.get("financial_data") or {}
        embedding_text = build_embedding_text(consultation_text, financial_data)

        metadata = build_metadata(payload, json_path.name)

        records.append(
            CaseRecord(
                case_id=int(case_id),
                source_file=json_path.name,
                embedding_text=embedding_text,
                metadata=metadata,
            )
        )

    return records


def generate_folds(case_ids: List[int], k: int, seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    ids = case_ids[:]
    rng.shuffle(ids)

    folds: Dict[str, List[int]] = {}
    n = len(ids)
    base = n // k
    remainder = n % k
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        fold_ids = ids[start : start + size]
        folds[f"fold_{i+1}"] = fold_ids
        start += size
    return folds


def load_or_create_folds(
    folds_path: Path, case_ids: List[int], seed: int
) -> Dict[str, List[int]]:
    if folds_path.exists():
        with folds_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    folds = generate_folds(case_ids, k=5, seed=seed)
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with folds_path.open("w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return folds


def ensure_index(pinecone_client: Pinecone, index_name: str) -> None:
    existing_indexes = [idx.get("name") for idx in pinecone_client.list_indexes()]
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists.")
        return

    print(f"Creating index '{index_name}'...")
    pinecone_client.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready...")
    while not pinecone_client.describe_index(index_name).status["ready"]:
        time.sleep(5)
    print("Index created and ready.")


def compute_embeddings(
    records: List[CaseRecord],
    embeddings: OpenAIEmbeddings,
    batch_size: int,
    sleep_time: int,
) -> Dict[int, List[float]]:
    vectors: Dict[int, List[float]] = {}
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        texts = [rec.embedding_text for rec in batch]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"Embedding batch {batch_num}/{total_batches} ({len(texts)} texts)...")
        batch_vectors = embeddings.embed_documents(texts)
        for rec, vec in zip(batch, batch_vectors):
            vectors[rec.case_id] = vec
        if i + batch_size < total:
            time.sleep(sleep_time)
    return vectors


def upsert_fold(
    index,
    namespace: str,
    records: Dict[int, CaseRecord],
    vectors: Dict[int, List[float]],
    training_ids: List[int],
    batch_size: int,
    sleep_time: int,
) -> None:
    total = len(training_ids)
    for i in range(0, total, batch_size):
        batch_ids = training_ids[i : i + batch_size]
        upsert_vectors = []
        for case_id in batch_ids:
            rec = records[case_id]
            vec = vectors[case_id]
            upsert_vectors.append((str(case_id), vec, rec.metadata))
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(
            f"  Upserting batch {batch_num}/{total_batches} "
            f"({len(upsert_vectors)} vectors) into {namespace}..."
        )
        index.upsert(vectors=upsert_vectors, namespace=namespace)
        if i + batch_size < total:
            time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 5-fold CV Pinecone DB from HIE_DB JSON."
    )
    parser.add_argument("--data-dir", default="HIE_DB")
    parser.add_argument("--index-name", default=INDEX_NAME_DEFAULT)
    parser.add_argument("--folds-file", default="cv_folds/hie_db_folds.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep-time", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / args.data_dir
    folds_path = base_dir / args.folds_file

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return

    if "_" in args.index_name:
        sanitized = args.index_name.replace("_", "-")
        print(
            f"Index name '{args.index_name}' is invalid. Using '{sanitized}' instead."
        )
        args.index_name = sanitized

    if load_dotenv is not None:
        load_dotenv("token.env")
        load_dotenv()

    require_env("OPENAI_API_KEY")
    pinecone_api_key = require_env("PINECONE_API_KEY")

    records = load_case_records(data_dir, args.limit)
    if not records:
        print("No records found to process.")
        return

    print(f"Loaded {len(records)} records from {data_dir}.")

    case_ids = [rec.case_id for rec in records]
    folds = load_or_create_folds(folds_path, case_ids, seed=args.seed)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print("Computing embeddings...")
    vectors = compute_embeddings(records, embeddings, args.batch_size, args.sleep_time)

    pinecone_client = Pinecone(api_key=pinecone_api_key)
    ensure_index(pinecone_client, args.index_name)
    index = pinecone_client.Index(args.index_name)

    record_map = {rec.case_id: rec for rec in records}
    all_ids_set = set(case_ids)

    for fold_name, fold_ids in folds.items():
        training_ids = sorted(all_ids_set - set(fold_ids))
        print(f"Upserting namespace '{fold_name}' with {len(training_ids)} vectors...")
        upsert_fold(
            index=index,
            namespace=fold_name,
            records=record_map,
            vectors=vectors,
            training_ids=training_ids,
            batch_size=args.batch_size,
            sleep_time=args.sleep_time,
        )

    print("Done.")


if __name__ == "__main__":
    main()
