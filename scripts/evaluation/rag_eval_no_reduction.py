"""
RAG evaluation script (no-reduction-index version)
- Retrieve similar cases from Pinecone and generate reduction-rate JSON with Gemini.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv  # type: ignore
import google.genai as genai

from langchain_openai import OpenAIEmbeddings  # type: ignore
from pinecone import Pinecone

# ====== Runtime settings (edit only this section) ======
TOTAL_LIMIT = None  # Set to None to process all cases
START_FOLD = "fold_1"  # Start processing from this fold in order
MODEL_NAME = "gemini-3-flash-preview"
INDEX_NAME = "your-pinecone-index-name"
TOP_K = 3
SKIP_EXISTING = True  # Skip IDs that already have output files
RETRY_COUNT = 2  # Retry count when JSON parsing fails

# ====== Path settings ======
BASE_DIR = Path(__file__).resolve().parent
MASTER_PROMPT_PATH = BASE_DIR / "prompts" / "master_prompt_rag_eval_no_reduction.txt"

USER_INPUT_DIR = BASE_DIR / "user_input"
FOLDS_PATH = BASE_DIR / "cv_folds" / "hie_db_folds.json"
RESULTS_BASE_DIR = BASE_DIR / "results" / "rag_no_reduction"
RETRIEVAL_BASE_DIR = RESULTS_BASE_DIR / "retrieval"


def load_master_prompt() -> str:
    """Load the master prompt (assumes no runtime editing)."""
    with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_folds() -> Dict[str, List[int]]:
    """Load fold definitions for cross validation."""
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_json_field(value: Any) -> Any:
    """If value is a JSON string, parse it to dict/list; otherwise return as-is."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def build_prompt(
    master_prompt: str,
    consultation_text: str,
    financial_data: Any,
    retrieved_cases: List[Dict[str, Any]],
) -> str:
    """Build the prompt sent to the LLM."""
    financial_json = json.dumps(financial_data, ensure_ascii=False)

    parts = [
        master_prompt,
        "",
        "## Consultation Text",
        consultation_text,
        "",
        "## Household Budget Data",
        financial_json,
        "",
        "## Similar Cases (up to 3)",
    ]

    if not retrieved_cases:
        parts.append("(none)")
        parts.append("")
        parts.append("## Minimal Output Example (format only)")
        parts.append(
            '{"advice_text":"...","reduction_plan":{"Insurance":{"reduction_rate":0.12345}}}'
        )
        return "\n".join(parts)

    for i, case in enumerate(retrieved_cases, start=1):
        parts.append(f"### Similar Case {i}")
        advice_text = case.get("expert_advice_text", "")
        financial_case = case.get("financial_data_json", {})

        parts.append("expert_advice_text:")
        parts.append(str(advice_text))
        parts.append("")
        parts.append("financial_data_json:")
        parts.append(json.dumps(financial_case, ensure_ascii=False))
        parts.append("")

    parts.append("## Minimal Output Example (format only)")
    parts.append(
        '{"advice_text":"...","reduction_plan":{"Insurance":{"reduction_rate":0.12345}}}'
    )
    return "\n".join(parts).strip()


def format_duration(seconds: float) -> str:
    """Return elapsed time in mm:ss or hh:mm:ss."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def retrieve_similar_cases(
    index,
    embeddings: OpenAIEmbeddings,
    query_text: str,
    namespace: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve similar cases from Pinecone and return only required fields."""
    if not query_text:
        return []

    query_vector = embeddings.embed_query(query_text)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    matches = getattr(response, "matches", []) or []
    results: List[Dict[str, Any]] = []
    for match in matches:
        metadata = getattr(match, "metadata", {}) or {}
        raw_case_id = metadata.get("case_id", getattr(match, "id", None))
        try:
            case_id = int(raw_case_id)
        except Exception:
            case_id = raw_case_id
        result = {
            "case_id": case_id,
            "expert_advice_text": metadata.get("expert_advice_text", ""),
            "financial_data_json": parse_json_field(
                metadata.get("financial_data_json")
            ),
        }
        results.append(result)
    return results


def save_retrieval_results(
    fold_name: str, case_id: int, retrieved_cases: List[Dict[str, Any]]
) -> None:
    """Save retrieval results."""
    retrieval_dir = RETRIEVAL_BASE_DIR / fold_name
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    out_path = retrieval_dir / f"{case_id:04d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(retrieved_cases, f, ensure_ascii=False, indent=2)


def main() -> None:
    total_start = time.time()
    load_dotenv(".env")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return
    if not openai_api_key or not pinecone_api_key:
        print("Error: OPENAI_API_KEY or PINECONE_API_KEY not found in .env")
        return

    if not MASTER_PROMPT_PATH.exists():
        print(f"Error: master prompt not found: {MASTER_PROMPT_PATH}")
        return

    folds = load_folds()
    if START_FOLD not in folds:
        print(f"Error: fold name not found: {START_FOLD}")
        return


    def fold_sort_key(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 9999

    fold_names_sorted = sorted(folds.keys(), key=fold_sort_key)
    start_index = fold_names_sorted.index(START_FOLD)
    fold_names_sorted = fold_names_sorted[start_index:]

    master_prompt = load_master_prompt()
    client = genai.Client(api_key=gemini_api_key)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error: failed to connect to Pinecone: {e}")
        return

    print(f"Using model: {MODEL_NAME}")
    print(f"Index: {INDEX_NAME}, top_k={TOP_K}")
    print(f"Total limit: {TOTAL_LIMIT}")

    processed_count = 0
    parse_fail_count = 0

    for fold_name in fold_names_sorted:
        ids = folds.get(fold_name, [])
        results_dir = RESULTS_BASE_DIR / fold_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Fold start: {fold_name} (count: {len(ids)}) ===")

        for raw_id in ids:
            if TOTAL_LIMIT is not None and processed_count >= TOTAL_LIMIT:
                print("=== Test complete: total limit reached ===")
                total_elapsed = format_duration(time.time() - total_start)
                print(
                    f"Succeeded: {processed_count}, JSON parse failures: {parse_fail_count}"
                )
                print(f"Total processing time: {total_elapsed}")
                return

            try:
                case_id = int(raw_id)
            except Exception:
                print(f"Skip: ID cannot be converted to int: {raw_id}")
                continue

            input_path = USER_INPUT_DIR / f"{case_id:04d}.json"
            if not input_path.exists():
                print(f"Skip: input file does not exist: {input_path}")
                continue

            out_path = results_dir / f"{case_id:04d}.json"
            if SKIP_EXISTING and out_path.exists():
                rel_path = out_path.relative_to(BASE_DIR)
                print(f"Skip: existing output found {rel_path}")
                continue

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception as e:
                print(f"Skip: failed to read input JSON {input_path.name}: {e}")
                continue

            consultation_text = record.get("consultation_text", "")
            financial_data = record.get("financial_data", {})

            retrieved_cases = retrieve_similar_cases(
                index=index,
                embeddings=embeddings,
                query_text=consultation_text,
                namespace=fold_name,
                top_k=TOP_K,
            )

            prompt = build_prompt(
                master_prompt=master_prompt,
                consultation_text=consultation_text,
                financial_data=financial_data,
                retrieved_cases=retrieved_cases,
            )

            try:
                item_start = time.time()
                result_json = None
                response_text = ""

                for attempt in range(1, RETRY_COUNT + 2):
                    retry_prompt = prompt
                    if attempt > 2:
                        retry_prompt = (
                            prompt
                            + "\n\n"
                            + "The previous response included non-JSON text. Output JSON only."
                        )
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[retry_prompt],
                    )
                    response_text = (response.text or "").strip()

                    try:
                        result_json = json.loads(response_text)
                        break
                    except Exception as e:
                        if attempt <= RETRY_COUNT:
                            print(
                                f"JSON parse failed: {case_id:04d}.json (retry {attempt}/{RETRY_COUNT})"
                            )
                            time.sleep(1.0)
                            continue

                        parse_fail_count += 1
                        print(f"JSON parse failed: {case_id:04d}.json: {e}")
                        # Why: save the failed output for later inspection.
                        fail_dir = results_dir / "__parse_failures__"
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        fail_path = fail_dir / f"{case_id:04d}.txt"
                        with open(fail_path, "w", encoding="utf-8") as f:
                            f.write(response_text)
                        result_json = None

                if result_json is None:
                    continue

                save_retrieval_results(
                    fold_name=fold_name,
                    case_id=case_id,
                    retrieved_cases=retrieved_cases,
                )

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=2)

                processed_count += 1
                rel_path = out_path.relative_to(BASE_DIR)
                item_elapsed = format_duration(time.time() - item_start)
                print(f"Saved: {rel_path} (total {processed_count}, {item_elapsed})")
                time.sleep(3.0)

            except Exception as e:
                print(f"Error: failed to process {case_id:04d}.json: {e}")
                continue

        print(f"=== Fold end: {fold_name} ===")

    print("=== Completed all folds ===")
    total_elapsed = format_duration(time.time() - total_start)
    print(f"Succeeded: {processed_count}, JSON parse failures: {parse_fail_count}")
    print(f"Total processing time: {total_elapsed}")


if __name__ == "__main__":
    main()
