import json
import logging
import os
from pathlib import Path
from typing import Any


logger = logging.getLogger("HypoReflect")


def get_sample_companies(doc_info_path: str = "data/financebench_document_information.jsonl") -> list[str]:
    """Determine sample companies (one per unique GICS sector)."""
    if not os.path.exists(doc_info_path):
        logger.error("Document info file %s not found.", doc_info_path)
        return []

    try:
        with open(doc_info_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]

        sector_companies: dict[str, set[str]] = {}
        for item in data:
            sector = item.get("gics_sector")
            company = item.get("company")
            if sector and company:
                if sector not in sector_companies:
                    sector_companies[sector] = set()
                sector_companies[sector].add(company)

        sample_companies: list[str] = []
        for sector in sorted(sector_companies.keys()):
            companies = sorted(list(sector_companies[sector]))
            sample_companies.append(companies[0])
        return sample_companies
    except Exception as exc:
        logger.error("Error determining sample companies: %s", exc)
        return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _escape_md(text: Any) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    return raw.replace("|", "\\|")


def _to_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_escape_md(col) for col in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
