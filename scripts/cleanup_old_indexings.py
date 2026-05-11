"""Pre-flight cleanup before re-indexing MS GraphRAG / HopRAG with their
official pipelines.

Drops:
  - Old hoprag indexings written under hyporeflect's GraphRAG engine
    (label `HO_<tag>_Chunk`, `HO_<tag>_Document`, indices `hoprag_<tag>_*`).
  - Old ms_graphrag indexings, same pattern (`MS_<tag>_Chunk`, `ms_graphrag_<tag>_*`).
  - Optionally smoke labels (`HO_smoke_*`).
  - parquet/lancedb trees under data/ms_graphrag_output and data/hoprag_output
    that match the same corpus tags (re-indexer rewrites these anyway).

Always preserves:
  - HY_*  (hyporeflect production)
  - NA_*  (naive)
  - hyporeflect_*_idx (vector/text indices over HY_ labels)
  - naive_*_idx
  - HO_smoke_h2 / smoke_v8 outputs by default (override with --drop-smoke).

Usage:
  python scripts/cleanup_old_indexings.py            # DRY RUN — print only
  python scripts/cleanup_old_indexings.py --apply    # actually delete
  python scripts/cleanup_old_indexings.py --apply --drop-smoke
  python scripts/cleanup_old_indexings.py --apply --neo4j-only
  python scripts/cleanup_old_indexings.py --apply --parquet-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from neo4j import GraphDatabase


_KEEP_LABEL_PREFIXES = ("HY_", "NA_")
_DROP_LABEL_PREFIXES = ("HO_", "MS_")
_DROP_INDEX_PREFIXES = ("HO_", "MS_", "hoprag_", "ms_graphrag_")


def _classify_label(label: str, drop_smoke: bool) -> str:
    if any(label.startswith(p) for p in _KEEP_LABEL_PREFIXES):
        return "keep"
    if not drop_smoke and "_smoke_" in label:
        return "keep"
    if any(label.startswith(p) for p in _DROP_LABEL_PREFIXES):
        return "drop"
    return "ignore"


def _classify_index(name: str, drop_smoke: bool) -> str:
    # Indices belonging to HY_/NA_ labels are NOT prefixed with HY_/NA_ themselves
    # (e.g., `hyporeflect_<tag>_vector_idx`). Catch by substring.
    if name.startswith(("HY_", "hyporeflect_", "NA_", "naive_")):
        return "keep"
    if not drop_smoke and "_smoke_" in name:
        return "keep"
    if any(name.startswith(p) for p in _DROP_INDEX_PREFIXES):
        return "drop"
    return "ignore"


def cleanup_neo4j(apply: bool, drop_smoke: bool) -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "1q2w3e4r")
    print(f"\n=== Neo4j cleanup ({'APPLY' if apply else 'DRY-RUN'}) ===")
    print(f"  URI: {uri}")
    if drop_smoke:
        print("  drop_smoke=True (HO_smoke_* / *_smoke_* indices will be removed)")

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        with driver.session() as s:
            # ----- Labels -----
            all_labels = [
                r["label"] for r in s.run(
                    "CALL db.labels() YIELD label RETURN label ORDER BY label"
                )
            ]
            to_drop_labels = []
            kept_labels = []
            for label in all_labels:
                cls = _classify_label(label, drop_smoke)
                count = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]
                if count == 0:
                    continue  # already empty
                if cls == "drop":
                    to_drop_labels.append((label, count))
                elif cls == "keep":
                    kept_labels.append((label, count))

            print(f"\n--- Labels: {len(to_drop_labels)} to drop, {len(kept_labels)} to keep ---")
            print("  KEEP:")
            for label, count in kept_labels:
                print(f"    {label:60s} {count:>10d}")
            print("  DROP:")
            total_to_drop = 0
            for label, count in to_drop_labels:
                print(f"    {label:60s} {count:>10d}")
                total_to_drop += count
            print(f"  Total nodes to drop: {total_to_drop:,}")

            if apply:
                for label, count in to_drop_labels:
                    print(f"  ... dropping {label} ({count:,} nodes)")
                    s.run(
                        f"MATCH (n:`{label}`) "
                        f"CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 5000 ROWS"
                    )

            # ----- Indices -----
            all_indices = list(s.run(
                "SHOW INDEXES YIELD name, type WHERE name <> '' RETURN name, type"
            ))
            to_drop_idx = []
            kept_idx = 0
            for r in all_indices:
                cls = _classify_index(r["name"], drop_smoke)
                if cls == "drop":
                    to_drop_idx.append(r["name"])
                elif cls == "keep":
                    kept_idx += 1

            print(f"\n--- Indices: {len(to_drop_idx)} to drop, {kept_idx} to keep ---")
            for name in to_drop_idx:
                print(f"    {name}")

            if apply:
                for name in to_drop_idx:
                    print(f"  ... dropping index {name}")
                    s.run(f"DROP INDEX `{name}` IF EXISTS")
    finally:
        driver.close()


def cleanup_parquet(apply: bool, drop_smoke: bool) -> None:
    print(f"\n=== Parquet/lancedb cleanup ({'APPLY' if apply else 'DRY-RUN'}) ===")
    roots = [Path("data/ms_graphrag_output"), Path("data/hoprag_output")]
    for root in roots:
        if not root.exists():
            print(f"  {root}: (not present)")
            continue
        children = sorted(p for p in root.iterdir() if p.is_dir())
        if not children:
            print(f"  {root}: empty")
            continue
        print(f"  {root}/:")
        for child in children:
            is_smoke = "smoke" in child.name.lower()
            keep = is_smoke and not drop_smoke
            try:
                size_mb = sum(p.stat().st_size for p in child.rglob("*") if p.is_file()) / 1024 / 1024
            except OSError:
                size_mb = 0.0
            tag = "[KEEP]" if keep else "[DROP]"
            print(f"    {tag} {child.name:50s} {size_mb:>8.1f} MB")
            if apply and not keep:
                shutil.rmtree(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--neo4j-only", action="store_true")
    parser.add_argument("--parquet-only", action="store_true")
    parser.add_argument("--drop-smoke", action="store_true",
                        help="Also drop *_smoke_* labels/indices/parquet (default: keep)")
    args = parser.parse_args()

    if args.neo4j_only and args.parquet_only:
        print("ERROR: --neo4j-only and --parquet-only are mutually exclusive", file=sys.stderr)
        return 2

    if not args.parquet_only:
        cleanup_neo4j(apply=args.apply, drop_smoke=args.drop_smoke)
    if not args.neo4j_only:
        cleanup_parquet(apply=args.apply, drop_smoke=args.drop_smoke)

    if not args.apply:
        print("\n(dry-run only — re-run with --apply to actually delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
