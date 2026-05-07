"""Paired bootstrap CI + McNemar test for benchmark result JSONs.

Reads the per-query `details` arrays from two benchmark result files (System A,
System B), pairs queries by their `query` text, and reports:

  - Per-system mean for each metric (llm_judge_score, hallucination, doc_match,
    page_match, answer_attempted, latency)
  - Paired bootstrap 95% CI for the mean difference (A - B), B=10000 by default
  - McNemar exact test on the binary judge_score >= 0.5 outcome
  - Win/loss/tie counts for each metric

Usage:

    python tools/bootstrap_ci.py \
        --a data/results/<ts>/hyporeflect/full/refl_on/agentic_off/*.json \
        --b data/results/<ts>/hoprag/hoprag_full/agentic_off/*.json

    # Compare every baseline against HypoReflect under one timestamp:
    python tools/bootstrap_ci.py --base-dir data/results/<ts> \
        --reference hyporeflect/full

The tool is read-only and does no GPU work; it operates entirely on existing
result files and runs in seconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

METRICS = (
    "llm_judge_score",
    "hallucination",
    "doc_match",
    "page_match",
    "answer_attempted",
    "latency",
)
LOWER_IS_BETTER = {"hallucination", "latency"}


@dataclass
class PerQuery:
    query: str
    values: dict[str, float]


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_details(path: str) -> list[PerQuery]:
    with open(path) as handle:
        data = json.load(handle)
    details = data.get("details") or []
    out: list[PerQuery] = []
    for entry in details:
        query = str(entry.get("query") or "").strip()
        if not query:
            continue
        values: dict[str, float] = {}
        for metric in METRICS:
            raw = _coerce_float(entry.get(metric))
            if raw is not None:
                values[metric] = raw
        out.append(PerQuery(query=query, values=values))
    return out


def pair_by_query(a: list[PerQuery], b: list[PerQuery]) -> list[tuple[PerQuery, PerQuery]]:
    by_query_b = {item.query: item for item in b}
    paired: list[tuple[PerQuery, PerQuery]] = []
    for item_a in a:
        item_b = by_query_b.get(item_a.query)
        if item_b is None:
            continue
        paired.append((item_a, item_b))
    return paired


def paired_bootstrap_ci(
    diffs: list[float],
    iterations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean_diff, lo, hi) percentile bootstrap CI of paired mean diff."""

    if not diffs:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(iterations):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += diffs[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * iterations)))
    hi_idx = min(iterations - 1, int(math.ceil((1.0 - alpha / 2.0) * iterations)) - 1)
    return (sum(diffs) / n, means[lo_idx], means[hi_idx])


def mcnemar_exact(a_wins: int, b_wins: int) -> float:
    """Two-sided exact McNemar p-value for discordant pair counts."""

    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    log_p_total = -n * math.log(2.0)
    cumulative = 0.0
    for i in range(k + 1):
        log_choose = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        )
        cumulative += math.exp(log_choose + log_p_total)
    p = min(1.0, 2.0 * cumulative)
    return p


def summarize_pair(
    name_a: str,
    name_b: str,
    paired: list[tuple[PerQuery, PerQuery]],
    iterations: int,
) -> dict:
    out = {
        "system_a": name_a,
        "system_b": name_b,
        "n_paired": len(paired),
        "metrics": {},
    }
    if not paired:
        return out

    for metric in METRICS:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for a, b in paired:
            if metric not in a.values or metric not in b.values:
                continue
            a_vals.append(a.values[metric])
            b_vals.append(b.values[metric])
        if not a_vals:
            continue

        diffs = [a_vals[i] - b_vals[i] for i in range(len(a_vals))]
        mean_diff, lo, hi = paired_bootstrap_ci(diffs, iterations=iterations)
        higher_is_better = metric not in LOWER_IS_BETTER
        a_wins = sum(
            1
            for diff in diffs
            if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better)
        )
        b_wins = sum(
            1
            for diff in diffs
            if (diff < 0 and higher_is_better) or (diff > 0 and not higher_is_better)
        )
        ties = len(diffs) - a_wins - b_wins
        ci_excludes_zero = (lo > 0.0) or (hi < 0.0)
        out["metrics"][metric] = {
            "n": len(a_vals),
            "mean_a": sum(a_vals) / len(a_vals),
            "mean_b": sum(b_vals) / len(b_vals),
            "mean_diff_a_minus_b": mean_diff,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "ci95_excludes_zero": ci_excludes_zero,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "lower_is_better": metric in LOWER_IS_BETTER,
        }

        if metric == "llm_judge_score":
            a_correct = sum(1 for value in a_vals if value >= 0.5)
            b_correct = sum(1 for value in b_vals if value >= 0.5)
            disc_a = sum(
                1
                for i, value in enumerate(a_vals)
                if value >= 0.5 and b_vals[i] < 0.5
            )
            disc_b = sum(
                1
                for i, value in enumerate(a_vals)
                if value < 0.5 and b_vals[i] >= 0.5
            )
            out["metrics"][metric]["binary_at_0.5"] = {
                "a_correct": a_correct,
                "b_correct": b_correct,
                "discordant_a_only": disc_a,
                "discordant_b_only": disc_b,
                "mcnemar_two_sided_p": mcnemar_exact(disc_a, disc_b),
            }
    return out


def format_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"=== {summary['system_a']}  vs  {summary['system_b']}    n={summary['n_paired']} paired ==="
    )
    if not summary.get("metrics"):
        lines.append("(no overlapping queries)")
        return "\n".join(lines)

    header = f"{'metric':<20}{'meanA':>10}{'meanB':>10}{'diff':>10}{'CI95_lo':>10}{'CI95_hi':>10}{'sig':>6}{'A>B':>6}{'B>A':>6}{'tie':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for metric in METRICS:
        row = summary["metrics"].get(metric)
        if row is None:
            continue
        sig = "*" if row["ci95_excludes_zero"] else ""
        lines.append(
            f"{metric:<20}"
            f"{row['mean_a']:>10.4f}"
            f"{row['mean_b']:>10.4f}"
            f"{row['mean_diff_a_minus_b']:>+10.4f}"
            f"{row['ci95_lo']:>+10.4f}"
            f"{row['ci95_hi']:>+10.4f}"
            f"{sig:>6}"
            f"{row['a_wins']:>6d}"
            f"{row['b_wins']:>6d}"
            f"{row['ties']:>6d}"
        )

    judge = summary["metrics"].get("llm_judge_score", {}).get("binary_at_0.5")
    if judge is not None:
        lines.append("")
        lines.append(
            f"McNemar (judge>=0.5): A_only_correct={judge['discordant_a_only']}, "
            f"B_only_correct={judge['discordant_b_only']}, "
            f"two-sided p={judge['mcnemar_two_sided_p']:.4g}"
        )
    return "\n".join(lines)


def _resolve_glob(pattern: str) -> str:
    matches = glob.glob(pattern)
    matches = [match for match in matches if match.endswith(".json")
               and ".summary." not in match
               and ".stage_diagnostics." not in match
               and not os.path.basename(match).startswith("run_")]
    if not matches:
        raise FileNotFoundError(f"No detail JSON matched: {pattern}")
    if len(matches) > 1:
        matches.sort()
        sys.stderr.write(f"[warn] glob matched {len(matches)} files; using {matches[0]}\n")
    return matches[0]


def _discover_detail_files(base_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in base_dir.rglob("*.json"):
        name = path.name
        if name.startswith("run_") or ".summary." in name or ".stage_diagnostics." in name:
            continue
        relative = path.relative_to(base_dir)
        parts = relative.parts
        if len(parts) < 2:
            continue
        key = "/".join(parts[:-1])
        if key not in out or path.stat().st_mtime > Path(out[key]).stat().st_mtime:
            out[key] = str(path)
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="Path or glob to System A detail JSON")
    parser.add_argument("--b", help="Path or glob to System B detail JSON")
    parser.add_argument("--base-dir", help="Discover all detail JSONs under this dir; pair against --reference")
    parser.add_argument("--reference", help="Subpath under --base-dir to use as System A (e.g. hyporeflect/full/refl_on/agentic_off)")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="Write structured results to this JSON file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    random.seed(args.seed)
    pair_summaries: list[dict] = []

    if args.base_dir:
        if not args.reference:
            parser.error("--base-dir requires --reference")
        base_dir = Path(args.base_dir)
        files = _discover_detail_files(base_dir)
        ref_match = next((path for key, path in files.items() if key.startswith(args.reference)), None)
        if ref_match is None:
            parser.error(f"reference path not found under {base_dir}: {args.reference}")
        a_details = load_details(ref_match)
        a_label = args.reference
        for key, path in sorted(files.items()):
            if key.startswith(args.reference):
                continue
            b_details = load_details(path)
            paired = pair_by_query(a_details, b_details)
            summary = summarize_pair(a_label, key, paired, args.iterations)
            pair_summaries.append(summary)
            print(format_report(summary))
            print()
    else:
        if not (args.a and args.b):
            parser.error("provide either --a/--b or --base-dir/--reference")
        path_a = _resolve_glob(args.a)
        path_b = _resolve_glob(args.b)
        a_details = load_details(path_a)
        b_details = load_details(path_b)
        paired = pair_by_query(a_details, b_details)
        summary = summarize_pair(path_a, path_b, paired, args.iterations)
        pair_summaries.append(summary)
        print(format_report(summary))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump({"pairs": pair_summaries, "iterations": args.iterations}, handle, indent=2)
        print(f"\n[ok] wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
