"""Score Mars data V3 predictions without making up 'intelligence' metrics.

Input JSONL rows: {"id": "...", "response": "model output"}.
Missing predictions count as wrong. Unknown IDs and duplicate IDs fail.
Outputs exact/normalized match by task and split, with limited scope.

A simple prompt-lookup baseline is included to demonstrate that exact
memorization of training question strings fails on held-out inputs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .dataset_v3 import DataPack, Example, build_pack, normalize_prompt


def normalize_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer.casefold()).strip()
    return answer.rstrip(" .!?,")


def read_predictions(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at line {line_number}")
        row_id, response = value.get("id"), value.get("response")
        if not isinstance(row_id, str) or not isinstance(response, str):
            raise ValueError(f"Expected string id and response at line {line_number}")
        if row_id in result:
            raise ValueError(f"Duplicate prediction id: {row_id}")
        result[row_id] = response
    return result


def score(pack: DataPack, predictions: dict[str, str]) -> dict:
    all_items = pack.train + pack.validation + pack.test
    ids = {x.id for x in all_items}
    unknown = set(predictions) - ids
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:5]}")
    output: dict = {
        "dataset": "mars-data-v3",
        "caution": ("Exact string match is a strict demonstration metric, not a "
                    "measure of general reasoning or semantic accuracy."),
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        rows = getattr(pack, split_name)
        tasks: dict[str, list[bool]] = defaultdict(list)
        strict_tasks: dict[str, list[bool]] = defaultdict(list)
        for item in rows:
            response = predictions.get(item.id, "")
            matched = normalize_answer(response) == normalize_answer(item.answer)
            exact = response.strip() == item.answer.strip()
            tasks[item.task].append(matched)
            strict_tasks[item.task].append(exact)

        output["splits"][split_name] = {
            "total": len(rows),
            "predicted": sum(item.id in predictions for item in rows),
            "normalized_exact_match": {
                task: {"correct": sum(values), "total": len(values)}
                for task, values in sorted(tasks.items())
            },
            "strict_exact_match": {
                task: {"correct": sum(strict_tasks[task]), "total": len(strict_tasks[task])}
                for task in sorted(strict_tasks)
            },
        }
    return output


def prompt_lookup_baseline(pack: DataPack) -> dict[str, str]:
    table = {normalize_prompt(x.prompt): x.answer for x in pack.train}
    return {
        x.id: table.get(normalize_prompt(x.prompt), "")
        for x in pack.train + pack.validation + pack.test
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Mars V3 generated answers")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", help="Model predictions as id/response JSONL")
    group.add_argument("--baseline", action="store_true", help="Exact prompt-lookup baseline")
    parser.add_argument("--output", default="artifacts/mars-data-v3/scores.json")
    args = parser.parse_args()

    pack = build_pack()
    predictions = (
        prompt_lookup_baseline(pack) if args.baseline
        else read_predictions(Path(args.predictions))
    )
    report = score(pack, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split, details in report["splits"].items():
        print(f"{split}: {details['predicted']}/{details['total']} answered")
        for task, result in details["normalized_exact_match"].items():
            print(f"  {task}: {result['correct']}/{result['total']} normalized matches")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
