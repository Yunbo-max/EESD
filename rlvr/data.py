"""Load CodeARC problems + invocations and build RLVR sources.

A source is a function signature + docstring (the prompt) plus its invocations
(input/output pairs). The first four invocations are public (used for reward);
the rest are held out for evaluator-only scoring.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping

PUBLIC_TESTS = 4
TOTAL_TESTS = 10


def _signature_and_doc(code: str) -> str:
    """Return the function signature + docstring, stopping before the body."""
    try:
        tree = ast.parse(code)
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        doc = ast.get_docstring(fn, clean=False)
        if doc is None:
            return code.split("\n")[0]
        prefix = code[: code.index(doc) + len(doc)]
        return prefix.rstrip()
    except (SyntaxError, ValueError, StopIteration):
        return code.split("\n")[0]


def _example_line(test: Mapping[str, str]) -> str:
    call = re.search(r"solution\([^\n]*?\)", test["input"])
    result = re.sub(r"^Result \d+: ", "", test["output"]).strip()
    if call is None:
        return f"# Example output: {result}"
    return f"# Example: {call.group(0)} -> {result}"


def _make_prompt(signature: str, public_tests: list[dict[str, str]]) -> str:
    example = _example_line(public_tests[0])
    return (
        f"{signature}\n"
        f"    {example}\n"
        f"    # Implement the function body (no input(), no main).\n"
    )


def make_chat_messages(raw_prompt: str) -> list[dict[str, str]]:
    """Wrap a raw code-completion prompt as an Instruct chat exchange.

    Qwen2.5-Coder-Instruct is a chat model; feeding it a bare continuation makes
    it emit markdown fences, prose, and a test harness. Wrapping the task as a
    chat message keeps it in its trained mode and returns clean code.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. Complete the function body "
                "with correct, runnable code. Return only the function body: no "
                "markdown fences, no explanation, and no test code."
            ),
        },
        {
            "role": "user",
            "content": (
                "Complete the function body (the indented part below the "
                "signature):\n\n" + raw_prompt
            ),
        },
    ]


def load_sources(
    problems_path: str | Path,
    invocations_path: str | Path,
    *,
    n_sources: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    problems_path = Path(problems_path)
    invocations_path = Path(invocations_path)
    invocations_by_problem: dict[int, list[Mapping[str, Any]]] = {}
    with invocations_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "anonymous":
                continue
            invocations_by_problem.setdefault(int(row["problem_id"]), []).append(row)

    sources: list[dict[str, Any]] = []
    with problems_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            problem = json.loads(line)
            if problem.get("type") != "anonymous":
                continue
            problem_id = int(problem["problem_id"])
            invocations = invocations_by_problem.get(problem_id, [])
            if len(invocations) < TOTAL_TESTS:
                continue
            invocations = sorted(invocations, key=lambda row: int(row["index"]))
            tests = [
                {"input": str(row["input"]), "output": str(row["output"])}
                for row in invocations[:TOTAL_TESTS]
            ]
            sources.append(
                {
                    "problem_id": problem_id,
                    "prompt": _make_prompt(
                        _signature_and_doc(str(problem["code"])), tests[:PUBLIC_TESTS]
                    ),
                    "gold": str(problem["code"]),
                    "public_tests": tests[:PUBLIC_TESTS],
                    "hidden_tests": tests[PUBLIC_TESTS:],
                }
            )
    sources.sort(key=lambda source: source["problem_id"])
    if n_sources is not None:
        import random

        rng = random.Random(seed)
        sources = rng.sample(sources, min(n_sources, len(sources)))
    return sources
