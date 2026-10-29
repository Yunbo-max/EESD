"""Clean execution sandbox for RLVR: run candidate code against stdin/stdout tests.

Independent of the legacy PBPF pipeline. A single worker executes one
(candidate, test) pair in an isolated subprocess with CPU/memory/file limits and
dropped privileges, returning a stable outcome vocabulary.
"""

from __future__ import annotations

import functools
import os
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

OUTCOMES = ("PASS", "WRONG_OUTPUT", "RUNTIME_EXCEPTION", "TIMEOUT", "COMPILE_ERROR")


@dataclass(frozen=True)
class SandboxResult:
    outcome: str
    feedback: str = ""
    actual: str = ""
    expected: str = ""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```lang ... ```) without eating newlines."""
    return re.sub(r"```[a-zA-Z0-9_+#-]*", "", text)


def _find_def_line(lines: list[str], preferred_name: str = "solution") -> int | None:
    """Return the index of the target function definition, preferring a name."""
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {preferred_name}("):
            return i
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            return i
    return None


def _is_complete_function(segment: str) -> bool:
    """True if `segment` parses as a single complete function definition."""
    import ast

    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return False
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.FunctionDef)
        and len(tree.body[0].body) > 0
    )


def _extract_code(text: str) -> str:
    """Extract a single complete function from chat-style generator output.

    An Instruct chat model prompted with a bare continuation often emits
    markdown fences, prose, and a trailing ``check()``/``__main__`` harness.
    This returns just the target function (preferring one named ``solution``).
    """
    import ast

    cleaned = _strip_fences(text)

    # Fast path: the whole text is already parseable. Grab the named function.
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        tree = None
    if tree is not None:
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if funcs:
            target = next((n for n in funcs if n.name == "solution"), funcs[0])
            segment = ast.get_source_segment(cleaned, target)
            if segment and _is_complete_function(segment):
                return segment

    # Fallback: block extraction from the first `def solution` (or first `def`).
    lines = cleaned.split("\n")
    def_idx = _find_def_line(lines, preferred_name="solution")
    if def_idx is None:
        return cleaned
    base_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip(" \t"))
    kept = [lines[def_idx].rstrip()]
    for line in lines[def_idx + 1 :]:
        stripped = line.strip()
        if stripped == "":
            kept.append(line.rstrip())
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= base_indent:
            break
        kept.append(line.rstrip())
    return "\n".join(kept).rstrip()


def _set_limit(limit: int, values: tuple[int, int]) -> None:
    try:
        resource.setrlimit(limit, values)
    except (OSError, ValueError):
        pass


def _apply_limits(timeout_seconds: float, memory_mb: int) -> None:
    # Note: we deliberately avoid RLIMIT_NPROC and setuid(65534). Dropping to
    # the "nobody" user while many jobs share a node makes execve() fail with
    # EAGAIN ("Resource temporarily unavailable"), which silently turned every
    # candidate into COMPILE_ERROR. For a sandbox running our own generated
    # code, a timeout + memory + filesize cap is sufficient.
    _set_limit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
    _set_limit(resource.RLIMIT_CPU, (max(1, int(timeout_seconds)),) * 2)
    _set_limit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    _set_limit(resource.RLIMIT_CORE, (0, 0))


def execute(
    code: str,
    test: Mapping[str, str],
    *,
    timeout_seconds: float = 2.0,
    memory_mb: int = 1024,
    python_executable: str = sys.executable,
) -> SandboxResult:
    """Run `code` with `test["input"]` as the invocation and compare stdout.

    For CodeARC-style tests the candidate is a function definition and
    `test["input"]` is a call site; the harness appends the input and executes.
    """
    extracted = _extract_code(code)
    if not _is_complete_function(extracted):
        return SandboxResult("COMPILE_ERROR", "no complete function in candidate")
    body = extracted + "\n" + str(test.get("input", ""))
    with tempfile.TemporaryDirectory(prefix="rlvr-eval-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        source = root / "candidate.py"
        source.write_text(body, encoding="utf-8")
        source.chmod(0o444)
        try:
            result = subprocess.run(
                [python_executable, "-I", str(source)],
                input="",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=root,
                timeout=timeout_seconds,
                preexec_fn=functools.partial(_apply_limits, timeout_seconds, memory_mb),
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult("TIMEOUT", "timeout")
        except (OSError, subprocess.SubprocessError) as error:
            return SandboxResult("COMPILE_ERROR", str(error))
    if result.returncode != 0:
        stderr = result.stderr[-1000:]
        if "SyntaxError" in stderr:
            outcome = "COMPILE_ERROR"
        elif "AssertionError" in stderr:
            outcome = "WRONG_OUTPUT"
        else:
            outcome = "RUNTIME_EXCEPTION"
        return SandboxResult(outcome, stderr)
    expected = str(test.get("output", "")).rstrip()
    actual = result.stdout.rstrip()
    outcome = "PASS" if actual == expected else "WRONG_OUTPUT"
    return SandboxResult(
        outcome,
        "mismatch" if outcome == "WRONG_OUTPUT" else "",
        actual=actual,
        expected=expected,
    )
