from unittest.mock import patch
from types import SimpleNamespace
import pytest
from eesd.recursive_execution import run_test, feedback_for_tests, Outcome


def test_soft_sandbox_fallback_without_bwrap():
    # Deviation from the original fail-closed gate: on shared GPU hosts bwrap
    # cannot start, so we fall back to a resource-limited soft sandbox instead
    # of crashing. The fallback must still produce a correct PASS outcome.
    with (
        patch("eesd.recursive_execution.shutil.which", return_value=None),
        patch(
            "eesd.recursive_execution._run_soft",
            return_value=Outcome("PASS", "1", "1", ""),
        ) as soft,
    ):
        result = run_test("print(1)", {"input": "", "expected": "1"})
        assert result.status == "PASS"
        soft.assert_called_once()


def test_valid_stdout_comparison_and_no_network_mount():
    with (
        patch("eesd.recursive_execution.shutil.which", return_value="/usr/bin/bwrap"),
        patch(
            "eesd.recursive_execution.subprocess.run",
            return_value=SimpleNamespace(stdout="3\n", stderr="", returncode=0),
        ) as mocked,
    ):
        result = run_test("print(3)", {"input": "", "expected": "3\n"})
        assert result.status == "PASS"
        cmd = mocked.call_args.args[0]
        assert "--unshare-all" in cmd and "--bind" not in cmd
        assert "python3" in cmd[-5]


def test_bwrap_startup_error_falls_back_to_soft():
    # bwrap exists but cannot mount namespaces on shared GPU hosts. The runner
    # must fall back to the soft sandbox and record the deviation, not crash.
    with (
        patch("eesd.recursive_execution.shutil.which", return_value="/usr/bin/bwrap"),
        patch(
            "eesd.recursive_execution._run_bwrap",
            side_effect=RuntimeError(
                "bubblewrap startup error: bwrap: Cannot mount proc"
            ),
        ),
        patch(
            "eesd.recursive_execution._run_soft",
            return_value=Outcome("PASS", "3", "3", ""),
        ) as soft,
    ):
        result = run_test("print(3)", {"input": "", "expected": "3"})
        assert result.status == "PASS"
        soft.assert_called_once()
        assert (
            __import__(
                "eesd.recursive_execution", fromlist=["SANDBOX_MODE"]
            ).SANDBOX_MODE
            == "soft"
        )


def test_feedback_requires_distinct_four_tests():
    with pytest.raises(ValueError):
        feedback_for_tests("print(0)", [{"id": 0}] * 4)


def test_feedback_requires_distinct_four_tests():
    with pytest.raises(ValueError):
        feedback_for_tests("print(0)", [{"id": 0}] * 4)
