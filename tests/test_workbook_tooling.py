from __future__ import annotations

import subprocess
import sys


def run_coach(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/coach.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_coach_lists_steps() -> None:
    result = run_coach("list")

    assert result.returncode == 0
    assert "01: Python numbers" in result.stdout
    assert "08: Attention" in result.stdout


def test_coach_shows_step_instructions() -> None:
    result = run_coach("show", "01")

    assert result.returncode == 0
    assert "Edit: workbook/steps/step_01_python_primitives.py" in result.stdout
    assert "Check: python3 tools/coach.py check 01" in result.stdout


def test_reference_solutions_pass_all_workbook_checks() -> None:
    result = run_coach("check", "all", "--solution")

    assert result.returncode == 0
    assert "Passed reference solutions check for step all." in result.stdout
