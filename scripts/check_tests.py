import sys
import subprocess
from pathlib import Path
from subprocess import CompletedProcess


SOURCE_PREFIXES = ("src/", "api/", "pipeline/", "utils/")


def get_staged_files() -> list[str]:
    """Return list of staged file paths.
    Uses 'git diff --cached --name-only' to get the list of staged files.
    Returns:
        List of staged file paths as strings.
    """
    result: CompletedProcess[str] = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error: Could not get staged files.")
        sys.exit(1)
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def main() -> int:
    staged: list[str] = get_staged_files()
    if not staged:
        print("No staged files to check.")
        return 0

    # Filter for changed source Python files in project source locations.
    source_files: list[str] = [
        f
        for f in staged
        if f.endswith(".py")
        and not f.startswith("tests/")
        and not f.endswith("__init__.py")
        and (f.startswith(SOURCE_PREFIXES) or f == "application.py")
    ]

    if not source_files:
        print("No project source Python files changed (excluding tests/__init__.py).")
        return 0

    staged_test_files: list[str] = [
        f for f in staged if f.startswith("tests/") and f.endswith(".py")
    ]

    errors: list[str] = []
    for src in source_files:
        src_stem: str = Path(src).stem
        has_staged_test: bool = any(
            src_stem in Path(test_file).stem for test_file in staged_test_files
        )
        if not has_staged_test:
            errors.append(
                f"  - {src}: no corresponding staged test found in tests/ for '{src_stem}'"
            )

    if errors:
        print("Commit rejected: The following changed source files lack unit tests:")
        for e in errors:
            print(e)
        print("\nPlease add or update matching tests in tests/ and stage them.")
        return 1

    print("All changed source files have corresponding test files staged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
