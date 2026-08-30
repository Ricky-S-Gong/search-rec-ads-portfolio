"""Small deterministic fallback runner for environments where pytest startup hangs."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
TEST_ROOTS = (
    ROOT / "research" / "spotify-music" / "tests",
    ROOT / "research" / "movielens-cf" / "tests",
    ROOT / "research" / "goodbooks-mf" / "tests",
)


def main() -> int:
    sys.path.insert(0, str(ROOT / "research" / "movielens-cf"))
    sys.path.insert(0, str(ROOT / "research" / "goodbooks-mf"))
    failures: list[str] = []
    passed = 0
    with tempfile.TemporaryDirectory(prefix="portfolio-tests-") as temp:
        temp_root = Path(temp)
        for file_index, path in enumerate(
            test_file for root in TEST_ROOTS for test_file in sorted(root.glob("test_*.py"))
        ):
            name = f"portfolio_test_{file_index}_{path.stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                failures.append(f"{path}: could not load module")
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as error:  # pragma: no cover - visible runner failure
                failures.append(f"{path}: import failed: {error}")
                continue
            for test_name, test in vars(module).items():
                if not test_name.startswith("test_") or not callable(test):
                    continue
                kwargs = {}
                parameters = inspect.signature(test).parameters
                if "tmp_path" in parameters:
                    fixture = temp_root / f"{file_index}-{test_name}"
                    fixture.mkdir()
                    kwargs["tmp_path"] = fixture
                unsupported = set(parameters) - set(kwargs)
                if unsupported:
                    failures.append(f"{path.name}::{test_name}: unsupported fixtures {sorted(unsupported)}")
                    continue
                try:
                    test(**kwargs)
                    passed += 1
                except Exception as error:  # pragma: no cover - visible runner failure
                    failures.append(f"{path.name}::{test_name}: {error}")
    print(f"Python tests: {passed} passed, {len(failures)} failed")
    for failure in failures:
        print(f"FAIL {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
