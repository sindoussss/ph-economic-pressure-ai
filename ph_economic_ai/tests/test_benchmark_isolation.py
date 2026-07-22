"""Guards the validated/exploratory boundary the README promises.

The benchmark is the defensible half of this project: strictly-causal
walk-forward backtests that anyone can reproduce with no API key, no GPU, and
no Qt. That claim is only worth making if it is enforced, so this test fails
the moment `benchmark/` grows a dependency on the exploratory app.
"""
import ast
import pathlib

import pytest

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1] / 'benchmark'

# Importing any of these would make the benchmark un-reproducible for a
# reviewer who installed only the validated-half requirements.
FORBIDDEN_PREFIXES = (
    'ph_economic_ai.engine',
    'ph_economic_ai.ui',
    'PyQt6',
    'ollama',
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _benchmark_files() -> list[pathlib.Path]:
    return sorted(
        p for p in BENCHMARK_DIR.rglob('*.py')
        if '__pycache__' not in p.parts
    )


def test_benchmark_package_is_not_empty():
    """Guard against the guard silently passing on an empty glob."""
    assert len(_benchmark_files()) > 5


@pytest.mark.parametrize('path', _benchmark_files(), ids=lambda p: p.name)
def test_benchmark_module_has_no_app_dependency(path):
    offenders = {
        mod for mod in _imported_modules(path)
        if mod.startswith(FORBIDDEN_PREFIXES)
    }
    assert not offenders, (
        f'{path.name} imports {sorted(offenders)}. The benchmark must stay '
        f'reproducible without the LLM app — see README "Validated vs exploratory".'
    )
