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
# reviewer who installed only the validated-half requirements. LLM backends are
# listed by provider rather than by our own wrapper, because `engine.llm` is
# provider-agnostic — pinning only 'ollama' would let a different backend in.
FORBIDDEN_PREFIXES = (
    'ph_economic_ai.engine',
    'ph_economic_ai.ui',
    'PyQt6',
    'PyQt5',
    'PySide6',
    'ollama',
    'openai',
    'anthropic',
    'transformers',
    'torch',
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Absolute module names imported by `path`, including inside functions.

    `ast.walk` reaches nested nodes, so a deferred `import` in a function body is
    caught too — that matters, because the tempting way to sneak an app
    dependency into the benchmark is a lazy import inside a helper.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _escaping_relative_imports(path: pathlib.Path) -> set[str]:
    """Relative imports that reach *outside* `benchmark/`.

    A relative import carries no package prefix, so the absolute-name scan above
    cannot see it: `from ..engine import llm` inside benchmark/ would sail past
    FORBIDDEN_PREFIXES entirely. Resolve the dot level against the file's depth
    instead — anything above the benchmark package root is escaping.
    """
    depth = len(path.relative_to(BENCHMARK_DIR).parts) - 1   # 0 for benchmark/*.py
    max_internal_level = depth + 1                           # the level landing on benchmark/
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > max_internal_level:
            target = node.module or ', '.join(a.name for a in node.names)
            out.add(f"{'.' * node.level}{target}")
    return out


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


@pytest.mark.parametrize('path', _benchmark_files(), ids=lambda p: p.name)
def test_benchmark_module_has_no_escaping_relative_import(path):
    """Closes the back door: a relative import has no module prefix, so it would
    otherwise bypass the FORBIDDEN_PREFIXES scan entirely."""
    escaping = _escaping_relative_imports(path)
    assert not escaping, (
        f'{path.name} reaches outside benchmark/ via {sorted(escaping)}. Relative '
        f'imports must stay inside the benchmark package.'
    )


def test_the_guard_actually_catches_violations(tmp_path):
    """A guard nobody has seen fail is not a guard. Feed it known-bad files and
    require that both checks fire."""
    absolute = tmp_path / 'bad_absolute.py'
    absolute.write_text('from ph_economic_ai.engine import llm\n', encoding='utf-8')
    assert any(m.startswith(FORBIDDEN_PREFIXES) for m in _imported_modules(absolute))

    lazy = tmp_path / 'bad_lazy.py'
    lazy.write_text('def f():\n    import ollama\n    return ollama\n', encoding='utf-8')
    assert any(m.startswith(FORBIDDEN_PREFIXES) for m in _imported_modules(lazy))

    # A relative escape, placed where a real benchmark module would sit.
    nested = BENCHMARK_DIR / 'unit_test_probe.py'
    tree = ast.parse('from ..engine import llm\n')
    assert any(
        n.level > 1 for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    ), 'depth arithmetic changed; _escaping_relative_imports needs review'
    assert not nested.exists()      # never actually write into benchmark/
