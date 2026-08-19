"""Every third-party module this package imports must be declared somewhere.

Found the hard way on 2026-08-19. `refresh_doe_adjustment` and
`refresh_meralco_all_in`, both written that day, import `pypdf` with no try/except
and no entry in any requirements file. They ran anyway on this machine, because the
UNPINNED system Python happens to have pypdf installed while the project's own
pinned venv does not. So the tools worked only outside the environment the project
specifies, and `python -m ph_economic_ai.tools.refresh_doe_adjustment` on a clean
checkout raises ModuleNotFoundError at the first PDF.

That is the failure mode worth guarding: a dependency that is satisfied by accident
is invisible until someone installs the project as documented.

The scan reads imports rather than trusting the requirements files, because the
direction that goes wrong is always the same. Nobody ships an import they did not
write; they ship one they did not declare.

An import may be absent from the requirements files only by appearing in
`_OPTIONAL_UNDECLARED` with a reason. That list is not an exemption so much as a
record: these four predate this test and were not investigated when it was written,
so they are named here instead of being silently tolerated.
"""
import ast
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / 'ph_economic_ai'

# Import name -> distribution name, where PyPI disagrees with the module.
_DISTRIBUTION = {
    'bs4': 'beautifulsoup4',
    'sklearn': 'scikit-learn',
    'fitz': 'PyMuPDF',
}

# Imported but deliberately not declared. Each entry states why, so that adding one
# is a decision rather than an oversight.
_OPTIONAL_UNDECLARED = {
    'praw': 'Reddit client, imported inside a try in refresh_social; the tool '
            'reports the provider as unavailable rather than failing',
    'pytrends': 'Google Trends client for an optional social signal',
    'fitz': 'PyMuPDF, used by the legacy doe_price_archive/series extractors; '
            'pypdf covers the paths that are current',
    'reportlab': 'PDF export in the stage-4 report, an app-side convenience',
}


_SPEC_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*$')


def _declaration(raw: str):
    """The distribution named by one requirements line, or None if it names none.

    Both requirements files are mostly prose. The first version of this parser
    stripped the comment marker and kept whatever followed, which put sentences
    into the set of declared packages; a comment merely CONTAINING the word pypdf
    would then have satisfied the very test this file exists to make fail. So a
    line counts as a declaration only if it looks like one.

    The dev pins sit behind `# ` because CI installs them separately, so commented
    lines are still read, but only when they carry a version pin. That is what
    separates `# pytest==9.1.1` from `# Dev / test only:`.
    """
    line = raw.strip()
    if line.startswith('#'):
        line = line.lstrip('#').strip()
        if '==' not in line:                     # prose, not a pin
            return None
    if not line or line.startswith('-r'):
        return None
    spec = line.split('#')[0].strip()            # drop any trailing comment
    if not spec or ' ' in spec:                  # a real spec has no spaces
        return None
    name = re.split(r'[=<>!~]', spec)[0].strip()
    return name.lower() if _SPEC_NAME.fullmatch(name) else None


def _declared() -> set:
    names = set()
    for req in ROOT.glob('requirements*.txt'):
        for raw in req.read_text(encoding='utf-8').splitlines():
            name = _declaration(raw)
            if name:
                names.add(name)
    return names


def _imported() -> dict:
    """Third-party root modules imported under ph_economic_ai/, excluding tests.

    Walks the AST rather than grepping, so an import nested inside a function is
    caught. Every pypdf import in this repository is nested inside a function.
    """
    found: dict = {}
    for path in PKG.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:                      # not our problem to report here
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:  # relative import, first party
                    continue
                mods = [node.module]
            else:
                continue
            for mod in mods:
                root = mod.split('.')[0]
                if root in sys.stdlib_module_names or root == 'ph_economic_ai':
                    continue
                found.setdefault(root, set()).add(
                    str(path.relative_to(ROOT)).replace('\\', '/'))
    return found


def test_every_third_party_import_is_declared_or_listed_optional():
    declared = _declared()
    missing = {}
    for mod, files in _imported().items():
        if mod in _OPTIONAL_UNDECLARED:
            continue
        if _DISTRIBUTION.get(mod, mod).lower() not in declared:
            missing[mod] = sorted(files)
    assert not missing, (
        'imported but not declared in any requirements file:\n' + '\n'.join(
            f'  {m}: {", ".join(f)}' for m, f in sorted(missing.items())) +
        '\nAdd it to requirements.txt, or to _OPTIONAL_UNDECLARED with a reason.')


def test_pypdf_specifically_is_declared():
    """The regression this file was written for.

    Named separately from the sweep above so that deleting or weakening the sweep
    cannot quietly restore the original defect.
    """
    assert 'pypdf' in _declared(), (
        'pypdf is imported unguarded by doe_adjustment, meralco and '
        'refresh_meralco_all_in. Undeclared, it is satisfied only by whatever the '
        'ambient interpreter happens to have.')


def test_the_optional_list_does_not_grow_stale():
    """An entry that is no longer imported should be removed, not left as decoration.

    Without this, the allowlist becomes the place undeclared dependencies go to be
    forgotten, which is the state this file was written to end.
    """
    imported = set(_imported())
    stale = sorted(set(_OPTIONAL_UNDECLARED) - imported)
    assert not stale, (
        f'listed as optional but no longer imported: {", ".join(stale)}. '
        'Remove the entry.')


@pytest.mark.parametrize('mod', sorted(_OPTIONAL_UNDECLARED))
def test_each_optional_entry_states_a_reason(mod):
    assert len(_OPTIONAL_UNDECLARED[mod].strip()) > 20, (
        f'{mod} needs a reason explaining why it is not declared')


@pytest.mark.parametrize('line', [
    '# Dev / test only:',
    '# PINNED, for the same reason as requirements.txt.',
    '#     ollama pull qwen2.5:3b        # fast tier',
    '# Strata -- benchmark dependencies (Python 3.10)',
    '# a pypdf mention inside prose must not count as declaring pypdf',
    '',
    '-r requirements.txt',
])
def test_prose_is_not_read_as_a_declaration(line):
    """The parser's own failure mode, pinned.

    A sentence must never enter the declared set. The last case is the one that
    matters: it would have made `test_pypdf_specifically_is_declared` pass while
    pypdf remained uninstallable.
    """
    assert _declaration(line) is None


@pytest.mark.parametrize('line,expected', [
    ('pandas==1.5.3', 'pandas'),
    ('  numpy==1.26.4  ', 'numpy'),
    ('# pytest==9.1.1', 'pytest'),
    ('openpyxl==3.1.5        # reading the World Bank workbook', 'openpyxl'),
    ('scikit-learn==1.7.2', 'scikit-learn'),
    ('PyQt6==6.10.0', 'pyqt6'),
])
def test_real_declarations_are_read(line, expected):
    """The other direction: tightening the parser must not blind it."""
    assert _declaration(line) == expected
