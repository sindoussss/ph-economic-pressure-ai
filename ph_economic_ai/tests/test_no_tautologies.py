"""A comparison of a value with itself asserts nothing, and reads as if it does.

`DEC-060` recorded the shape once, from `attempts == _TRENDS_ATTEMPTS` surviving
a mutation. It has since appeared twice more:

* `honesty.regions_resting_on_an_assumption` filtered on
  `g['name'] == g.get('name')`, a condition that was always true.
* `test_the_identity_does_not_depend_on_which_agent_finished_first` built both
  orderings, discarded the second, and asserted `one == one`. The comparison in
  the test's own name never happened.

Three occurrences is a pattern rather than an accident, so the sweep that found
the third runs as a test. It is an AST walk, not a grep: the source text of a
correct fix contains the defective expression in the comment that explains it,
which is how the first version of this check failed on a corrected function.

`f(x) == f(x)` is deliberately allowed. Calling the same function twice and
comparing is the standard way to assert determinism, and this codebase does it
in six places on purpose -- seeds, embedding keys, roster assignment.
"""
import ast
import pathlib

import ph_economic_ai

_ROOT = pathlib.Path(ph_economic_ai.__file__).parent


def _sources():
    for path in sorted(_ROOT.rglob('*.py')):
        try:
            yield path, ast.parse(path.read_bytes().decode('utf-8-sig'))
        except SyntaxError:                     # not ours to police
            continue


def _is_accessor(node) -> bool:
    """`d.get('k')` and `d.get('k', default)` -- a lookup, not a computation."""
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
            and node.args and isinstance(node.args[0], ast.Constant))


def _is_call(node) -> bool:
    """Whether an expression computes something, and so may differ between two
    evaluations. `f(x) == f(x)` is a determinism assertion, not a tautology.

    A `.get(literal)` lookup is exempt from that exemption: the real defect was
    `g['name'] == g.get('name')`, where one side is a call and the other is not.
    Treating `.get` as a computation would let exactly that form through.
    """
    return any(isinstance(n, ast.Call) and not _is_accessor(n)
               for n in ast.walk(node))


def _key(node) -> str:
    """A comparable form for an expression, with `d.get('k')` read as `d['k']`.

    The one semantic equivalence this guard understands, because it is the one
    that actually occurred. It does not attempt general semantic comparison:
    two expressions that mean the same thing by any other route will pass.
    """
    class _Fold(ast.NodeTransformer):
        def visit_Call(self, n):            # noqa: N802 - ast naming
            self.generic_visit(n)
            if _is_accessor(n):
                return ast.Subscript(value=n.func.value, slice=n.args[0],
                                     ctx=ast.Load())
            return n

    return ast.dump(_Fold().visit(ast.parse(ast.unparse(node), mode='eval').body))


def test_nothing_is_compared_with_itself():
    offenders = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                continue
            left, right = node.left, node.comparators[0]
            if _key(left) != _key(right) or _is_call(left):
                continue
            offenders.append(
                f'{path.relative_to(_ROOT)}:{node.lineno}  {ast.unparse(node)}')
    assert not offenders, (
        'a value compared with itself asserts nothing:\n  '
        + '\n  '.join(offenders))


def test_no_boolean_operand_is_repeated():
    """`x and x` and `x or x` are the same defect wearing different syntax."""
    offenders = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp):
                continue
            seen = set()
            for value in node.values:
                if _is_call(value):
                    continue
                dumped = _key(value)
                if dumped in seen:
                    offenders.append(f'{path.relative_to(_ROOT)}:{node.lineno}  '
                                     f'{ast.unparse(node)}')
                    break
                seen.add(dumped)
    assert not offenders, (
        'a repeated boolean operand adds nothing:\n  ' + '\n  '.join(offenders))


def test_the_sweep_would_catch_the_three_it_was_written_for():
    """A guard that cannot fail is the thing it is guarding against.

    Each of these is one of the three real occurrences, reduced to its shape.
    """
    for snippet in ("if g['name'] == g.get('name'): pass",
                    'assert one == one',
                    'if flag and flag: pass'):
        tree = ast.parse(snippet)
        found = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Compare) and len(node.comparators) == 1
                    and _key(node.left) == _key(node.comparators[0])
                    and not _is_call(node.left)):
                found = True
            if isinstance(node, ast.BoolOp):
                dumps = [_key(v) for v in node.values if not _is_call(v)]
                if len(dumps) != len(set(dumps)):
                    found = True
        assert found, f'the sweep would not catch: {snippet}'


def test_a_determinism_assertion_is_not_flagged():
    """Six real call sites compare a function with itself on purpose. Flagging
    those would make the guard noise, and noise gets suppressed."""
    tree = ast.parse("assert seed(0, 'x') == seed(0, 'x')")
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            assert _is_call(node.left), 'a call must be exempt'
