#!/usr/bin/env python3
"""Pin the rule that a missing optional dependency may not delete tests.

`experiments/dfao_min_states.py` guarded its optional pysat dependency with
`raise SystemExit(...)` at module scope. `SystemExit` does not inherit from
`Exception`, so the `except Exception -> unittest.SkipTest` guards in
`tests/test_dfao_drat_proofs.py` and `tests/test_grammar_min_size.py` could not
catch it. unittest's loader special-cases `SkipTest` but not `SystemExit`, so
each of those modules collapsed to a single failed-import placeholder instead
of contributing its tests.

Measured on 2026-08-30 with pysat blocked: **65 tests ran instead of 88**. The
23 that vanished are the ones checking *which* instances the s*(n) certificate
is required to prove - the coverage logic that stops a certificate reporting
"100% verified" over a silently smaller set. None of them needed a solver.

The failure is invisible: the run says FAILED with 2 errors, which reads like
two broken tests rather than twenty-three absent ones.

This test is a static scan, so it costs nothing and runs everywhere.
"""

import ast
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = ("experiments", "tools", "gpu")


def _module_level_systemexit(tree: ast.Module) -> list[int]:
    """Line numbers of `raise SystemExit` reachable at import time.

    Bodies of function and class definitions are not import-time, and neither
    is an `if __name__ == "__main__":` block, which is the legitimate place to
    exit with a status. Everything else at module scope is.
    """
    bad: list[int] = []

    def is_main_guard(node: ast.stmt) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        return (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__")

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue  # not executed at import
            if is_main_guard(node):
                continue  # the sanctioned place to exit
            if isinstance(node, ast.Raise):
                exc = node.exc
                name = None
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    name = exc.func.id
                elif isinstance(exc, ast.Name):
                    name = exc.id
                if name == "SystemExit":
                    bad.append(node.lineno)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    walk([child])
                else:
                    for sub in ast.iter_child_nodes(child):
                        if isinstance(sub, ast.stmt):
                            walk([sub])

    walk(tree.body)
    return bad


class TestNoImportTimeSystemExit(unittest.TestCase):

    def test_no_module_level_system_exit(self):
        offenders = []
        for directory in SCAN_DIRS:
            root = REPO_ROOT / directory
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
                for lineno in _module_level_systemexit(tree):
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}")
        self.assertEqual(offenders, [], (
            "`raise SystemExit` at module scope deletes tests instead of "
            "skipping them: SystemExit is not an Exception, so importing "
            "modules cannot guard it, and unittest reduces the whole module "
            "to one failed-import placeholder.\n"
            "Set an availability flag at import and raise SystemExit inside "
            "main() instead. Offenders:\n  " + "\n  ".join(offenders)))

    def test_scan_reproduces_the_historical_failure(self):
        """A lint that would not have caught its own incident is worth nothing."""
        historical = ast.parse(
            "import unittest\n"
            "try:\n"
            "    import pysat\n"
            "except ImportError as exc:\n"
            "    raise SystemExit('python-sat is required') from exc\n"
            "def main():\n"
            "    raise SystemExit(0)\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n")
        found = _module_level_systemexit(historical)
        self.assertEqual(found, [5], "must flag the import guard on line 5")

    def test_scan_allows_main_guard_and_function_bodies(self):
        clean = ast.parse(
            "def main():\n"
            "    raise SystemExit('bad args')\n"
            "class C:\n"
            "    def go(self):\n"
            "        raise SystemExit(2)\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n")
        self.assertEqual(_module_level_systemexit(clean), [])


if __name__ == "__main__":
    unittest.main()
