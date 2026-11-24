# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Check every command a sweep would launch against its driver's argument parser.

Runs on a laptop without GPUs: ``sweep.py <sweep> --dry-run`` prints the driver
commands, and this script feeds each one's arguments to the driver's
``_parse_args`` so a flag the driver does not know (or a wrong value) fails
here instead of on the cluster.  Usage::

    python benchmarks/preflight.py oom32 chunks32 collectives16
    python benchmarks/preflight.py --all-paper
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _driver_args(cmd: list[str]) -> tuple[Path, list[str]]:
    """Split a torchrun command into (driver script, driver arguments)."""
    for i, tok in enumerate(cmd):
        if tok.endswith(".py") and HERE in Path(tok).parents:
            return Path(tok), cmd[i + 1 :]
    raise ValueError(f"no driver script in {cmd}")


def _parser_of(script: Path):
    """Return the driver's ``_parse_args`` without running the driver.

    Imports the module when its dependencies exist locally; otherwise (the
    baseline drivers import their own environment) executes only the
    ``_parse_args`` function's source, which must then depend on argparse alone.
    """
    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    for d in (HERE, HERE.parent / "src"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    # Registered before execution, as importlib's recipe does: a dataclass
    # under ``from __future__ import annotations`` (bench_model.py) resolves
    # its field types through ``sys.modules[cls.__module__]``.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        return mod._parse_args
    except ImportError:
        sys.modules.pop(spec.name, None)
        tree = ast.parse(script.read_text())
        fn = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_parse_args"
            ),
            None,
        )
        if fn is None:
            fn = _parser_from_main(tree)

        # Names the parser body reads from the module (help strings such as
        # ``f"one of {LAYOUT_NAMES}"``) resolve to a placeholder: only the
        # argument grammar matters here, not the help text.
        class _Loose(dict):
            def __missing__(self, key: str) -> object:
                return getattr(builtins, key, f"<{key}>")

        ns = _Loose({"argparse": argparse, "__doc__": ast.get_docstring(tree) or ""})
        exec(compile(ast.Module([fn], []), str(script), "exec"), ns)
        return ns["_parse_args"]


def _parser_from_main(tree: ast.Module) -> ast.FunctionDef:
    """Synthesise ``_parse_args`` from a driver whose ``main`` builds the parser inline.

    Takes ``main``'s statements up to the one that calls ``parse_args`` and
    returns the name that call is assigned to.

    Args:
        tree: The driver module's AST.

    Returns:
        A function definition equivalent to the parser-building prefix of ``main``.

    Raises:
        ValueError: ``main`` is missing or never calls ``parse_args``.
    """
    main = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    if main is None:
        raise ValueError("driver has neither _parse_args nor main")
    body: list[ast.stmt] = []
    for stmt in main.body:
        body.append(stmt)
        calls = [
            c
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "parse_args"
        ]
        if (
            calls
            and isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
        ):
            body.append(ast.Return(ast.Name(stmt.targets[0].id, ast.Load())))
            fn = ast.FunctionDef(
                name="_parse_args",
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=body,
                decorator_list=[],
                returns=None,
            )
            return ast.fix_missing_locations(ast.copy_location(fn, main))
    raise ValueError("main never assigns parse_args()")


def check(sweep: str, nnodes: int) -> int:
    dry = subprocess.run(
        [
            sys.executable,
            str(HERE / "sweep.py"),
            sweep,
            "--out",
            "/dev/null",
            "--nnodes",
            str(nnodes),
            "--master-addr",
            "x",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    bad = 0
    parsers: dict[Path, object] = {}
    for line in dry.stdout.splitlines():
        cmd = json.loads(line)
        script, argv = _driver_args(cmd)
        parse = parsers.setdefault(script, _parser_of(script))
        sys.argv = [str(script), *argv]
        try:
            parse()
        except SystemExit as e:
            bad += 1
            print(f"FAIL {sweep} {script.name} {' '.join(argv)} (exit {e.code})")
            continue
        # argparse keeps the last of a repeated option silently; a repeated
        # --label would make rows collide.
        flags = [a for a in argv if a.startswith("--")]
        dup = sorted({f for f in flags if flags.count(f) > 1})
        if dup:
            bad += 1
            print(f"FAIL {sweep} repeated {dup}: {' '.join(argv)}")
        # The H100 default of --dense-ring is "auto" (= off); a ring config
        # without an explicit choice would silently run the block-sparse ring.
        if "ring" in argv and "--kv-exchange" in argv and "--dense-ring" not in argv:
            print(
                f"WARN {sweep} ring config without --dense-ring: {' '.join(argv[:2])}"
            )
    print(f"{sweep}: {len(dry.stdout.splitlines())} commands, {bad} rejected")
    return bad


def main() -> None:
    sys.path.insert(0, str(HERE))
    from sweep import PAPER_WORLDS  # noqa: E402

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("sweeps", nargs="*")
    p.add_argument("--all-paper", action="store_true", help="Every PAPER_WORLDS sweep.")
    args = p.parse_args()
    names = list(PAPER_WORLDS) if args.all_paper else args.sweeps
    total = sum(check(n, max(1, PAPER_WORLDS.get(n, 8) // 8)) for n in names)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
