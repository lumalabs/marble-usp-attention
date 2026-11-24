# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Build one isolated, pinned baseline environment on a Linux CUDA node.

Usage: python repro/setup_baseline.py flashmask --gpu h100 --root /tmp/baselines
Use --dry-run to inspect commands without cloning, installing, or writing files.
Existing target directories are refused; no checkout or environment is reset.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

TORCH = "torch==2.11.0+cu129"
CUDA_PYTHON = ["cuda-python==12.9.4", "cuda-bindings==12.9.7"]
INDEX = "https://download.pytorch.org/whl/cu129"
SOURCES = {
    "flashmask": ("https://github.com/PaddlePaddle/FlashMask.git", "6a150fc"),
    "magi": ("https://github.com/SandAI-org/MagiAttention.git", "de13b2a"),
}
PINS = {
    "flashmask": ["nvidia-cutlass==4.2.0.0", "nvidia-cutlass-dsl[cu13]==4.5.2"],
    "magi": ["nvidia-cutlass-dsl==4.4.2", "quack-kernels==0.4.1", "debugpy==1.8.20"],
    "te": ["transformer-engine[pytorch]==2.8.0"],
    "yunchang": ["yunchang==0.6.4", "flash-attn-3==3.0.0"],
}


def main() -> None:
    """Install a pinned baseline and save manifests using CLI options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
        subprocess.CalledProcessError: A required installation or
            environment-validation command fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=sorted(PINS))
    parser.add_argument("--gpu", choices=["h100", "b300"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter; Magi defaults to 3.12.14, TE to 3.10.19",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and sys.platform != "linux":
        parser.error(
            "GPU baseline installation requires Linux; use --dry-run to preview"
        )
    if args.baseline == "yunchang" and args.gpu == "b300":
        parser.error("the recorded yunchang FA3 baseline requires Hopper (H100)")
    target = args.root.resolve() / f"{args.baseline}-{args.gpu}"
    if target.exists():
        parser.error(f"target already exists; choose a fresh root: {target}")
    venv, source = target / "venv", target / "source"
    python = str(venv / "bin/python")
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
        "MAX_JOBS": os.environ.get("MAX_JOBS", "8"),
    }
    # Avoid the image's NCCL overriding torch's bundled runtime, as in the runs.
    env.pop("LD_PRELOAD", None)
    constraint = target / "constraints.txt"
    env["PIP_CONSTRAINT"] = str(constraint)
    env["UV_CONSTRAINT"] = str(constraint)
    print(f"Target: {target}; LD_PRELOAD unset; MAX_JOBS={env['MAX_JOBS']}")

    def run(command: list[str], *, cwd: Path | None = None) -> None:
        print(
            (f"cd {shlex.quote(str(cwd))} && " if cwd else "") + shlex.join(command),
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(command, cwd=cwd, env=env, check=True)

    if not args.dry_run:
        target.mkdir(parents=True)
        constraint.write_text(
            "\n".join(
                pin.split("[")[0] + "==" + pin.split("==")[1] if "[" in pin else pin
                for pin in [TORCH, *CUDA_PYTHON, *PINS[args.baseline]]
            )
            + "\n"
        )
    # TE 2.8's py3-none wheel contains a cp310-cp310 WHEEL tag. Match that
    # declared ABI instead of suppressing pip's platform consistency check.
    interpreter = args.python or {"magi": "3.12.14", "te": "3.10.19"}.get(
        args.baseline, sys.executable
    )
    run(["uv", "venv", "--seed", "--python", interpreter, str(venv)])
    run(["uv", "pip", "install", "--python", python, "--index-url", INDEX, TORCH])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "setuptools==80.9.0",
            "wheel==0.45.1",
            "packaging==25.0",
            "ninja==1.11.1.4",
            "cmake==3.31.6",
            "numpy==2.2.6",
            "pytest==8.4.2",
            *CUDA_PYTHON,
        ]
    )
    if args.baseline in SOURCES:
        url, ref = SOURCES[args.baseline]
        run(["git", "clone", "--no-checkout", url, str(source)])
        run(["git", "checkout", "--detach", ref], cwd=source)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
    if args.baseline == "flashmask":
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                *PINS["flashmask"],
                "apache-tvm-ffi==0.1.8.post0",
                "typing-extensions==4.15.0",
            ]
        )
        env.update(FLASHMASK_FRAMEWORK="torch", FLASHMASK_BUILD="core+cpb")
        # The overlap bridge needs a separately provisioned NCCL >= 2.30.5.
        # This recipe reproduces the public balanced all-gather baseline.
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "-e",
                str(source),
            ]
        )
        env["FLASHMASK_DIR"] = str(source)
        smoke = "import flashmask.cute.ops.fwd; import flashmask.cp_balance.cp_balance_cuda_kernels"
    elif args.baseline == "magi":
        # Explicit CUDA-12 opt-in for the requested torch cu129 stack.
        env["MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12"] = "1"
        # JIT kernels compile at benchmark warm-up; avoid prebuilding every
        # possible shape during installation. Python >=3.12 is required.
        env["MAGI_ATTENTION_PREBUILD_FFA"] = "0"
        # Upstream imports debugpy while prebuilding FFA, but omits it from
        # requirements.txt at the requested commit.
        run([python, "-m", "pip", "install", "debugpy==1.8.20"])
        run([python, "-m", "pip", "install", "-r", str(source / "requirements.txt")])
        if args.gpu == "b300":
            env.update(
                MAGI_ATTENTION_PREBUILD_FFA="0",
                MAGI_ATTENTION_FA4_BACKEND="1",
                MAGI_ATTENTION_KERNEL_BACKEND="fa4",
            )
            run(["bash", "scripts/install_flash_attn_cute.sh", "sm100"], cwd=source)
        run([python, "-m", "pip", "install", "--no-build-isolation", "-e", str(source)])
        env["MAGI_DIR"] = str(source)
        smoke = "import magi_attention"
    elif args.baseline == "te":
        env["NVTE_CUDA_ARCHS"] = "90" if args.gpu == "h100" else "103"
        # torch ships NCCL's headers/libraries under site-packages rather
        # than the system include path searched by TE's C++ extension.
        locate_nccl = "import nvidia.nccl; from pathlib import Path; print(Path(next(iter(nvidia.nccl.__path__))))"
        if args.dry_run:
            nccl = venv / "lib/pythonX.Y/site-packages/nvidia/nccl"
        else:
            nccl = Path(
                subprocess.check_output(
                    [python, "-c", locate_nccl], env=env, text=True
                ).strip()
            )
        env["CPATH"] = str(nccl / "include") + os.pathsep + env.get("CPATH", "")
        env["LIBRARY_PATH"] = (
            str(nccl / "lib") + os.pathsep + env.get("LIBRARY_PATH", "")
        )
        env["LD_LIBRARY_PATH"] = (
            str(nccl / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )
        run([python, "-m", "pip", "install", "--no-build-isolation", *PINS["te"]])
        smoke = "import transformer_engine.pytorch"
    else:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "flash-attn-3==3.0.0",
                "--index-url",
                INDEX,
                "--extra-index-url",
                "https://pypi.org/simple",
            ]
        )
        run([python, "-m", "pip", "install", "yunchang==0.6.4"])
        smoke = "import yunchang; import flash_attn_interface"
    run([python, "-c", smoke])
    run([python, "-m", "pip", "check"])
    # Fail rather than silently accept a transitive resolver changing core pins.
    pins = [TORCH, *CUDA_PYTHON, *PINS[args.baseline]]
    expected = {pin.split("==")[0].split("[")[0]: pin.split("==")[1] for pin in pins}
    verify = (
        "import importlib.metadata as m; "
        f"expected = {expected!r}; "
        "actual = {name: m.version(name) for name in expected}; "
        "mismatch = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}; "
        "\nif mismatch: raise RuntimeError(f'Baseline pins changed: {mismatch}')"
    )
    run([python, "-c", verify])
    if args.dry_run:
        return
    frozen = subprocess.check_output(
        [python, "-m", "pip", "freeze", "--all"], env=env, text=True
    )
    (target / "resolved-requirements.txt").write_text(frozen)
    exported = {
        key: env[key]
        for key in (
            "FLASHMASK_DIR",
            "FLASHMASK_FRAMEWORK",
            "MAGI_DIR",
            "MAGI_ATTENTION_KERNEL_BACKEND",
            "MAGI_ATTENTION_FA4_BACKEND",
            "NVTE_CUDA_ARCHS",
        )
        if key in env
    }
    (target / "activate.sh").write_text(
        f"source {shlex.quote(str(venv / 'bin/activate'))}\nunset LD_PRELOAD\n"
        + "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in exported.items()
        )
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from provenance import git_state  # standalone script, no package dependency

    (target / "setup.json").write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "gpu": args.gpu,
                "python": subprocess.check_output(
                    [python, "--version"], env=env, text=True
                ).strip(),
                "requested_pins": pins,
                "source": git_state(source) if source.exists() else None,
                "environment": exported,
                "build_environment": {
                    key: value
                    for key, value in env.items()
                    if key.startswith(("FLASHMASK_", "MAGI_ATTENTION_", "NVTE_"))
                    or key in ("MAX_JOBS", "CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH")
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Ready: source {target / 'activate.sh'}")


if __name__ == "__main__":
    main()
