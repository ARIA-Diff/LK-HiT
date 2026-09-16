"""Launch the paper's seed sweeps from the composed experiment configs.

    python scripts/run_sweep.py --suite main
    python scripts/run_sweep.py --suite ablations --seeds 1 2 3 4 5
    python scripts/run_sweep.py --task ecthr_a --model lkhit --seeds 1

Each job is ``python -m lkhit.cli.train --config ... --seed k``. Failed jobs
are recorded and the sweep continues; the process exits non-zero if any job
failed. ``--dry-run`` prints the commands without starting training.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments"
SEEDS = (1, 2, 3, 4, 5)
TASKS = ("ecthr_a", "ecthr_b", "eurlex", "cail2018")
ABLATION_TASKS = ("ecthr_a", "eurlex", "cail2018")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["main", "ablations", "zero_shot", "one"], default="main")
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--model", default=None, help="config stem under configs/experiments/<task>/")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="forwarded to lkhit.cli.train after --")
    return parser.parse_args()


def config_stems(task: str, suite: str) -> list[str]:
    directory = CONFIG_ROOT / task
    stems = sorted(p.stem for p in directory.glob("*.yaml"))
    if suite == "main":
        return [s for s in stems if not s.startswith("lkhit_") and s != "qwen_zero_shot"]
    if suite == "ablations":
        return [s for s in stems if s.startswith("lkhit_")]
    if suite == "zero_shot":
        return [s for s in stems if s == "qwen_zero_shot"]
    return stems


def jobs_for(args: argparse.Namespace) -> list[tuple[Path, int, list[str]]]:
    if args.suite == "one":
        if not args.task or not args.model:
            raise SystemExit("--suite one needs --task and --model")
        tasks = [args.task]
        stems = [args.model]
    else:
        tasks = [args.task] if args.task else (list(ABLATION_TASKS) if args.suite == "ablations" else list(TASKS))
        stems = None
    jobs = []
    for task in tasks:
        names = [args.model] if args.model else config_stems(task, args.suite)
        for name in names:
            config = CONFIG_ROOT / task / f"{name}.yaml"
            if not config.exists():
                raise FileNotFoundError(config)
            if name == "qwen_zero_shot":
                cmd = [args.python, "-m", "lkhit.cli.zero_shot_llm", "--config", str(config)]
                jobs.append((config, 1, cmd))
                continue
            for seed in args.seeds:
                cmd = [args.python, "-m", "lkhit.cli.train", "--config", str(config), "--seed", str(seed)]
                if args.extra:
                    cmd.extend(args.extra)
                jobs.append((config, seed, cmd))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = jobs_for(args)
    print(f"{len(jobs)} jobs", flush=True)
    failed = []
    for i, (config, seed, cmd) in enumerate(jobs, start=1):
        display = " ".join(cmd)
        print(f"[{i}/{len(jobs)}] {display}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append((config, seed, result.returncode))
            print(f"FAILED {config.name} seed{seed} exit={result.returncode}", flush=True)
    if failed:
        print(f"{len(failed)} failed jobs:", flush=True)
        for config, seed, code in failed:
            print(f"  {config} seed{seed} -> {code}", flush=True)
        raise SystemExit(1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
