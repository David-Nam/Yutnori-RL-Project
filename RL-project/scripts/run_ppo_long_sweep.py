"""Run long PPO training sweeps across baseline opponents and seeds."""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import shutil
import struct
import subprocess
import sys
import termios
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from yutnori.env import (  # noqa: E402
    OBSERVATION_MODE_BASE,
    OBSERVATION_MODES,
    REWARD_MODE_TERMINAL,
    REWARD_MODES,
)
from yutnori.training.env_factory import OPPONENT_NAMES  # noqa: E402
from yutnori.training.env_factory import VEC_ENV_TYPES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=OPPONENT_NAMES,
        default=list(OPPONENT_NAMES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default="base")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="terminal")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--timesteps-label", default=None)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--vec-env", choices=VEC_ENV_TYPES, default="dummy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--train-eval-episodes", type=int, default=100)
    parser.add_argument("--final-eval-episodes", type=int, default=10_000)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--logs-root", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)

    timesteps_label = args.timesteps_label or _timesteps_label(args.total_timesteps)
    for opponent in args.opponents:
        for seed in args.seeds:
            run_name = _run_name(
                opponent,
                seed,
                timesteps_label,
                args.n_envs,
                args.observation_mode,
                args.reward_mode,
            )
            run_dir = args.runs_root / run_name
            log_path = args.logs_root / f"{run_name}.log"
            _run_one(args, opponent, seed, run_dir, log_path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.n_steps <= 1:
        raise ValueError("n_steps must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.train_eval_episodes < 0:
        raise ValueError("train_eval_episodes must be non-negative")
    if args.final_eval_episodes < 0:
        raise ValueError("final_eval_episodes must be non-negative")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint_freq must be non-negative")


def _run_one(
    args: argparse.Namespace,
    opponent: str,
    seed: int,
    run_dir: Path,
    log_path: Path,
) -> None:
    completed = _training_completed(run_dir)
    if completed and not args.overwrite:
        print(f"skip training: completed run exists at {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        message = (
            f"run directory is not empty and has no completed model: {run_dir}. "
            "Use --overwrite to rerun or --skip-existing to ignore it."
        )
        if args.skip_existing:
            print(f"skip training: {message}")
            return
        raise FileExistsError(message)
    else:
        train_command = _train_command(args, opponent, seed, run_dir)
        _run_command(train_command, log_path, dry_run=args.dry_run)

    if args.skip_final_eval or args.final_eval_episodes == 0:
        return
    model_path = run_dir / "model.zip"
    if args.dry_run:
        model_path = run_dir / "model.zip"
    elif not model_path.exists():
        raise FileNotFoundError(f"cannot run final eval without model: {model_path}")

    for eval_opponent in OPPONENT_NAMES:
        output_path = run_dir / f"eval_{eval_opponent}_{args.final_eval_episodes}.json"
        if output_path.exists() and not args.overwrite:
            print(f"skip eval: existing output {output_path}")
            continue
        eval_command = _eval_command(args, model_path, eval_opponent, seed, output_path)
        _run_command(eval_command, log_path, dry_run=args.dry_run)


def _training_completed(run_dir: Path) -> bool:
    return (run_dir / "model.zip").exists() and (run_dir / "summary.json").exists()


def _train_command(
    args: argparse.Namespace,
    opponent: str,
    seed: int,
    run_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_ppo.py"),
        "--total-timesteps",
        str(args.total_timesteps),
        "--seed",
        str(seed),
        "--opponent",
        opponent,
        "--observation-mode",
        args.observation_mode,
        "--reward-mode",
        args.reward_mode,
        "--n-envs",
        str(args.n_envs),
        "--vec-env",
        args.vec_env,
        "--device",
        args.device,
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--ent-coef",
        str(args.ent_coef),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--run-dir",
        str(run_dir),
        "--eval-episodes",
        str(args.train_eval_episodes),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.no_progress_bar:
        command.append("--no-progress-bar")
    return command


def _eval_command(
    args: argparse.Namespace,
    model_path: Path,
    opponent: str,
    seed: int,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "evaluate_ppo.py"),
        "--model-path",
        str(model_path),
        "--opponent",
        opponent,
        "--observation-mode",
        args.observation_mode,
        "--reward-mode",
        args.reward_mode,
        "--episodes",
        str(args.final_eval_episodes),
        "--seed",
        str(seed + 100_000),
        "--device",
        args.device,
        "--output",
        str(output_path),
    ]
    if args.no_progress_bar:
        command.append("--no-progress-bar")
    return command


def _run_command(command: list[str], log_path: Path, *, dry_run: bool) -> None:
    rendered = " ".join(command)
    header = f"\n[{datetime.now(UTC).isoformat()}] $ {rendered}\n"
    print(header, end="", flush=True)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        log_file.write(header.encode())
        log_file.flush()
        return_code = _run_command_in_pty(command, log_file)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _run_command_in_pty(command: list[str], log_file) -> int:
    master_fd, slave_fd = pty.openpty()
    _set_pty_window_size(slave_fd)
    progress_log = _CarriageReturnLogFilter(log_file)
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    try:
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            progress_log.write(data)
    finally:
        progress_log.close()
        os.close(master_fd)

    return process.wait()


class _CarriageReturnLogFilter:
    """Collapse terminal redraws before writing persistent log lines."""

    def __init__(self, log_file) -> None:
        self._log_file = log_file
        self._line = bytearray()
        self._pending_cr = False

    def write(self, data: bytes) -> None:
        for byte in data:
            self._consume_byte(byte)
        self._log_file.flush()

    def close(self) -> None:
        self._pending_cr = False
        if self._line:
            self._write_line()
        self._log_file.flush()

    def _consume_byte(self, byte: int) -> None:
        if self._pending_cr:
            if byte == ord("\n"):
                self._write_line()
                self._pending_cr = False
                return
            self._line.clear()
            self._pending_cr = False

        if byte == ord("\r"):
            self._pending_cr = True
        elif byte == ord("\n"):
            self._write_line()
        elif byte == ord("\b"):
            if self._line:
                self._line.pop()
        else:
            self._line.append(byte)

    def _write_line(self) -> None:
        self._log_file.write(bytes(self._line))
        self._log_file.write(b"\n")
        self._line.clear()


def _set_pty_window_size(slave_fd: int) -> None:
    size = shutil.get_terminal_size(fallback=(120, 40))
    winsize = struct.pack("HHHH", size.lines, size.columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)


def _timesteps_label(total_timesteps: int) -> str:
    if total_timesteps % 1_000_000 == 0:
        return f"{total_timesteps // 1_000_000}m"
    if total_timesteps % 1_000 == 0:
        return f"{total_timesteps // 1_000}k"
    return str(total_timesteps)


def _run_name(
    opponent: str,
    seed: int,
    timesteps_label: str,
    n_envs: int,
    observation_mode: str,
    reward_mode: str,
) -> str:
    suffixes = []
    if observation_mode != OBSERVATION_MODE_BASE:
        suffixes.append(observation_mode)
    if reward_mode != REWARD_MODE_TERMINAL:
        suffixes.append(reward_mode)
    mode_suffix = "" if not suffixes else f"_{'_'.join(suffixes)}"
    return f"{opponent}_seed_{seed}_{timesteps_label}_nenv{n_envs}{mode_suffix}"


if __name__ == "__main__":
    main()
