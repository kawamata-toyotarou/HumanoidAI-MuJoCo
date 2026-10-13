from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_phase4 import (  # noqa: E402
    PRESETS,
    apply_model_learning_rate,
    metadata_path_for_model,
    optimizer_learning_rates,
    resolve_learning_rate,
)


def run_help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.stdout


def main() -> None:
    class Args:
        learning_rate = None

    class Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-4}, {"lr": 1e-4}]

    class Policy:
        def __init__(self) -> None:
            self.optimizer = Optimizer()

    class FakeModel:
        def __init__(self) -> None:
            self.policy = Policy()
            self.learning_rate = 1e-4
            self.lr_schedule = lambda _: 1e-4

    assert PRESETS["baseline"].n_steps == 512
    assert PRESETS["baseline"].batch_size == 64
    assert PRESETS["baseline"].n_epochs == 5
    assert PRESETS["baseline"].learning_rate == 3e-4
    assert PRESETS["baseline"].net_arch == [64, 64]
    assert PRESETS["candidate"].n_steps == 1024
    assert PRESETS["candidate"].batch_size == 256
    assert PRESETS["candidate"].n_epochs == 10
    assert PRESETS["candidate"].learning_rate == 1e-4
    assert PRESETS["candidate"].ent_coef == 0.01
    assert PRESETS["candidate"].net_arch == [256, 256]
    assert resolve_learning_rate(Args(), PRESETS["candidate"]) == 1e-4
    Args.learning_rate = 5e-5
    assert resolve_learning_rate(Args(), PRESETS["candidate"]) == 5e-5
    fake_model = FakeModel()
    apply_model_learning_rate(fake_model, 5e-5)
    assert fake_model.learning_rate == 5e-5
    assert fake_model.lr_schedule(1.0) == 5e-5
    assert optimizer_learning_rates(fake_model) == [5e-5, 5e-5]

    assert metadata_path_for_model(Path("model.zip")) == Path("model_metadata.json")
    assert metadata_path_for_model(Path("model")) == Path("model_metadata.json")

    train_help = run_help("train_phase4.py")
    for flag in [
        "--preset",
        "--total-timesteps",
        "--seed",
        "--n-envs",
        "--device",
        "--checkpoint-dir",
        "--run-name",
        "--resume",
        "--vec-normalize-stats",
        "--save-freq",
        "--torch-threads",
        "--learning-rate",
    ]:
        assert flag in train_help

    eval_help = run_help("evaluate_phase4.py")
    for flag in ["--model", "--vec-normalize-stats", "--episodes", "--seed", "--device"]:
        assert flag in eval_help
    print("phase4_cli_tests_done=true")


if __name__ == "__main__":
    main()
