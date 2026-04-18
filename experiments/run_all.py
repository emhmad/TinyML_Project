from __future__ import annotations

import argparse

from experiments import (
    e1_finetune,
    e2_baseline_eval,
    e3_calibration,
    e4_pruning_matrix,
    e5_perlayer_breakdown,
    e6_diagnostic_safety,
    e7_e10_nonuniform,
    e11_e13_quantization,
    e14_e16_distillation,
)


def run(config_path: str, pillars: list[int]) -> None:
    if 0 in pillars:
        e1_finetune.run(config_path)
        e2_baseline_eval.run(config_path)
        e3_calibration.run(config_path)

    if 1 in pillars:
        e4_pruning_matrix.run(config_path)
        e5_perlayer_breakdown.run(config_path)
        e6_diagnostic_safety.run(config_path)

    if 2 in pillars:
        e7_e10_nonuniform.run(config_path)

    if 3 in pillars:
        e11_e13_quantization.run(config_path)

    if 4 in pillars:
        e14_e16_distillation.run(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the implemented TinyML experiment pillars.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pillars", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.pillars)


if __name__ == "__main__":
    main()
