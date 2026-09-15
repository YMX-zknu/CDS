import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "synapse": ("section_2_1/synaptic_transmission.py", "Single-synapse transmission example"),
    "calcium": ("section_2_2/calcium_dynamics.py", "Calcium dynamics and parameter scans"),
    "selectivity": ("section_2_3/temporal_selectivity.py", "Temporal transmission selectivity"),
    "lif": ("section_2_4/coupled_lif_dynamics.py", "Coupled CDS-LIF dynamics"),
    "sequences": ("section_2_5/sequence_preservation.py", "Structured-sequence preservation"),
    "train": ("section_2_6/run_section_2_6_multiseed.py", "Train all three tasks across seeds"),
    "train-dvs": ("section_2_6/section_2_6_dvsgesture_train.py", "Train DVS-Gesture"),
    "train-shd": ("section_2_6/section_2_6_shd_train.py", "Train SHD"),
    "train-stmnist": ("section_2_6/section_2_6_stmnist_train.py", "Train ST-MNIST"),
    "ablate": ("section_2_6/section_2_6_cds_ablation.py", "Train component ablations"),
    "plot-tasks": ("section_2_6/section_2_6_multimodal_plot.py", "Plot task metrics from CSV files"),
    "visualize": ("section_2_6/plot_fig6g_cross_modal_inputs.py", "Export twelve input TIFF images"),
    "analyze": ("section_2_7/run_section_2_7_multiseed.py", "Analyze all three tasks across seeds"),
    "analyze-dvs": ("section_2_7/section_2_7_dvsgesture_analysis.py", "Analyze DVS-Gesture checkpoints"),
    "analyze-shd": ("section_2_7/section_2_7_shd_analysis.py", "Analyze SHD checkpoints"),
    "analyze-stmnist": ("section_2_7/section_2_7_stmnist_analysis.py", "Analyze ST-MNIST checkpoints"),
    "plot-propagation": ("section_2_7/section_2_7_multimodal_plot.py", "Plot propagation metrics from CSV files"),
}
MECHANISM_COMMANDS = ("synapse", "calcium", "selectivity", "lif", "sequences")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="CDS experiments. Run from the repository root.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name:20s}{desc}" for name, (_, desc) in COMMANDS.items())
        + "\n\nUse python run.py COMMAND --help for experiment options.",
    )
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    selected = parser.parse_args(args[:1]).command
    forwarded = args[1:]
    if selected in MECHANISM_COMMANDS and forwarded:
        if forwarded in (["--help"], ["-h"]):
            print(f"{COMMANDS[selected][1]}. No options. Outputs: results/section_2_{MECHANISM_COMMANDS.index(selected) + 1}/")
            return 0
        parser.error(f"{selected} accepts no options")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    try:
        return subprocess.run(
            [sys.executable, str(ROOT / "experiments" / COMMANDS[selected][0]), *forwarded],
            env=env,
            check=False,
        ).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
