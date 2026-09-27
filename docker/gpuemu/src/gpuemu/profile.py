"""``profile_plot`` - draw what a job did over its lifetime.

Slurm's own profiling records a time series rather than the single figures
``seff`` reports, and NeSI wrap the plotting of it in a ``profile_plot``
command. This is that command, reading the series the scheduler writes for any
job submitted with ``--profile task``.

The five panels, and their order, follow the cluster's output, because the
point of the exercise is that a learner can read the real thing afterwards.
The two GPU panels are boxed in red for the same reason.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from .slurm import JobStore, slurm_dir

__all__ = ["profile_plot", "main"]

# Panels, top to bottom: (column, y-label, is_gpu_panel)
_PANELS = [
    ("cpus", "CPUs", False),
    ("rss_mb", "Memory (MB)", False),
    ("io", "Cumulative I/O (MB)", False),
    ("gpu_util", "GPUs", True),
    ("gpu_mem_mb", "GPU Mem (MB)", True),
]


def profile_path(job_id: int) -> Path:
    return slurm_dir() / "profiles" / f"{job_id}.csv"


def read_series(job_id: int) -> list[dict[str, float]]:
    path = profile_path(job_id)
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                try:
                    rows.append({k: float(v) for k, v in raw.items() if v != ""})
                except (TypeError, ValueError):
                    continue
    except OSError:
        return []
    return rows


def _draw(job_id: int, rows: list[dict[str, float]], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t = [r.get("t", 0.0) for r in rows]

    fig, axes = plt.subplots(len(_PANELS), 1, figsize=(5.5, 10), sharex=True)
    fig.suptitle(title, fontsize=10)

    for ax, (key, label, _is_gpu) in zip(axes, _PANELS):
        if key == "io":
            ax.plot(t, [r.get("read_mb", 0.0) for r in rows], lw=1.2, label="Read")
            ax.plot(
                t,
                [r.get("write_mb", 0.0) for r in rows],
                lw=1.2,
                ls="--",
                label="Write",
            )
            ax.legend(fontsize=7, loc="upper left", frameon=False)
        else:
            ax.plot(t, [r.get(key, 0.0) for r in rows], lw=1.4)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)

    # GPU utilisation is a fraction of one card, as on the cluster, so the axis
    # is fixed at 0-1 rather than scaled to the data. A job idling at 0.02 that
    # filled the panel would read as a busy GPU.
    axes[3].set_ylim(0, 1)
    axes[-1].set_xlabel("seconds since the job started", fontsize=8)

    # Box the two GPU panels together. The layout has to be settled first:
    # tight_layout moves every axis, so positions read before it are wrong.
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.canvas.draw()
    top = axes[3].get_position()
    bot = axes[4].get_position()
    pad = 0.012
    fig.patches.append(
        Rectangle(
            (min(top.x0, bot.x0) - pad, bot.y0 - pad),
            max(top.x1, bot.x1) - min(top.x0, bot.x0) + 2 * pad,
            top.y1 - bot.y0 + 2 * pad,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#e8321e",
            lw=2,
            zorder=10,
        )
    )

    fig.savefig(out, dpi=110)
    plt.close(fig)


def profile_plot(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="profile_plot",
        description="Plot the resource usage recorded for a job over its runtime.",
    )
    ap.add_argument("jobid", help="job to plot")
    ap.add_argument("-o", "--output", default=None, help="output file")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    base = str(args.jobid).split(".")[0].split("_")[0]
    if not base.isdigit():
        print(f"profile_plot: '{args.jobid}' is not a job ID", file=sys.stderr)
        return 2
    job_id = int(base)

    job = JobStore().load(job_id)
    if job is None:
        print(f"profile_plot: job {job_id} not found.", file=sys.stderr)
        return 2

    rows = read_series(job_id)
    if not rows:
        print(
            f"profile_plot: no profile data for job {job_id}.\n"
            "  Profiling is off unless the job asked for it. Add to the script:\n"
            "      #SBATCH --profile    task\n"
            "      #SBATCH --acctg-freq 1\n"
            "  and submit it again.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.output) if args.output else Path.cwd() / f"{job_id}_profile.png"
    title = f"Job {job_id}  ({job.name})"
    if job.gpu_type:
        title += f"  -  {job.gpu_type}"
    try:
        _draw(job_id, rows, out, title)
    except Exception as exc:  # matplotlib is the only thing that can fail here
        print(f"profile_plot: could not draw the plot: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


def main() -> int:
    return profile_plot()


if __name__ == "__main__":
    raise SystemExit(main())
