"""Pre-recorded jobs for the "putting it all together" chapter.

The chapter asks a learner to compare the same job across five GPUs, and then
across five core counts. Running those ten jobs live would take most of a
session and, worse, would be measured against the emulated card - which reports
1 GB whatever you ask for, so every card would look identical and the exercise
would teach nothing.

So they are recorded instead: real job records, written once at session start,
carrying the figures a cluster would have produced. ``seff`` and
``profile_plot`` read them exactly as they read a job you ran yourself.

Nothing here is a fake tool. The tools are real; only the jobs are.
"""

from __future__ import annotations

import time
from pathlib import Path

from .slurm import COMPLETED, FAILED, Job, JobStore, slurm_dir

__all__ = ["seed", "main", "BY_CARD", "BY_CORES"]

MB = 1024

# The cards, as the cluster has them.
CARDS = {
    "l4": 24 * MB,
    "a100_40": 40 * MB,
    "a100": 80 * MB,
    "h100": 94 * MB,
    "pro_6000": 96 * MB,
}

# The job needs about 30 GB, so it does not fit an L4 and fits everything else.
# That single fact is the whole of exercise 2.
NEEDS_MB = 30.1 * MB

# (job_id, card)
BY_CARD = [
    (2001001, "l4"),
    (2001002, "a100_40"),
    (2001003, "a100"),
    (2001004, "h100"),
    (2001005, "pro_6000"),
]

# (job_id, cores, cpu_efficiency_pct, gpu_util_pct) - the curve from the
# "how many CPUs" chapter: utilisation climbs steeply, then flattens at four.
BY_CORES = [
    (2002001, 1, 100.0, 24.0),
    (2002002, 2, 81.0, 31.0),
    (2002003, 4, 64.0, 35.0),
    (2002004, 8, 38.0, 36.0),
    (2002005, 16, 20.0, 36.0),
]

GPU_WORK_S = 210.0  # fixed work on the card; wall-time follows from utilisation
TIME_LIMIT_S = 15 * 60
ACCOUNT = "nesi99991"


def _rss_mb(cores: int) -> float:
    """Peak CPU memory for a given core count.

    Memory grows with cores because each one holds batches in flight, which is
    the point the memory chapter makes and the reason exercise 3b says to
    measure at the core count you actually settled on.
    """
    return (2.0 + 1.05 * cores) * MB


def _job(
    job_id: int,
    name: str,
    *,
    cores: int,
    card: str,
    state: str,
    wall_s: float,
    cpu_seconds: float,
    gpu_util_pct: float,
    gpu_mem_mb: float,
    mem_mb: int,
    exit_code: int = 0,
    age_s: float = 3600.0,
) -> Job:
    end = time.time() - age_s
    start = end - wall_s
    return Job(
        job_id=job_id,
        name=name,
        user="trainee",
        script=f"~/gpu-training/07_putting_it_together/{name}.sl",
        workdir=str(Path.home() / "gpu-training" / "07_putting_it_together"),
        stdout=f"{name}-{job_id}.out",
        stderr=f"{name}-{job_id}.out",
        account=ACCOUNT,
        state=state,
        cpus=cores,
        mem_mb=mem_mb,
        gpus=1,
        gpu_ids=[0],
        ntasks=1,
        time_limit_s=TIME_LIMIT_S,
        submit_time=start - 20,
        start_time=start,
        end_time=end,
        exit_code=exit_code,
        qos="debug",
        profile="task",
        acctg_freq=1,
        cpu_seconds=cpu_seconds,
        max_rss_mb=_rss_mb(cores),
        gpu_util_sum=gpu_util_pct,
        gpu_util_samples=1,
        gpu_mem_peak_mb=gpu_mem_mb,
        gpu_type=card,
        gpu_total_mb=float(CARDS[card]),
    )


def _series(job: Job, gpu_util_pct: float, gpu_mem_mb: float, failed_at: float | None):
    """A plausible time series for a recorded job.

    Shapes rather than noise: everything ramps over the first few samples and
    then holds, except cumulative I/O which climbs all the way and GPU memory
    which is allocated once and never moves. A learner reading the plot should
    see the same story seff tells, with the addition of *when*.
    """
    wall = job.elapsed
    step = max(1, job.acctg_freq)
    rows = ["t,cpus,rss_mb,read_mb,write_mb,gpu_util,gpu_mem_mb"]
    t = 0.0
    peak_rss = job.max_rss_mb
    cpus_steady = job.cpus * (job.cpu_seconds / (wall * job.cpus)) if wall else 0.0
    while t <= wall:
        ramp = min(1.0, t / 25.0) if t < 25.0 else 1.0
        if failed_at is not None and t >= failed_at:
            break
        rows.append(
            ",".join(
                [
                    f"{t:.1f}",
                    f"{cpus_steady * ramp:.2f}",
                    f"{peak_rss * ramp:.1f}",
                    f"{60.0 * ramp:.1f}",
                    f"{0.9 * t:.1f}",
                    f"{(gpu_util_pct / 100.0) * ramp:.3f}",
                    f"{gpu_mem_mb * ramp:.1f}",
                ]
            )
        )
        t += step
    return "\n".join(rows) + "\n"


def seed(force: bool = False) -> int:
    """Write the recorded jobs. Returns how many were written."""
    store = JobStore()
    prof_dir = slurm_dir() / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for job_id, card in BY_CARD:
        if store.load(job_id) is not None and not force:
            continue
        fits = CARDS[card] > NEEDS_MB
        wall = GPU_WORK_S / 0.35
        if fits:
            job = _job(
                job_id,
                f"fit-test-{card}",
                cores=4,
                card=card,
                state=COMPLETED,
                wall_s=wall,
                cpu_seconds=wall * 4 * 0.64,
                gpu_util_pct=35.0,
                gpu_mem_mb=NEEDS_MB,
                mem_mb=32 * MB,
            )
            series = _series(job, 35.0, NEEDS_MB, None)
        else:
            # Ran until the card filled, then raised CUDA OOM.
            died = 48.0
            job = _job(
                job_id,
                f"fit-test-{card}",
                cores=4,
                card=card,
                state=FAILED,
                wall_s=died,
                cpu_seconds=died * 4 * 0.55,
                gpu_util_pct=31.0,
                gpu_mem_mb=CARDS[card] * 0.985,
                mem_mb=32 * MB,
                exit_code=1,
            )
            series = _series(job, 31.0, CARDS[card] * 0.985, died)
        store.save(job)
        (prof_dir / f"{job_id}.csv").write_text(series, encoding="utf-8")
        written += 1

    for job_id, cores, cpu_pct, gpu_pct in BY_CORES:
        if store.load(job_id) is not None and not force:
            continue
        wall = GPU_WORK_S / (gpu_pct / 100.0)
        job = _job(
            job_id,
            f"core-scan-{cores:02d}",
            cores=cores,
            card="a100_40",
            state=COMPLETED,
            wall_s=wall,
            cpu_seconds=wall * cores * (cpu_pct / 100.0),
            gpu_util_pct=gpu_pct,
            gpu_mem_mb=NEEDS_MB,
            mem_mb=32 * MB,
        )
        store.save(job)
        (prof_dir / f"{job_id}.csv").write_text(
            _series(job, gpu_pct, NEEDS_MB, None), encoding="utf-8"
        )
        written += 1

    return written


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="gpuemu-seed-workshop",
        description="Write the pre-recorded jobs the workshop's final chapter uses.",
    )
    ap.add_argument(
        "-f", "--force", action="store_true", help="rewrite jobs that already exist"
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    n = seed(force=args.force)
    if not args.quiet:
        print(f"Seeded {n} recorded job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
