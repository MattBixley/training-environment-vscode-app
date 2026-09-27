"""Slurm native profiling, and the pre-recorded jobs the final chapter reads.

These cover the two things the workshop's last stage depends on: that a script
copied out of the documentation runs at all, and that the ten recorded jobs
report the figures the chapter tells a learner they will see. If those figures
drift, the exercises quietly stop making sense - the answer is still "take the
40 GB card", but the numbers on screen no longer say so.
"""

from __future__ import annotations

import pytest

from gpuemu import seed as seedmod
from gpuemu.profile import profile_plot, read_series
from gpuemu.slurm import JobStore, _acctg_seconds, sbatch, seff


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUEMU_STATE_FILE", str(tmp_path / "state.bin"))
    monkeypatch.setenv("GPUEMU_CLAIMS_DIR", str(tmp_path / "claims"))
    monkeypatch.setenv("GPUEMU_SLURM_DIR", str(tmp_path / "slurm"))


# --------------------------------------------------------------- --acctg-freq


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 30),  # Slurm's own default for --profile task
        ("1", 1),
        ("30", 30),
        ("task=1", 1),
        ("energy=60,task=1", 1),  # task wins when it is named
        ("task=5,energy=60", 5),
        ("0", 1),  # never zero: it is used as a divisor
        ("nonsense", 30),
    ],
)
def test_acctg_freq_accepts_every_documented_spelling(raw, expected):
    assert _acctg_seconds(raw) == expected


def test_sbatch_accepts_the_documented_test_script(tmp_path, capsys):
    """A script copied from the chapter must submit, not fail on an option.

    --acctg-freq used to be absent from the parser, so the very script the
    documentation tells people to write errored out.
    """
    script = tmp_path / "test-job.sl"
    script.write_text(
        "#!/bin/bash -e\n"
        "#SBATCH --time          00:15:00\n"
        "#SBATCH --qos           debug\n"
        "#SBATCH --profile       task\n"
        "#SBATCH --acctg-freq    1\n"
        "#SBATCH --cpus-per-task 2\n"
        "true\n",
        encoding="utf-8",
    )
    assert sbatch(["--parsable", str(script)]) == 0
    job_id = int(capsys.readouterr().out.strip())

    job = JobStore().load(job_id)
    assert job is not None
    assert job.qos == "debug"
    assert job.profile == "task"
    assert job.acctg_freq == 1


# ------------------------------------------------------------ recorded jobs


def test_seeding_is_idempotent():
    assert seedmod.seed() == 10
    assert seedmod.seed() == 0, "a second session must not duplicate the jobs"
    assert seedmod.seed(force=True) == 10


def test_only_the_l4_job_failed():
    seedmod.seed()
    store = JobStore()
    states = {card: store.load(jid).state for jid, card in seedmod.BY_CARD}
    assert states["l4"] == "FAILED"
    assert all(s == "COMPLETED" for c, s in states.items() if c != "l4")


def test_card_percentages_match_the_chapter():
    """Exercise 2's answer table, to the whole percent."""
    seedmod.seed()
    store = JobStore()
    expected = {"a100_40": 75, "a100": 38, "h100": 32, "pro_6000": 31}
    for job_id, card in seedmod.BY_CARD:
        if card == "l4":
            continue
        job = store.load(job_id)
        pct = round(100 * job.gpu_mem_peak_mb / job.gpu_total_mb)
        assert pct == expected[card], f"{card} reported {pct}%"


def test_the_job_needs_more_than_an_l4_has():
    """The whole lesson rests on this one inequality."""
    assert seedmod.NEEDS_MB > seedmod.CARDS["l4"]
    assert seedmod.NEEDS_MB < seedmod.CARDS["a100_40"]


def test_core_scan_reproduces_the_curve():
    """Exercise 3a: GPU utilisation climbs, then flattens; CPU efficiency falls."""
    seedmod.seed()
    store = JobStore()
    rows = []
    for job_id, cores, cpu_pct, gpu_pct in seedmod.BY_CORES:
        job = store.load(job_id)
        measured_cpu = 100 * job.cpu_seconds / (job.elapsed * job.cpus)
        rows.append((cores, round(measured_cpu), round(job.mean_gpu_util)))
        assert round(measured_cpu) == pytest.approx(cpu_pct, abs=1)
        assert round(job.mean_gpu_util) == pytest.approx(gpu_pct, abs=1)

    gpu = [r[2] for r in rows]
    cpu = [r[1] for r in rows]
    assert gpu == sorted(gpu), "GPU utilisation must never fall as cores are added"
    assert cpu == sorted(cpu, reverse=True), "CPU efficiency must fall throughout"
    assert gpu[-1] - gpu[2] <= 1, "past four cores there should be nothing left to gain"


def test_memory_grows_with_cores():
    """Chapter 4's point, and the reason 3b says to measure at your chosen count."""
    seedmod.seed()
    store = JobStore()
    peaks = [store.load(jid).max_rss_mb for jid, *_ in seedmod.BY_CORES]
    assert peaks == sorted(peaks)
    four_core = store.load(2002003)
    assert round(four_core.max_rss_mb / 1024, 1) == 6.2  # the figure in the chapter


def test_seff_reports_the_recorded_card_not_the_emulated_one(capsys):
    """Without this the exercise is unteachable.

    seff normally asks the live device how big the card is. The emulated device
    reports about a gigabyte, so every recorded job would read as 100% full and
    the five cards would be indistinguishable.
    """
    seedmod.seed()
    assert seff(["2001002"]) == 0
    out = capsys.readouterr().out
    assert "30.10 GB of 40 GB" in out
    assert "75%" in out


# -------------------------------------------------------------- profile_plot


def test_recorded_jobs_carry_a_time_series():
    seedmod.seed()
    rows = read_series(2002003)
    assert len(rows) > 10
    assert {"t", "cpus", "rss_mb", "gpu_util", "gpu_mem_mb"} <= set(rows[0])
    assert max(r["gpu_util"] for r in rows) <= 1.0, "GPUs panel is a fraction of a card"


def test_profile_plot_writes_a_png(tmp_path, capsys):
    seedmod.seed()
    out = tmp_path / "plot.png"
    assert profile_plot(["2002003", "-o", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 1000
    assert "Wrote" in capsys.readouterr().out


def test_profile_plot_explains_itself_when_profiling_was_off(tmp_path, capsys):
    """The likely mistake is forgetting --profile task, so say so."""
    seedmod.seed()
    (tmp_path / "slurm" / "profiles" / "2002003.csv").unlink()
    assert profile_plot(["2002003"]) == 1
    assert "--profile" in capsys.readouterr().err


def test_profile_plot_rejects_a_job_that_does_not_exist(capsys):
    assert profile_plot(["9999999"]) == 2
    assert "not found" in capsys.readouterr().err
