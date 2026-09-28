"""A node that holds more than one kind of card.

The workshop spends two chapters on choosing a GPU, and until now the session
had exactly one to choose from - every request was satisfied by the same
device, so `--gpus-per-node h100:1` and `--gpus-per-node l4:1` were the same
job with different spelling. These cover the part that makes the choice real:
that the node presents the cards it was configured with, that asking for one it
does not have is refused, and that a job which asks for a particular card is
given that card and measured against it.
"""

from __future__ import annotations

import time

import pytest

from gpuemu import client, spec
from gpuemu.daemon import Daemon
from gpuemu.shm import StateReader
from gpuemu.slurm import (
    ACTIVE_STATES,
    COMPLETED,
    JobStore,
    Scheduler,
    _seff_capacity,
    node_gres,
    parse_gpu_request,
    parse_gres_request,
    sbatch,
    seff,
    sinfo,
)

WHOLE_FLEET = "l4,a100_40,a100,h100,pro_6000"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUEMU_STATE_FILE", str(tmp_path / "state.bin"))
    monkeypatch.setenv("GPUEMU_CLAIMS_DIR", str(tmp_path / "claims"))
    monkeypatch.setenv("GPUEMU_SLURM_DIR", str(tmp_path / "slurm"))
    monkeypatch.delenv("GPUEMU_FLEET", raising=False)


@pytest.fixture
def whole_fleet(monkeypatch):
    monkeypatch.setenv("GPUEMU_FLEET", WHOLE_FLEET)


# ------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("l4", ["l4"]),
        ("l4,h100", ["l4", "h100"]),
        ("l4:2", ["l4", "l4"]),  # a bare number is a count
        ("l4:3:2GiB", ["l4", "l4", "l4"]),
        ("  l4 , h100  ", ["l4", "h100"]),
        ("", ["l4"]),  # falls back to the single-card spelling
        (",,", ["l4"]),
    ],
)
def test_fleet_string_decides_what_the_node_has(monkeypatch, text, expected):
    monkeypatch.setenv("GPUEMU_FLEET", text)
    assert [d.gres_name for d in spec.fleet()] == expected


@pytest.mark.parametrize(
    "text,mib",
    [
        ("l4", 200),  # DEFAULT_VRAM unless told otherwise
        ("l4:2GiB", 2048),
        ("l4:4:512MiB", 512),
        ("l4:full", 23034),  # the real board
    ],
)
def test_vram_is_per_card(monkeypatch, text, mib):
    monkeypatch.setenv("GPUEMU_FLEET", text)
    assert spec.fleet()[0].mem_total_mib == mib


def test_cards_keep_their_own_sizes(monkeypatch):
    """Switching one card to full size must not resize the others."""
    monkeypatch.setenv("GPUEMU_FLEET", "l4,a100:full,h100:4GiB")
    sizes = {d.gres_name: d.mem_total_mib for d in spec.fleet()}
    assert sizes == {"l4": 200, "a100": 81920, "h100": 4096}


def test_an_unknown_card_names_the_ones_that_exist(monkeypatch):
    monkeypatch.setenv("GPUEMU_FLEET", "v100")
    with pytest.raises(SystemExit) as exc:
        spec.fleet()
    assert "v100" in str(exc.value)
    assert "a100_40" in str(exc.value)


def test_a_fleet_too_big_for_the_state_file_is_refused(monkeypatch):
    monkeypatch.setenv("GPUEMU_FLEET", "l4:9")
    with pytest.raises(SystemExit) as exc:
        spec.fleet()
    assert str(spec.MAX_DEVICES) in str(exc.value)


def test_the_old_single_card_spelling_still_works(monkeypatch):
    monkeypatch.setenv("GPUEMU_DEVICE", "h100")
    monkeypatch.setenv("GPUEMU_GPUS", "2")
    monkeypatch.setenv("GPUEMU_MEM_TOTAL", "2GiB")
    devices = spec.fleet()
    assert [d.gres_name for d in devices] == ["h100", "h100"]
    assert all(d.mem_total_mib == 2048 for d in devices)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("gpu:1", ("", 1)),
        ("gpu:2", ("", 2)),
        ("gpu:l4:1", ("l4", 1)),
        ("gpu:a100_40:3", ("a100_40", 3)),
        ("gpu", ("", 1)),
        ("nvme:1", ("", 0)),
    ],
)
def test_gres_request_carries_the_card_name(text, expected):
    assert parse_gres_request(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("1", ("", 1)), ("l4:1", ("l4", 1)), ("pro_6000:2", ("pro_6000", 2))],
)
def test_gpus_per_node_carries_the_card_name(text, expected):
    assert parse_gpu_request(text) == expected


# -------------------------------------------------------------- the devices


def test_every_card_is_published_with_its_own_identity(whole_fleet):
    """nvidia-smi and nvtop have to show five different boards, not five L4s."""
    daemon = Daemon()
    daemon.tick(0.2)
    gpus = StateReader(daemon.writer.path).read().gpus

    assert len(gpus) == 5
    assert len({g.name for g in gpus}) == 5
    assert len({g.uuid for g in gpus}) == 5
    # Power envelope is per board, and is the reading that gives the game away
    # if every device is secretly the same one.
    assert [g.power_limit_mw // 1000 for g in gpus] == [72, 250, 400, 400, 600]


def test_a_card_can_be_left_off_the_node(monkeypatch):
    monkeypatch.setenv("GPUEMU_FLEET", "l4,h100")
    assert node_gres() == ["l4", "h100"]


def test_capacity_is_read_per_device(monkeypatch):
    monkeypatch.setenv("GPUEMU_FLEET", "l4:1GiB,a100:4GiB")
    assert client.device_capacity(1) > client.device_capacity(0)


# ------------------------------------------------------------------- sbatch


def _submit(tmp_path, capsys, *args):
    script = tmp_path / "job.sl"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    code = sbatch([*args, "--parsable", str(script)])
    return code, capsys.readouterr()


def test_asking_for_a_card_the_node_has_is_accepted(whole_fleet, tmp_path, capsys):
    code, out = _submit(tmp_path, capsys, "--gpus-per-node", "a100:1")
    assert code == 0, out.err
    job = JobStore().load(int(out.out.split()[0]))
    assert job is not None
    assert job.gpu_type == "a100"
    assert job.gpus == 1


def test_asking_for_a_card_the_node_lacks_is_refused(monkeypatch, tmp_path, capsys):
    """The cluster refuses this at submit rather than leaving it pending forever."""
    monkeypatch.setenv("GPUEMU_FLEET", "l4")
    code, out = _submit(tmp_path, capsys, "--gpus-per-node", "h100:1")
    assert code == 1
    assert "not available" in out.err
    assert "h100" in out.err
    assert "l4" in out.err, "the error should say what there is instead"


def test_asking_for_more_of_a_card_than_exists_is_refused(whole_fleet, tmp_path, capsys):
    code, out = _submit(tmp_path, capsys, "--gpus-per-node", "h100:2")
    assert code == 1
    assert "1 on" in out.err  # one H100 on this node, two asked for


def test_gres_spelling_is_honoured_too(whole_fleet, tmp_path, capsys):
    code, out = _submit(tmp_path, capsys, "--gres", "gpu:pro_6000:1")
    assert code == 0, out.err
    assert JobStore().load(int(out.out.split()[0])).gpu_type == "pro_6000"


def test_an_unnamed_gpu_request_takes_whatever_is_free(whole_fleet, tmp_path, capsys):
    code, out = _submit(tmp_path, capsys, "--gpus-per-node", "1")
    assert code == 0, out.err
    assert JobStore().load(int(out.out.split()[0])).gpu_type == ""


def test_sinfo_lists_the_cards_you_can_ask_for(whole_fleet, capsys):
    assert sinfo(["-l"]) == 0
    out = capsys.readouterr().out
    for card in ("gpu:l4:1", "gpu:a100_40:1", "gpu:h100:1", "gpu:pro_6000:1"):
        assert card in out


# --------------------------------------------------------------- allocation


def _run_to_completion(store: JobStore) -> None:
    sched = Scheduler()
    deadline = time.time() + 30
    while time.time() < deadline:
        sched.tick()
        job = store.all()[0]
        if job.state not in ACTIVE_STATES:
            return
        time.sleep(0.05)
    raise AssertionError(f"job never finished: {store.all()[0].state}")


def test_a_job_is_given_the_card_it_asked_for(whole_fleet, tmp_path, monkeypatch):
    """Not just any free GPU - the one named in the request.

    Device 3 is the H100 in this fleet. A scheduler that handed out the first
    free device would give this job the L4 at index 0, and every conclusion the
    learner drew from it would be about the wrong card.
    """
    script = tmp_path / "job.sl"
    script.write_text(
        "#!/bin/bash\n#SBATCH --gpus-per-node h100:1\n"
        'echo "devices=$CUDA_VISIBLE_DEVICES"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert sbatch([str(script)]) == 0

    store = JobStore()
    _run_to_completion(store)

    job = store.all()[0]
    assert job.state == COMPLETED
    assert job.gpu_ids == [3]
    assert "devices=3" in (tmp_path / f"slurm-{job.job_id}.out").read_text()


def test_seff_measures_against_the_card_the_job_ran_on(monkeypatch, tmp_path, capsys):
    """A 4 GiB card and a 1 GiB card must not report the same denominator."""
    monkeypatch.setenv("GPUEMU_FLEET", "l4:1GiB,a100:4GiB")
    script = tmp_path / "job.sl"
    script.write_text("#!/bin/bash\n#SBATCH --gpus-per-node a100:1\ntrue\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert sbatch([str(script)]) == 0

    store = JobStore()
    _run_to_completion(store)
    capsys.readouterr()

    job = store.all()[0]
    assert job.gpu_ids == [1]
    assert seff([str(job.job_id)]) == 0
    assert "of 4 GB" in capsys.readouterr().out


@pytest.mark.parametrize(
    "gb,expected",
    [
        (40.0, "40 GB"),
        (1.0, "1 GB"),
        (200 / 1024, "200 MB"),  # the default card
        (100 / 1024, "100 MB"),
        (0.5, "512 MB"),
    ],
)
def test_a_small_card_is_reported_in_megabytes(gb, expected):
    """A 100 MB card rounded to whole GB reads as "of 0 GB"."""
    assert _seff_capacity(gb) == expected


def test_seff_does_not_measure_against_a_zero_byte_card(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GPUEMU_FLEET", "l4:100MiB")
    script = tmp_path / "job.sl"
    script.write_text("#!/bin/bash\n#SBATCH --gpus-per-node l4:1\ntrue\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert sbatch([str(script)]) == 0

    store = JobStore()
    _run_to_completion(store)
    capsys.readouterr()

    assert seff([str(store.all()[0].job_id)]) == 0
    out = capsys.readouterr().out
    assert "of 100 MB" in out
    assert "of 0 GB" not in out
