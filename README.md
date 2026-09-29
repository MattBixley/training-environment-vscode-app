# REANNZ training environment VS Code app

An Open OnDemand app that gives each learner **VS Code in the browser**
([code-server](https://github.com/coder/code-server)) running in their own pod
on the NeSI training environment. It is built for research software workshops:
Python, git, SSH and GitHub work out of the box. Optional **GPU and Slurm
emulators** let the same app teach GPU and HPC batch-job skills on hardware
with no GPUs and no cluster scheduler.

The first workshop to use it is [Better Research Software][brs]. The app is a
fork of [training-environment-jupyter-gpu-app][gpu-app], which is where the
emulators come from.

[brs]: https://carpentries-incubator.github.io/better-research-software/
[gpu-app]: https://github.com/nesi/training-environment-jupyter-gpu-app

## Capabilities

**Editor**

- VS Code in the browser via code-server 4.139.1, opened at the learner's home directory
- Password login handled by Open OnDemand: the **Connect to VS Code** button logs straight in
- Integrated terminal (bash login shell), file explorer, Source Control panel, Test Explorer, debugger
- Pre-installed extensions, copied into each learner's own extensions directory on first start so they can add more from Open VSX:
  - Python and Python Debugger
  - basedpyright as the language server (Pylance is Microsoft-only and not on Open VSX)
  - autoDocstring, for generating docstrings
  - Git Graph, for a visual commit history
  - Live Preview, for opening HTML output (coverage reports, documentation sites) inside VS Code
- Sensible defaults (no telemetry, no update nags, no workspace-trust prompts), written only when the learner has no settings of their own

**Python**

- Python 3.12 on Ubuntu 24.04, with `venv` and `pip`
- The system Python is marked externally managed, so `pip install` outside a virtual environment is refused. This pushes learners into the venv workflow most lessons teach.

**git, SSH and GitHub**

- git with system defaults `init.defaultBranch main` and `core.editor "nano -w"`; learners set their own name and email
- `ssh` and `ssh-keygen`; keys live in the learner's persistent NFS home, so they survive session restarts
- GitHub SSH is routed through `ssh.github.com:443`, so `git@github.com:` remotes work even where outbound port 22 is blocked
- GitHub's published SSH host keys are pinned, so learners never have to trust an unverified fingerprint
- GitHub CLI (`gh`) as an HTTPS fallback (`gh auth login --web`)

**Optional GPU and Slurm emulators** (off by default; set **Emulated GPUs and Slurm** to **On** on the launch form)

- Emulated NVIDIA L4, A100 40/80 GB, H100 NVL and RTX PRO 6000 cards, each with configurable memory
- `nvidia-smi` and `nvtop` with live telemetry
- PyTorch `torch.cuda` works and enforces device memory limits, so out-of-memory errors are real
- Numba's CUDA simulator runs `@cuda.jit` kernels
- A single-node Slurm stand-in: `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`, `scontrol`, `srun`, `seff`
- GPU workshop exercises and pre-recorded jobs in `~/gpu-training/`
- **All computation runs on the CPU.** Timings mean nothing about GPU performance, and the session says so.

**Operations**

- One image, `ghcr.io/mattbixley/training-environment-vscode-app`, about 0.9 GB compressed
- Launch form: CPUs (2 or 4), memory (4 or 8 GB), emulator Off/On, wall time; the per-card GPU memory options appear only when the emulators are On
- Workshop material is copied into home directories without overwriting learner work, so a restarted session picks up where it left off
- CI checks that `form.yml` parses the way OnDemand parses it, that the templates are executable, and that the release tag matches the pinned image, before any image is pushed

## Use case examples

**Better Research Software (Carpentries Incubator, 2 days).** Learners start
with a messy Python project and turn it into reproducible, tested,
documented, citable software.

- Unzip the lesson's `~/spacewalks.zip` (already staged) and open the folder in VS Code
- `git init`, commit, then create a GitHub repo and push over SSH:
  `ssh-keygen -t ed25519`, paste `~/.ssh/id_ed25519.pub` into GitHub, `ssh -T git@github.com`
- `python3 -m venv venv_spacewalks`, install pandas and matplotlib, then pick the interpreter with **Python: Select Interpreter**
- Refactor with basedpyright hints and autoDocstring, then run `pytest --cov --cov-report=html`. Right-click `htmlcov/index.html` and choose **Show Preview**.
- `mkdocs build` and preview `site/index.html`, then `mkdocs gh-deploy` to GitHub Pages
- Branch, merge and open pull requests, watching history in Git Graph

Launch with the defaults: 2 CPUs, 4 GB, emulators off.

**Intro to GPU computing.** Set the emulators to On and choose 4 CPUs / 8 GB.
Learners run `nvidia-smi`, write an `sbatch` script requesting
`--gpus-per-node l4:1`, watch it in `nvtop`, hit and fix a CUDA out-of-memory
error, and read job efficiency with `seff`. They do all of this in the same editor
they will use on the real cluster.

**Preparing for HPC.** With the emulators on, learners write and debug Slurm
job scripts (`#SBATCH` directives, `squeue`, `scancel`, `sacct`) in VS Code
before they have a cluster account, then move the same scripts to Mahuika.

**Git and GitHub collaboration.** Pair learners on one repository: SSH keys,
branches, pull requests, code review and resolving merge conflicts in VS Code's
merge editor, with Git Graph showing what happened.

## Notes for instructors

- Remind learners to delete their workshop SSH key from GitHub
  (**Settings → SSH and GPG keys**) at the end of the workshop.
- Better Research Software's setup check runs `code --list-extensions`. In a
  code-server terminal `code` is the remote CLI and may behave differently, so
  skip that step.
- Extensions installed from inside the session come from Open VSX, not the
  Microsoft marketplace; a few Microsoft-only extensions (Pylance, Live Share)
  are not available.

## Repository layout

```
form.yml               session options: CPUs, memory, emulator toggle, per-card GPU memory, wall time
submit.yml.erb         k8s pod spec; composes GPUEMU_FLEET and GPUEMU_ENABLE from the form
template/script.sh.erb starts the emulators if enabled, copies the material, launches code-server
docker/code-server/    default VS Code settings
docker/ssh/            GitHub SSH-over-443 config
docker/Dockerfile      the session image
docker/gpuemu/         the emulator (see docker/gpuemu/README.md)
docker/workshop/       the exercises, copied to ~/gpu-training/
```

## Session options

The launch form asks for CPUs (2 or 4), memory (4 or 8 GB), wall time, and
whether to start the **Emulated GPUs and Slurm**. Everything below applies only
when the emulators are On; while Off, the card controls are hidden.

With the emulators on, the session's node presents **every card on Mahuika at once**, one of each:

| Slurm name | Reports as | Memory | Notes |
|---|---|---|---|
| `l4` | NVIDIA L4 | 24 GB | passive, 72 W; fp64 at 1/62 of fp32 |
| `a100_40` | NVIDIA A100-PCIE-40GB | 40 GB | Ampere, 250 W; fp64 at 1/2. Slurm name provisional |
| `a100` | NVIDIA A100-SXM4-80GB | 80 GB | Ampere, 400 W; fp64 at 1/2 |
| `h100` | NVIDIA H100 NVL | 94 GB | Hopper, 400 W; fp64 at 1/2 |
| `pro_6000` | NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | Blackwell, 600 W, PCIe Gen5, cc 12.0; fp64 at 1/63 |

So `--gpus-per-node a100:1` and `--gpus-per-node l4:1` are genuinely different
requests that get different devices, and asking for a card the node does not
have is refused by `sbatch` the way the cluster refuses one the partition
lacks. `sinfo -l` lists what there is. Each card changes the reported name,
memory, clocks, power envelope and compute capability, and nothing else; no
configuration is any more or less real than the others.

These match the cards on the cluster rather than the nearest generic part, so
a learner comparing what they see here against the hardware documentation
finds the same VRAM figures and the same Slurm names.

The form has one **GPU memory** control per card. It sets how much device
memory that card reports, and **defaults to 200 MB** rather than the board's real
size. That is deliberate: a small card is what makes memory pressure teachable,
since a learner hits a genuine out-of-memory error with a tensor that costs the
session almost nothing. Filling a real 24 GB card would need 24 GB of host RAM
per session, and the node holds five cards.

Emulated VRAM is accounted rather than reserved, so the cards cost nothing until
a tensor is actually made; a filled card does cost that much host memory, which
is why the default is small.

Choose **Full size** when the workshop does not cover memory, or when the
reported capacity is itself the point — comparing an L4 against an RTX PRO
6000, say. Choose **Not on this node** to leave a card out entirely.

For more than one of a card — which is how to teach device selection and
`CUDA_VISIBLE_DEVICES` — set `GPUEMU_FLEET` directly; see the
[configuration reference](#configuration-reference). The scheduler allocates
devices to jobs independently.

CPU and memory default to 2 cores and 4 GB, which is plenty for Better
Research Software. Choose 4 cores and 8 GB for GPU workshops: emulated GPU
utilisation is derived from real CPU use, so on one or two cores every trivial
job pins the meter at 100% and the reading stops teaching anything.

## Trying it on your own machine

```bash
./run-local.sh
```

This starts the same image the cluster runs, with the emulators on and no
password, and prints a VS Code URL. Docker is the only requirement.

```bash
./run-local.sh --fleet 'l4,a100:full'          # an L4 and a full-size A100
./run-local.sh --fleet l4:4:2GiB               # four L4s with 2 GiB each
./run-local.sh --shell                         # a terminal instead of VS Code
./run-local.sh --build                         # build from this checkout first
```

Everything a learner actually does inside the session behaves identically:
`nvidia-smi`, `nvtop`, `sbatch`, `seff`, the exercises, PyTorch, OOM errors. What it
does not reproduce is the Open OnDemand wrapper — no login, no k8s, no NFS
home directories, no LDAP — so `form.yml`, `submit.yml.erb` and
`template/script.sh.erb` are only exercised by an actual deployment. That
matters: the one bug that reached the cluster (`template/*.erb` committed
non-executable) was in exactly that untested layer, which is why CI now checks
it directly.

On an Apple Silicon Mac the published image is amd64 and runs under emulation —
correct but slow to start. `--build` produces a native image if you are
iterating.

## Building and testing

The image build runs the emulator's test suite, loads the NVML shim and
enumerates the device through it, so a broken emulator fails the build rather
than surfacing in front of a workshop.

Locally:

```bash
cd docker/gpuemu
python3 -m pytest tests -q          # ABI, accounting, telemetry, scheduler, app config
```

The ABI test compiles `nvml/abi_dump.c` and compares the C compiler's struct
layout against the Python `struct` format. If those two drift apart nothing
raises on its own — the shim just reads the wrong offsets and reports
convincing nonsense — so that test is the thing keeping the whole mechanism
honest.

Full image:

```bash
docker build -t vscode-app docker/
docker run --rm -it vscode-app bash -lc 'gpuemu-ctl start && nvidia-smi && nvtop'
```

Pushing to this repo triggers the container build in
`.github/workflows/build_container.yml`.

## Deploying it

The app is deployed through the [training-environment][te] repo, as an entry
under `ood_apps:` in `vars/ondemand-config.yml.example` (for example
`vscode_brs`); see
[the deployment tutorial][deploy]. The image is published to
`ghcr.io/mattbixley/training-environment-vscode-app`, which must be set to
**public** in the package settings before the cluster can pull it.

[te]: https://github.com/nesi/training-environment
[deploy]: https://nesi.github.io/training-environment/tutorials/deployment-on-nesi/

## Troubleshooting

**Connect to VS Code shows a login page or "incorrect password".** The session
password is set when the pod starts. Relaunch the session from My Interactive
Sessions rather than reusing an old browser tab.

**An extension is missing.** Extensions are copied into
`~/.local/share/code-server/extensions` only when they are not already there.
Delete that directory and restart the session to get the image's set back.

**`ssh -T git@github.com` hangs.** The pod cannot reach `ssh.github.com:443`.
Use HTTPS instead: `gh auth login --web`, then `gh auth setup-git`.

**`pip install` says "externally-managed-environment".** That is deliberate.
Create and activate a venv first: `python3 -m venv .venv && . .venv/bin/activate`.

**`nvidia-smi` says it cannot talk to the driver.** The daemon is not running.
`gpuemu-ctl status`, then `gpuemu-ctl start`.

**A job stays `PENDING` forever.** The scheduler is not running — the same
fix. `sbatch` warns about this at submission time.

**nvtop shows no devices.** The shim is not on the loader path. Check
`ldconfig -p | grep nvidia-ml` and that `/etc/ld.so.conf.d/gpuemu.conf` exists.

**Utilisation sits at 100% for trivial work, or never rises.** It is derived
from CPU use against the container's CPU allocation. Give the session more
cores, or tune `GPUEMU_UTIL_GAIN`.

**Something needs real CUDA.** Custom CUDA extensions, Triton kernels,
`torch.compile` with a CUDA backend and bitsandbytes cannot work here and will
fail where they try to load device code. There is no workaround short of real
hardware; design the workshop around it.

---

# GPU and Slurm emulator reference

Set **Emulated GPUs and Slurm** to **On** on the launch form. The session then presents
emulated NVIDIA GPUs: `nvidia-smi` and `nvtop` show devices with live
telemetry, `sbatch`/`squeue`/`scancel` queue and run jobs that request GPUs,
PyTorch's CUDA API works, and device memory is genuinely limited so
oversubscribing it raises a real out-of-memory error.

All computation runs on the CPU. There is no GPU anywhere. While it is Off, no
emulator daemons run and the card controls are hidden.

## What this can and cannot teach

This exists because most of what a GPU workshop covers does not actually need
the silicon. The parts that do cannot be faked, and pretending otherwise would
be worse than useless.

| Taught faithfully | Cannot be taught here |
|---|---|
| Requesting a GPU from the scheduler, and checking you got one | Anything about speed |
| Reading `nvidia-smi`: memory, utilisation, processes, power | Kernel-level performance, occupancy, profiling |
| Watching a job live with `nvtop` | Tensor cores, mixed-precision speedups |
| Diagnosing "my job ran but ignored the GPU" | Multi-GPU scaling behaviour, NCCL |
| Device memory budgeting, and recovering from OOM | Custom CUDA C++ extensions, Triton, `torch.compile` |
| Reading `seff`: CPU, memory and GPU efficiency after a job | Whether one kernel is faster than another |
| Structuring a batch job script around a GPU allocation | GPU-specific numerical behaviour (TF32, etc.) |
| Working out which parts of a workload belong on a GPU | How much slower fp64 really is on a 1:64 card |
| Writing correct CUDA kernels — threads, blocks, shared memory, races | Anything that needs a real driver or real device code |

**The single rule to give learners: no timing measured in this environment
means anything about GPU performance.** The session banner, the workshop
README and the exercise scripts all say so, in those words. Please do
not remove those notices — an emulator this convincing is only safe to use if
it is loud about what it is.

---

## How the emulation works

Three pieces, in `docker/gpuemu/`.

### 1. A stand-in NVML library

`nvml/libnvidia_ml.c` builds `libnvidia-ml.so.1` and installs it on the loader
path. This works because GPU monitoring tools never link NVML at build time:
nvtop `dlopen`s `libnvidia-ml.so.1` and resolves every entry point with
`dlsym`. A shared object exporting the right symbols with the right signatures
is, from the caller's side, indistinguishable from the real driver — and needs
no kernel module, no `/dev/nvidia*`, and no privileges.

It implements the ~45 NVML functions nvtop, `nvidia-smi` and `pynvml` actually
call. Functions for hardware the L4 does not have (NVLink, MIG) return
`NVML_ERROR_NOT_SUPPORTED`, which is what a real L4 returns, so callers hide
those panels exactly as they would on the real card.

### 2. A device simulator

`gpuemu/daemon.py` (`gpuemud`) maintains the device's state and publishes it
through a small mmap'd file that the C shim reads. It models the behaviour that
makes the readings worth watching:

- utilisation ramps and decays rather than snapping between 0 and 100
- clocks boost under load, drop to idle, and trim near the thermal limit
- power rises faster than linearly with utilisation, capped at 72 W
- temperature lags power with a ~45 s time constant, so a finished job still
  looks warm — which is the relationship a learner should come away with
- the fan reports `N/A`, because the L4 is passively cooled

Utilisation is derived, by default, from **the CPU time the claiming process
and its children actually burn**. That is what connects the display to the
learner's own code: when their training loop runs, the emulated GPU is busy;
when it stops, it idles. Nothing has to be annotated for this to work.

### 3. Memory accounting that bites

A process claims device memory by dropping a small JSON file in a claims
directory; the daemon aggregates the live ones and reaps any whose owner has
died, so a killed script frees its memory the way a real driver would
clean up a dead context.

Claims are checked against the card's capacity, so asking for more than it has
fails with a genuine out-of-memory error. Hitting OOM and learning to read it
is a large part of what a GPU workshop is for, and an emulator with infinite
memory would quietly teach the opposite.

That capacity defaults to **200 MB per card**, not the board's real size — see
[Session options](#session-options). A small card is what makes the exercise
affordable: filling a real 24 GB card would cost 24 GB of host RAM per session,
and the node holds five cards.

### PyTorch

`gpuemu/torch_shim.py` makes `torch.cuda` work: `is_available()` is true,
`.cuda()` and `.to("cuda")` place tensors, `get_device_name()` returns
`NVIDIA L4`, and every tensor placed on "cuda" is counted against the emulated
card and reported to NVML. Exceeding it raises `torch.cuda.OutOfMemoryError`
with the message PyTorch would really produce.

Tensors live in host memory, so `tensor.device` reports `cpu`. Set
`GPUEMU_TORCH_SPOOF_DEVICE=1` to have it report `cuda:0` instead, for
demonstrations where the illusion matters more than the accuracy.

### CUDA kernels

The image sets `NUMBA_ENABLE_CUDASIM=1`, so Numba's `@cuda.jit` kernels
execute in the interpreter with faithful CUDA semantics: `threadIdx`,
`blockIdx`, grid-stride loops, shared memory, `syncthreads` and race conditions
all behave as they would on hardware. This is the one part of the environment
where learners write GPU code and get genuinely correct GPU behaviour back,
including the bugs — remove a `syncthreads` and the answer goes wrong for the
right reason. It is slow, so keep problem sizes small.

See `docker/workshop/supplementary/cuda_kernel.py`. The main workshop does
not cover this: it is written for researchers running GPU software, not for
people writing GPU kernels.

### Batch jobs

`gpuemu/slurm.py` provides `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`,
`scontrol show job` and `srun` over a single-node queue inside the container,
with Slurm's flags, output columns and job states. `#SBATCH` directives are
parsed from the script's leading comment block and stop at the first real
command, as Slurm does.

The detail that earns its place: **a job that did not request a GPU is started
with `CUDA_VISIBLE_DEVICES` empty**, so it genuinely cannot see the device and
`nvidia-smi` inside it fails. Forgetting `--gpus-per-node` therefore produces
the same baffling symptom here as it does on Mahuika, which is exactly the
lesson. `docker/workshop/01_requesting_a_gpu/forgot-the-gpu.sl` is that
mistake, on purpose.

It is a teaching scaffold, not Slurm: one node, first-come-first-served, no
fair-share, no backfill, no accounting database.

### Job efficiency: seff and svisit

`seff <jobid>` reports what a finished job actually used, and `svisit <jobid>`
opens a terminal "on the node" running a job so `nvtop` can watch it. Both
follow [nesi/opt-nesi-bin][bin] rather than the summary on the documentation
site, which is a version behind — the labels are `Avg CPU Utilisation` and
`Peak Mem Utilisation`, percentages are whole numbers right-aligned so every
`%` lands in the same column, and `Cluster:` only appears with `-M`.

```
Job ID: 1000
State: COMPLETED
Tasks: 1
Cores: 2
Job Wall-time:          2%  00:00:02 of 00:02:00 time limit
Avg CPU Utilisation:   42%  00:00:01 of 00:00:04 core-walltime
Peak Mem Utilisation:   3%  13.41 MB of 512.00 MB
Peak GPU Utilisation:   0%
Peak GPU Memory Util:   0%  0.00 MB of 200 MB
```

The numbers are measured, not invented. CPU time and peak resident memory come
from `wait4()` on the job — the same accounting a cgroup would give, and the
only way to get an honest figure out of a job that runs for four seconds.
Utilisation is a rate with no total to read afterwards, so that alone is
sampled while the job runs.

A job that asked for no GPU gets no GPU lines, exactly as on the cluster, and
that absence is the diagnosis for "why was my GPU job so slow?".

[bin]: https://github.com/nesi/opt-nesi-bin

---

## Configuration reference

Read by the emulator at startup; set in `submit.yml.erb` or the Dockerfile.

| Variable | Default | Effect |
|---|---|---|
| `GPUEMU_FLEET` | the whole fleet | The node's GPUs; see below |
| `GPUEMU_DEVICE` | `l4` | One card, when `GPUEMU_FLEET` is unset |
| `GPUEMU_MEM_TOTAL` | `200MiB` | Its memory; empty means the real board's size |
| `GPUEMU_GPUS` | `1` | How many of it to present (max 8) |
| `GPUEMU_UTIL_GAIN` | `1.0` | Scales derived utilisation; raise if jobs look idle |
| `GPUEMU_AUTOCLAIM` | `1` | Claim the device when a process imports torch or numba.cuda |
| `GPUEMU_TORCH_SPOOF_DEVICE` | unset | Make `tensor.device` report `cuda:0` |
| `GPUEMU_STATE_FILE` | `/run/gpuemu/state.bin` | Where device state lives |
| `NUMBA_ENABLE_CUDASIM` | `1` | Run `@cuda.jit` kernels in the simulator |

`GPUEMU_FLEET` is a comma-separated list of `name[:count][:vram]`, and is what
lets the node hold more than one kind of card. A bare number in the second
field is a count, so `l4:2` is two cards and `l4:2GiB` is one card with 2 GiB.
`full` means the board's real capacity. Device order follows the list, so the
first entry is device 0.

```bash
GPUEMU_FLEET=l4,a100_40,a100,h100,pro_6000   # the default: one of each, 200 MB
GPUEMU_FLEET=l4:4:2GiB,h100:2                # four L4s at 2 GiB, two H100s
GPUEMU_FLEET=a100:full                       # one A100 at its real 80 GB
```

At most 8 devices; the state file has no room for more. A name the emulator
does not know, or a fleet that will not fit, fails at startup rather than
quietly presenting something else.
