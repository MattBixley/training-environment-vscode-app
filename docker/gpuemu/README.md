# gpuemu

An emulated NVIDIA GPU for machines that have none.

There is no way to run real CUDA kernels without a real GPU, and this does not
pretend otherwise. What it does is make everything *around* the GPU behave
correctly, which turns out to be most of what a GPU workshop teaches.

## Layout

```
nvml/gpuemu_shm.h      shared-memory layout, the contract between C and Python
nvml/libnvidia_ml.c    the stand-in libnvidia-ml.so.1
nvml/abi_dump.c        prints the C struct layout, for the ABI test
src/gpuemu/spec.py     device definitions (L4, A100, H100) and their physics constants
src/gpuemu/shm.py      Python half of the shared-memory layout
src/gpuemu/daemon.py   gpuemud: the simulation loop
src/gpuemu/client.py   claims: how a process registers memory and load
src/gpuemu/smi.py      nvidia-smi
src/gpuemu/slurm.py    sbatch, squeue, scancel, sinfo, sacct, scontrol, srun
src/gpuemu/torch_shim.py  makes torch.cuda work
src/gpuemu/autoclaim.py   claims the device on first torch / numba.cuda import
src/gpuemu/cli.py      gpuemu-ctl, gpuemu-burn
```

## How the pieces fit

```
  nvtop ─┐
         ├─ dlopen("libnvidia-ml.so.1") ─→ libnvidia_ml.c ─┐
  pynvml ┘                                                 │  mmap, seqlock
                                                           ├─→ state.bin
  nvidia-smi ──→ gpuemu.smi ──→ StateReader ───────────────┘       ↑
                                                                   │ writes
  learner's code ──→ gpuemu.client.Claim ──→ claims/*.json ──→ gpuemud
                     (memory, utilisation)                    (simulation loop)
```

**The claims directory is the input, the state file is the output.** A process
that wants to appear on the device writes a small JSON claim; the daemon reads
every live claim each tick, runs the device model, and publishes the result to
a memory-mapped state file that NVML readers consume.

Claims are files rather than messages to a server so that **a process that
crashes cannot leak GPU memory**: the daemon checks each claim's pid and reaps
the dead ones, which is what a real driver cleaning up an orphaned context
amounts to here.

## Why a shared library works

Monitoring tools never link NVML at build time. nvtop does:

```c
libnvidia_ml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
nvmlInit = dlsym(libnvidia_ml_handle, "nvmlInit_v2");
...
```

Every entry point is resolved by name at runtime. A shared object exporting
those names with those signatures is therefore indistinguishable from the real
driver, and needs no kernel module, no `/dev/nvidia*` and no privileges.

`libnvidia_ml.c` implements the ~45 functions nvtop, `nvidia-smi` and `pynvml`
actually call, including the versioned variants (`_v1`/`_v2`/`_v3`) and NVML's
"ask twice" convention for enumerating processes. Functions for hardware the
emulated card lacks (NVLink, MIG) return `NVML_ERROR_NOT_SUPPORTED`, which is
what the real card returns, so callers hide those panels exactly as they would.

## The ABI contract

`gpuemu_shm.h` and `shm.py` describe the same bytes. Both are packed, so C's
layout matches Python's `struct` in `<` mode field for field.

If they ever drift, **nothing raises**: the shim reads the wrong offsets and
reports convincing nonsense. `tests/test_abi.py` compiles `abi_dump.c` and
compares the C compiler's sizes and offsets against Python's, which is what
keeps that from happening quietly. Add a field to one, add it to the other, and
run the tests.

Concurrency is a seqlock: the writer bumps `seq` to odd before touching
anything and to even when done; readers retry while it is odd or changed. One
writer, any number of readers, and no lock a crashing writer could leave held.

## The device model

`daemon.py` models behaviour rather than just echoing requested numbers,
because readings that snap between 0 and 100 teach nothing. Utilisation ramps
and decays, clocks boost and drop back, power rises faster than linearly with
utilisation, and temperature lags power with a ~45 s time constant — so a
finished job still looks warm, which is the relationship worth learning.

By default a claim's utilisation is **derived from the CPU time the claiming
process and its children actually burn**, normalised against the container's
CPU allocation. That is what connects the display to the learner's own code
without anything having to be annotated: their training loop runs, the GPU is
busy; it stops, the GPU idles.

Two consequences worth knowing:

- Give sessions enough CPUs. On one or two cores, every trivial job pins the
  meter at 100% and the reading stops meaning anything.
- `GPUEMU_UTIL_GAIN` scales it if jobs look busier or idler than you want.

## Memory is real

Claims are checked against the card's capacity, so asking for more than the
emulated device has raises `OutOfMemoryError` — and through the torch shim,
`torch.cuda.OutOfMemoryError` with the message PyTorch would really produce.

This is deliberate. Hitting OOM and learning to read it is a large part of what
a GPU workshop is for, and an emulator with infinite memory would quietly teach
the opposite of the thing that matters.

Cards can be shrunk — `GPUEMU_FLEET=l4:100MiB` gives an L4 with 100 MB of VRAM,
which is the default.
This is the practical way to run a memory-pressure exercise: the learner gets a
real out-of-memory error from a tensor that costs the host almost nothing,
instead of the session having to allocate 24 GB to reach the limit. The
driver's reservation is scaled to the same proportion the real board has, so
`total`, `used` and `free` stay consistent.

## A node with more than one kind of card

`GPUEMU_FLEET` lists the node's GPUs as `name[:count][:vram]`, comma separated,
and they need not all be the same board:

```bash
GPUEMU_FLEET=l4,a100_40,a100,h100,pro_6000   # one of each, 100 MB apiece
GPUEMU_FLEET=l4:4:2GiB,h100:2                # four L4s at 2 GiB, two H100s
GPUEMU_FLEET=a100:full                       # one A100 at its real 80 GB
```

A bare number in the second field is a count, so `l4:2` is two cards and
`l4:2GiB` is one card with 2 GiB. Device order follows the list.

With a mixed fleet, `--gpus-per-node a100:1` means what it says: `sbatch`
refuses a card the node does not have, the scheduler hands the job a device of
the type it asked for, and `seff`, `nvidia-smi`, `nvtop` and
`torch.cuda.get_device_name()` all report that card rather than device 0's.

`GPUEMU_DEVICE`, `GPUEMU_GPUS` and `GPUEMU_MEM_TOTAL` are the older
single-card spelling, still read when `GPUEMU_FLEET` is unset.

## Tests

```bash
python3 -m pytest tests -q
```

65 tests: the C/Python ABI, memory accounting and OOM, reaping dead claims,
telemetry behaviour (idle, under load, thermal lag, sharing between claims),
Slurm argument parsing, the scheduler running jobs to completion, and every
command-line entry point.

That last group exists because a `squeue -h` that crashed on an argparse option
conflict once shipped past the behaviour tests — Slurm spells `-h` as
`--noheader`, not `--help`, and nothing had ever invoked the command itself.

The image build runs this suite, loads the shim and enumerates the device
through it, so a broken emulator fails the build rather than a workshop.

## Using it directly

```python
import gpuemu

# Claim the device for a block of work
with gpuemu.gpu(memory="4GiB", util=85):
    do_something()

# Or inspect it
snap = gpuemu.snapshot()
for g in snap.gpus:
    print(g.name, g.util_gpu, g.mem_used // 1024**2)
```

```bash
gpuemu-ctl start|stop|restart|status|reset
gpuemu-burn --time 60 --memory 4GiB     # known load, for demonstrations
gpuemud --once -v                        # write one frame and exit
```

## Adding a device

Add a `DeviceSpec` to `spec.py`, register it in `DEVICES` under the name Slurm
knows the card as, and add that name to `FLEET`. Use the real board's published
figures — learners compare what they see against documentation, and round
numbers give the game away for no benefit. It can then be asked for by name in
`GPUEMU_FLEET` and in `--gpus-per-node`.
