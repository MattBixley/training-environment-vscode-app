# Introduction to Using GPUs — exercises

Everything here runs from the **terminal**. In JupyterLab, open one with
**File → New → Terminal**, then:

```bash
cd ~/gpu-training
ls
```

The folders are numbered in the order the workshop covers them. The lesson text
is at <https://nesi.github.io/reannz-intro-gpu-workshop/>.

| Folder | Chapter |
|---|---|
| `06_tools_for_measuring` | A test submit script, and reading it back with `seff` and `profile_plot` |
| `07_putting_it_together` | The whole loop on one job: watch it, size the card, tune cores and memory |
| `supplementary` | Optional: checking the GPU is used, precision, splitting CPU/GPU work, CUDA kernels |

Chapters 1 to 5 are read rather than run — they are about deciding what to put
in a submit script, and the deciding happens before there is anything to
submit.

## The commands you will use

| Command | What it is for |
|---|---|
| `sbatch script.sl` | Submit a job |
| `squeue --me` | See your jobs. `ST` is the state: `PD` pending, `R` running |
| `svisit <jobid>` | Open a terminal inside a running job |
| `nvtop` | Watch GPU utilisation and memory live. `q` to quit |
| `nvidia-smi` | Check a GPU is there. A snapshot, not a monitor |
| `seff <jobid>` | See what a finished job actually used |
| `scancel <jobid>` | Stop a job |

## About this environment

**There is no GPU in this training environment.** It emulates one, so that a
workshop about using GPUs can run on hardware that has none.

What that means in practice:

* `nvidia-smi`, `nvtop`, `seff`, `sbatch` and PyTorch's CUDA API all behave
  the way they do on the cluster. Utilisation, memory, processes, out-of-memory
  errors and job accounting are all real.
* The arithmetic runs on the CPU. **No timing you measure here says anything
  about GPU performance.** Nothing in this workshop asks you to time anything,
  and if you find yourself comparing two runs by their wall-clock time, stop —
  that is the one question this environment cannot answer.
* The node has one of every card on Mahuika — `l4`, `a100_40`, `a100`, `h100`
  and `pro_6000` — so `--gpus-per-node a100:1` gets you an A100 and asking for
  a card that is not there is refused, as on the cluster. `sinfo -l` lists them.
* Each reports **1 GB of VRAM**, not the 24 to 96 GB the real boards have.
  That is deliberate: it makes running out of memory something you can do in
  a few seconds with a tensor that costs the session almost nothing.

Everything you learn about *reading* these tools transfers unchanged. Nothing
you learn about speed does.
