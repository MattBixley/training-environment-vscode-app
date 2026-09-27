#!/bin/bash -e
#SBATCH --job-name      gpu-test
#SBATCH --account       nesi99991
#SBATCH --time          00:15:00        # a test job, not a real one
#SBATCH --qos           debug           # high priority, short limit
#SBATCH --gpus-per-node l4:1
#SBATCH --cpus-per-task 2
#SBATCH --mem           4GB
#SBATCH --profile       task            # record a time series
#SBATCH --acctg-freq    1               # one sample per second
#SBATCH --output        gpu-test-%j.out

# The shape of a test submit script. Four of these lines exist only because
# this is a test, and all four should come out again before the script
# becomes a real job:
#
#   --time 00:15:00   long enough to reach a steady state, short enough that
#                     a bad guess costs you fifteen minutes
#   --qos debug       starts almost immediately; strictly limited in return.
#                     Left in a production script, your job is killed at the
#                     debug walltime limit
#   --profile task    records what the job did over time, rather than only a
#                     summary at the end
#   --acctg-freq 1    samples that record once a second instead of once every
#                     thirty. Fine for fifteen minutes, wasteful for a week
#
# Submit it, then read it back:
#
#   sbatch test-job.sl
#   squeue --me                  # wait for it to finish
#   seff <JOBID>                 # one number per resource
#   profile_plot <JOBID>         # the same job, over time
#
# seff answers "how much did it use?". profile_plot answers "when?". A job
# averaging 40% might have held a steady 40% or alternated between 100% and
# 0%, and only the plot tells you which.

python3 ../07_putting_it_together/sawtooth.py --minutes 3
