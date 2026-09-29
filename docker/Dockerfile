FROM ubuntu:24.04

# install system dependencies
# (ldap-utils, libnss-ldapd, libpam-ldapd, nscd, nslcd are needed to resolve
#  training users against the environment's LDAP, as in the other apps)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
        dnsutils \
        gnupg \
        htop \
        locales \
        ldap-utils \
        libnss-ldapd \
        libpam-ldapd \
        less \
        nano \
        nscd \
        nslcd \
        openssh-client \
        pciutils \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync \
        time \
        unzip \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_NZ -c -f UTF-8 -A /usr/share/locale/locale.alias en_NZ.UTF-8 \
    # 24.04 ships an "ubuntu" user at uid 1000; drop it so the uid can only
    # resolve to a training user through LDAP
    && userdel -r ubuntu

ENV LANG=en_NZ.utf8

# install kubectl, required for running on the k8s cluster
ARG KUBECTL_VERSION=v1.31.8
RUN curl -LO https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl \
    && mv kubectl /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl

# copy in ood k8s utils
ARG UTILS_HASH=6298fb01f7a7c66a8454e3f0fd74437a32491423
RUN git clone https://github.com/nesi/training-environment-k8s-utils.git /opt/ood-k8s-utils \
    && cd /opt/ood-k8s-utils \
    && git checkout $UTILS_HASH \
    && chmod +x /opt/ood-k8s-utils/files/* \
    && mv /opt/ood-k8s-utils/files/* /bin/ \
    && rm -rf /opt/ood-k8s-utils

# make a dummy module command to avoid warnings from ondemand job_script_content.sh
RUN echo "#!/bin/bash" > /bin/module \
    && chmod +x /bin/module

# ---------------------------------------------------------------------------
# VS Code (code-server) and the tools the Better Research Software lesson uses
# ---------------------------------------------------------------------------
ARG CODE_RELEASE=4.139.1
RUN mkdir -p /app/code-server \
    && curl -fsSL -o /tmp/code-server.tar.gz \
        "https://github.com/coder/code-server/releases/download/v${CODE_RELEASE}/code-server-${CODE_RELEASE}-linux-amd64.tar.gz" \
    && tar xf /tmp/code-server.tar.gz -C /app/code-server --strip-components=1 \
    && rm -f /tmp/code-server.tar.gz \
    && ln -s /app/code-server/bin/code-server /usr/local/bin/code-server \
    && code-server --version

# Extensions are installed once here and copied into each user's extensions
# directory on first start, so learners can still add their own. Pylance is
# Microsoft-only and not on Open VSX; basedpyright stands in as the language server.
RUN for ext in \
        ms-python.python@2026.4.0 \
        ms-python.debugpy@2026.6.0 \
        detachhead.basedpyright@1.40.1 \
        njpwerner.autodocstring@0.6.1 \
        mhutchie.git-graph@1.30.0 \
        ms-vscode.live-server@0.4.20 \
    ; do code-server --extensions-dir /opt/code-server/extensions --install-extension "$ext" || exit 1; done \
    && chmod -R a+rX /opt/code-server \
    && code-server --extensions-dir /opt/code-server/extensions --list-extensions --show-versions

# GitHub CLI, for learners whose network blocks GitHub SSH entirely
RUN curl -fsSL -o /usr/share/keyrings/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/* \
    && gh --version

# Git defaults the lesson's setup check expects; learners still set their own
# user.name and user.email
RUN git config --system init.defaultBranch main \
    && git config --system core.editor "nano -w"

# GitHub SSH goes over port 443 (ssh.github.com), which works where outbound 22
# is blocked, and GitHub's published host keys are pinned so learners are not
# asked to trust an unverified fingerprint. The keys are committed rather than
# fetched from api.github.com at build time, which rate-limits CI runners (403);
# check them against https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
COPY ssh/github.conf /etc/ssh/ssh_config.d/github.conf
COPY ssh/known_hosts /etc/ssh/ssh_known_hosts
RUN chmod 0644 /etc/ssh/ssh_config.d/github.conf /etc/ssh/ssh_known_hosts \
    && ssh-keygen -lf /etc/ssh/ssh_known_hosts | grep -q "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"

COPY code-server/settings.json /opt/code-server/settings.json

# ---------------------------------------------------------------------------
# nvtop
#
# Built from source rather than installed from apt. The jammy package is 1.x,
# which predates the dlopen-based NVIDIA backend this image relies on, and it
# also drags in a large set of AMD/Intel GPU dependencies we have no use for.
# Building with only the NVIDIA backend enabled keeps it small and guarantees
# it goes through libnvidia-ml.so.1, which is what our shim provides.
# ---------------------------------------------------------------------------
ARG NVTOP_VERSION=3.1.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake libncurses-dev libdrm-dev \
    && git clone --depth 1 --branch $NVTOP_VERSION https://github.com/Syllo/nvtop.git /tmp/nvtop \
    && mkdir -p /tmp/nvtop/build \
    && cd /tmp/nvtop/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DNVIDIA_SUPPORT=ON \
        -DAMDGPU_SUPPORT=OFF \
        -DINTEL_SUPPORT=OFF \
        -DMSM_SUPPORT=OFF \
        -DPANFROST_SUPPORT=OFF \
        -DPANTHOR_SUPPORT=OFF \
        -DASCEND_SUPPORT=OFF \
        -DTPU_SUPPORT=OFF \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/nvtop \
    && apt-get purge -y cmake && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python stack for the emulators
#
# This comes before the emulator source is copied so that editing the emulator
# does not invalidate a two-gigabyte download. torch is the CPU build: the CUDA
# build would be far larger and could not run anyway, since there is no device.
# gpuemu.torch_shim is what makes CUDA-flavoured code work on top of it.
# ---------------------------------------------------------------------------

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668). The
# emulator stack is installed into it deliberately, so these builds pass
# --break-system-packages; learners are left with the default, which steers
# `pip install` into the venv the lesson teaches.
RUN pip3 --no-cache-dir install --break-system-packages \
        matplotlib \
        numpy \
        pandas \
        nvidia-ml-py \
    && pip3 --no-cache-dir install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip3 --no-cache-dir install --break-system-packages numba pytest pyyaml

# Numba's CUDA simulator runs @cuda.jit kernels in the interpreter, so learners
# can write and debug real CUDA kernel code - threads, blocks, shared memory,
# syncthreads - and have it execute correctly, if slowly.
ENV NUMBA_ENABLE_CUDASIM=1

# ---------------------------------------------------------------------------
# The GPU emulator
#
# libnvidia-ml.so.1 is installed to /opt/gpuemu/lib and put on the loader path
# through /etc/ld.so.conf.d. Everything that looks for a GPU - nvtop, pynvml,
# anything else that dlopens NVML - finds it there and sees an L4.
# ---------------------------------------------------------------------------
COPY gpuemu /opt/gpuemu/src

RUN cd /opt/gpuemu/src/nvml \
    && make PREFIX=/opt/gpuemu install \
    && echo "/opt/gpuemu/lib" > /etc/ld.so.conf.d/gpuemu.conf \
    && ldconfig \
    && cd /opt/gpuemu/src \
    && pip3 --no-cache-dir install --break-system-packages . \
    && install -m 0755 /opt/gpuemu/src/bin/nvidia-smi /usr/local/bin/nvidia-smi \
    && install -d /run/gpuemu /run/gpuemu/claims /run/gpuemu/slurm/jobs \
    && chmod 1777 /run/gpuemu /run/gpuemu/claims /run/gpuemu/slurm /run/gpuemu/slurm/jobs

# Verify the emulator at build time rather than discovering it in front of a
# workshop. The C and Python views of the shared state must agree, the shim must
# load and enumerate a device, torch must see the emulated card and refuse an
# oversized allocation, and a real @cuda.jit kernel must execute.
RUN cd /opt/gpuemu/src \
    && python3 -m pytest tests -q \
    && gpuemud --once \
    && nvidia-smi -L \
    && nvidia-smi \
    && seff -h | grep -q "Usage: seff" \
    && svisit -h | grep -q "svisit \[-t\]" \
    && python3 -c "import ctypes; l=ctypes.CDLL('libnvidia-ml.so.1'); assert l.nvmlInit_v2()==0; n=ctypes.c_uint(); assert l.nvmlDeviceGetCount_v2(ctypes.byref(n))==0; assert n.value==1, n.value; print('NVML shim OK')" \
    && python3 -c "\
import numpy as np; from numba import cuda;\
exec('@cuda.jit\ndef add(a,b,o):\n i=cuda.grid(1)\n if i<o.size: o[i]=a[i]+b[i]\n');\
a=np.arange(64,dtype=np.float32); b=a*2; o=np.zeros_like(a);\
add[2,32](a,b,o);\
assert np.allclose(o,a+b), o;\
print('numba cudasim OK')" \
    && pip3 uninstall -y --break-system-packages pytest \
    && rm -rf /run/gpuemu/state.bin /opt/gpuemu/src/.pytest_cache

# Claim the device automatically when a process imports torch or numba.cuda, so
# a learner's notebook shows up in nvidia-smi without any boilerplate.
ENV GPUEMU_AUTOCLAIM=1
ENV GPUEMU_STATE_FILE=/run/gpuemu/state.bin
ENV GPUEMU_CLAIMS_DIR=/run/gpuemu/claims
ENV GPUEMU_SLURM_DIR=/run/gpuemu/slurm

# The node's GPUs: one of every card Mahuika has, so `--gpus-per-node a100:1`
# and `--gpus-per-node l4:1` are genuinely different requests and choosing
# between cards is something a learner can do here rather than only read about.
# The app form overrides this; entries are name[:count][:vram].
#
# Each is deliberately shrunk to 200 MB of VRAM. A small card is what makes
# memory pressure teachable: a learner hits a real out-of-memory error with a
# tensor that costs the session almost nothing, rather than needing to allocate
# tens of gigabytes of host RAM to reach the limit. Every exercise is sized to
# fit, with room to spare. Say `l4:1GiB` for a roomier card, or `l4:full` for
# the real 24 GB.
ENV GPUEMU_FLEET=l4,a100_40,a100,h100,pro_6000

# The older single-card spelling, still read when GPUEMU_FLEET is unset.
ENV GPUEMU_MEM_TOTAL=200MiB

# Install the autoclaim hook into every Python process.
#
# This used to be written as sitecustomize.py, which did not work: Ubuntu ships
# its own /usr/lib/python3.10/sitecustomize.py, that directory comes earlier on
# sys.path, and only the first one found is imported. Ours was never loaded, so
# `import torch` got no shim and torch.cuda.is_available() was False inside
# every job - the exact symptom this workshop teaches as "you forgot to ask for
# a GPU", produced by a job that had one.
#
# A .pth file is the right mechanism: site processes *every* .pth in *every*
# site directory, so this cannot be shadowed. It goes in whichever directory
# gpuemu itself was installed into, since that is guaranteed to be a site
# directory on this interpreter's path. The zz- prefix makes it run last, after
# anything that adjusts sys.path.
RUN SITEDIR="$(python3 -c 'import gpuemu, pathlib; print(pathlib.Path(gpuemu.__file__).parent.parent)')" \
    && printf '%s\n' \
        'try:' \
        '    from gpuemu.autoclaim import install as _install' \
        '    _install()' \
        'except Exception:' \
        '    pass' \
        > "${SITEDIR}/gpuemu_autostart.py" \
    && echo 'import gpuemu_autostart' > "${SITEDIR}/zz-gpuemu-autostart.pth" \
    && python3 -c "import sys; assert 'gpuemu_autostart' in sys.modules, 'the .pth did not run'" \
    && rm -f /usr/local/lib/python3*/dist-packages/sitecustomize.py

# A short banner so anyone opening a terminal knows the GPU is emulated. Being
# quiet about that would be the one genuinely harmful thing this image could do.
COPY profile.d/gpuemu.sh /etc/profile.d/gpuemu.sh
RUN chmod 0644 /etc/profile.d/gpuemu.sh

EXPOSE 8443
