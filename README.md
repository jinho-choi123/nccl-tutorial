## NCCL Tutorial

### Prerequisites
- Environment with 2 NVIDIA GPUs
- CUDA Toolkit(>=11.8)
- NCCL(>=2.18.3)

### Setup
```bash
uv sync
```

### Project Structure

Scenario 1: single thread - multiple devices (st_md)

Scenario 2: multiple thread - multiple device (mt_md, each thread has its own device)

### Run st_md
```bash
mkdir build
cd build
cmake ..
make
./st_md/st_md_main
```

### Run mt_md
```bash
mkdir build
cd build
cmake ..
make

# For 2 GPUs
mpirun -n 2 ./mt_md/mt_md_main

# For n_gpus GPUs
mpirun -n <n_gpus> ./mt_md/mt_md_main
```

### Profiling with Nsight System

```bash
sudo nsys profile \
--trace nvtx \
--cuda-memory-usage true \
--env-var NCCL_P2P_DISABLE=1 \
--force-overwrite true \
--output profile_nccl_tutorial \
/root/nccl-tutorial/.venv/bin/mpirun -n 4 --allow-run-as-root ./mt_md/mt_md_main
```