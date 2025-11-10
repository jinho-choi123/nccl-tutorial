## NCCL Tutorial

### Prerequisites
- Use runpod.io with A5000 x 2 or A5000 x 4 GPUs
- CUDA Toolkit(11.8)
- NCCL(2.18.3)
- OpenMPI

### Install OpenMPI
```bash
sudo apt install openmpi-bin libopenmpi-dev -y
```

### Setup
```bash
uv sync
```

### Project Structure

Scenario 1: single thread - multiple devices (st_md)

Scenario 2: multiple thread - multiple device (mt_md, each thread has its own device)
