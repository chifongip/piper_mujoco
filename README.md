# piper_mujoco
This repository contains the PD control and impedance control implementations for the Piper robot in the MuJoCo simulation environment.

# Requirements
- Python 3.11
- MuJoCo 3.7.0
- Pinocchio 4.0.0

# Installation
1. Create conda environment:
```bash
conda create -n piper_mujoco python=3.11
conda activate piper_mujoco
```

2. Install Pinocchio:
```bash
conda install pinocchio -c conda-forge
```

3. Install piper_mujoco:
```bash
git clone https://github.com/yourusername/piper_mujoco.git
cd piper_mujoco
pip install -r requirements.txt
```
