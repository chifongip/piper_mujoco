# piper_mujoco

MuJoCo simulation for robot arm impedance control.

## Robots

| Robot | DOF | Base | Scripts |
|-------|-----|------|---------|
| Piper | 6 arm + 2 gripper | Fixed | `piper_*.py` |
| Unitree G1 | 10 (5 per arm) | Fixed | `g1_impedance_control.py` |

## Features

- Joint-space impedance control via Pinocchio (mass matrix + bias forces)
- IK with damped least-squares Jacobian and reachability validation
- Lead-through programming: drag the arm, it holds position, target updates
- Gravity compensation via Pinocchio RNEA

## Requirements

- Python 3.11
- MuJoCo 3.7.0
- Pinocchio 4.0.0

## Installation

```bash
conda create -n piper_mujoco python=3.11
conda activate piper_mujoco
conda install pinocchio -c conda-forge
pip install mujoco
```

## Usage

```bash
cd scripts
python3 g1_impedance_control.py       # G1 upper body
python3 piper_impedance_control.py     # Piper arm
```

**Viewer commands:**
- `left 0.3 0.2 0.5` — move left arm end-effector to XYZ position
- `right 0.3 0.2 0.5` — move right arm
- `home` — return both arms to home pose
- `exit` — quit

**Lead-through:** Physically drag an arm in the viewer. The arm holds wherever you leave it.

## Project Structure

```
assets/robots/
  piper/                    # Piper MJCF + meshes
  unitree_g1/               # G1 MJCF + STL meshes
    scene_g1_23dof_upper.xml   # Upper body scene (fixed base, arms only)
    g1_23dof_upper.xml         # Standalone for Pinocchio (no ghost joints)
    scene_g1_23dof.xml         # Full body scene (floating base, with legs)
scripts/
  g1_impedance_control.py   # G1 impedance control
  piper_impedance_control.py
  piper_gravity_compensation.py
  piper_gravity_compensation_sim.py
  piper_gravity_compensation_impedance_control.py
```
