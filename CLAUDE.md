# CLAUDE.md

## Project Overview

MuJoCo simulation for robot arm impedance control. Two robots:
- **Piper** — 6-DOF fixed-base arm + 2-DOF gripper
- **Unitree G1** — 10-DOF upper body (waist removed, fixed base, arms only)

## Key Files

```
scripts/
  piper_impedance_control.py              # Piper: impedance + IK + lead-through
  piper_gravity_compensation.py           # Piper: PD + gravity comp (MuJoCo bias)
  piper_gravity_compensation_sim.py       # Piper: PD + gravity comp (Pinocchio)
  piper_gravity_compensation_impedance_control.py  # Piper: impedance + IK + reachability
  g1_impedance_control.py                 # G1: impedance + IK + lead-through

assets/robots/
  piper/scene.xml, piper.xml              # Piper MJCF
  unitree_g1/
    scene_g1_23dof_upper.xml              # G1 upper body scene (fixed base, arms only)
    g1_23dof_upper.xml                    # G1 standalone for Pinocchio (no ghost joints)
    scene_g1_23dof.xml                    # G1 full body scene (floating base, with legs)
    g1_23dof.xml                          # G1 full body standalone for Pinocchio
```

## Architecture

All scripts follow the same pattern:
1. Load MuJoCo model for simulation, Pinocchio model for dynamics
2. Build joint index maps (joint name → qpos/qvel/ctrl indices)
3. Control loop: compute torques via Pinocchio (crba + rnea), apply to MuJoCo ctrl, step
4. Lead-through: drag detection snaps target to current position when external force detected
5. IK: damped least-squares Jacobian with reachability validation

## Control Law

**Piper (fixed base):** `tau = M @ a_des + bias`
- `M` from `pin.crba`, `bias` from `pin.rnea`
- MuJoCo applies `M*qacc = ctrl + qfrc_bias`, bias terms cancel

**G1 (fixed base, simplified):** Same as Piper: `tau = M @ a_des + bias`
- Fixed base means no floating-base coupling, direct Piper pattern works

## Important Conventions

- MuJoCo actuator names omit `_joint` suffix (e.g., `left_shoulder_pitch` not `left_shoulder_pitch_joint`)
- Ghost joints (wrist_pitch, wrist_yaw, waist_roll, waist_pitch) are detached at z=20, not in kinematic tree
- `g1_23dof_upper.xml` has no ghost joints for Pinocchio; `scene_g1_23dof_upper.xml` has ghost joints for MuJoCo actuators
- Pinocchio qpos/qvel indices match MuJoCo DOF indices for the simplified model

## Running

```bash
cd scripts && python3 g1_impedance_control.py
cd scripts && python3 piper_impedance_control.py
```

Commands: `home`, `exit`, `left X Y Z`, `right X Y Z`

## Dependencies

- Python 3.11
- MuJoCo 3.7.0 (`pip install mujoco`)
- Pinocchio 4.0.0 (`conda install pinocchio -c conda-forge`)
