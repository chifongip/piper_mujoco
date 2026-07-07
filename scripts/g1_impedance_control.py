"""Impedance control for the Unitree G1 upper body (arms only, fixed base).

Joint-space impedance control using Pinocchio for mass matrix and bias forces.
"""

import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


class G1Robot:
  """Unitree G1 23DOF arm impedance controller.

  Attributes:
    model: MuJoCo model.
    data: MuJoCo data.
    model_pin: Pinocchio model (from standalone g1_23dof.xml, no ghost joints).
    data_pin: Pinocchio data.
    q_target_left: Target joint positions for left arm (5 DOF).
    q_target_right: Target joint positions for right arm (5 DOF).
  """

  TIMESTEP = 0.002
  DRAG_THRESHOLD = 15.0  # > Kp * autonomous_threshold (60 * 0.1 = 6.0)
  IK_REACHABILITY_THRESHOLD = 0.05  # 5cm tolerance

  # Arm joint names (5 per side, physical joints only)
  LEFT_ARM_JOINTS = [
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
  ]
  RIGHT_ARM_JOINTS = [
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
  ]

  # End-effector body names for IK
  LEFT_EE_BODY = "left_wrist_roll_rubber_hand"
  RIGHT_EE_BODY = "right_wrist_roll_rubber_hand"

  # Impedance gains (uniform across all arm joints)
  KP = 60.0
  KD = 1.5

  # Home pose (all zeros)
  HOME_POSE = np.zeros(5)

  def __init__(self, scene_xml: str, robot_xml: str):
    """Initialize MuJoCo and Pinocchio models.

    Args:
      scene_xml: Path to the scene XML (with weld constraint, sensors, etc.).
      robot_xml: Path to the standalone robot XML (for Pinocchio dynamics).
    """
    # MuJoCo
    self.model = mujoco.MjModel.from_xml_path(scene_xml)
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = self.TIMESTEP

    # Pinocchio (standalone robot XML, no ghost joints)
    self.model_pin = pin.buildModelFromMJCF(robot_xml)
    self.data_pin = self.model_pin.createData()
    # Match gravity direction
    self.model_pin.gravity.linear = self.model.opt.gravity

    # Pinocchio frame IDs for end-effectors
    self.ee_frame_ids = {
        "left": self.model_pin.getFrameId(self.LEFT_EE_BODY),
        "right": self.model_pin.getFrameId(self.RIGHT_EE_BODY),
    }

    # Pinocchio joint IDs for arm joints (for mass matrix slicing)
    self.pin_arm_joint_ids = {
        "left": [self.model_pin.getJointId(name) for name in self.LEFT_ARM_JOINTS],
        "right": [self.model_pin.getJointId(name) for name in self.RIGHT_ARM_JOINTS],
    }

    # Build joint index maps: joint name -> (qpos_idx, qvel_idx, ctrl_idx)
    self._build_index_maps()

    # Control targets
    self.q_target_left = np.copy(self.HOME_POSE)
    self.q_target_right = np.copy(self.HOME_POSE)
    self.is_autonomous = {"left": False, "right": False}
    self.visual_goal_xyz = {"left": None, "right": None}

    mujoco.mj_forward(self.model, self.data)

  def _build_index_maps(self):
    """Build mappings from joint names to MuJoCo qpos/qvel/ctrl indices."""
    self.joint_qpos = {}
    self.joint_qvel = {}
    self.joint_ctrl = {}

    for i in range(self.model.njnt):
      name = self.model.joint(i).name
      self.joint_qpos[name] = self.model.jnt_qposadr[i]
      self.joint_qvel[name] = self.model.jnt_dofadr[i]

    for i in range(self.model.nu):
      name = self.model.actuator(i).name
      self.joint_ctrl[name] = i

    # Arm qpos/qvel index arrays for fast slicing
    self.left_arm_qpos = np.array([self.joint_qpos[n] for n in self.LEFT_ARM_JOINTS])
    self.left_arm_qvel = np.array([self.joint_qvel[n] for n in self.LEFT_ARM_JOINTS])
    self.right_arm_qpos = np.array([self.joint_qpos[n] for n in self.RIGHT_ARM_JOINTS])
    self.right_arm_qvel = np.array([self.joint_qvel[n] for n in self.RIGHT_ARM_JOINTS])

    # Arm ctrl index arrays (actuator names omit "_joint" suffix)
    left_arm_act = [n.replace("_joint", "") for n in self.LEFT_ARM_JOINTS]
    right_arm_act = [n.replace("_joint", "") for n in self.RIGHT_ARM_JOINTS]
    self.left_arm_ctrl = np.array([self.joint_ctrl[n] for n in left_arm_act])
    self.right_arm_ctrl = np.array([self.joint_ctrl[n] for n in right_arm_act])

    # Pinocchio arm velocity indices (nv offsets)
    self.left_arm_pin_nv = np.array(
        [self.model_pin.joints[jid].idx_v for jid in self.pin_arm_joint_ids["left"]]
    )
    self.right_arm_pin_nv = np.array(
        [self.model_pin.joints[jid].idx_v for jid in self.pin_arm_joint_ids["right"]]
    )

  def _compute_impedance_torques(self, side: str) -> np.ndarray:
    """Joint-space impedance control using Pinocchio (matching Piper script).

    Computes full-system tau = M @ a_des + bias, then maps to actuator ctrl.
    Non-arm DOFs get a_des=0, so they receive only gravity compensation.

    Returns torques for ALL actuators (29 elements).
    """
    q_target = self.q_target_left if side == "left" else self.q_target_right
    arm_qpos = self.left_arm_qpos if side == "left" else self.right_arm_qpos
    arm_qvel = self.left_arm_qvel if side == "left" else self.right_arm_qvel
    is_auto = self.is_autonomous[side]

    q_arm = self.data.qpos[arm_qpos]
    v_arm = self.data.qvel[arm_qvel]
    q_error = q_target - q_arm
    v_error = -v_arm
    a_des_arm = self.KP * q_error + self.KD * v_error

    # Lead-through / drag detection
    if is_auto:
      # Only exit autonomous when arm has settled (small error AND small velocity)
      if np.linalg.norm(q_error) < 0.1 and np.linalg.norm(v_arm) < 0.1:
        self.is_autonomous[side] = False
    else:
      if np.linalg.norm(a_des_arm) > self.DRAG_THRESHOLD:
        if side == "left":
          self.q_target_left = q_arm.copy()
        else:
          self.q_target_right = q_arm.copy()
        a_des_arm = self.KD * v_error

    # Pinocchio dynamics (same as Piper: crba + rnea)
    nq = self.model_pin.nq
    nv = self.model_pin.nv
    q_pin = self.data.qpos[:nq]
    v_pin = self.data.qvel[:nv]
    pin.crba(self.model_pin, self.data_pin, q_pin)
    bias = pin.rnea(self.model_pin, self.data_pin, q_pin, v_pin,
                    np.zeros(nv))

    # Full desired acceleration: arm gets impedance, rest gets zero
    nv_idx = self.left_arm_pin_nv if side == "left" else self.right_arm_pin_nv
    a_des_full = np.zeros(nv)
    a_des_full[nv_idx] = a_des_arm

    # Impedance law (Piper pattern): tau = M @ a_des + bias
    tau_full = self.data_pin.M @ a_des_full + bias

    # Map Pinocchio nv -> MuJoCo ctrl for each actuator
    ctrl = np.zeros(self.model.nu)
    for i in range(self.model.nu):
      jnt_id = self.model.actuator_trnid[i, 0]
      dof_id = self.model.jnt_dofadr[jnt_id]
      if dof_id < nv:
        ctrl[i] = tau_full[dof_id]

    return ctrl

  def compute_ik(self, target_pos: np.ndarray, side: str) -> np.ndarray:
    """Compute IK for the target end-effector position.

    Args:
      target_pos: Desired 3D position [x, y, z].
      side: "left" or "right".

    Returns:
      5-element joint angle array, or current pose if unreachable.
    """
    q = self.data.qpos[:self.model_pin.nq].copy()
    ee_id = self.ee_frame_ids[side]

    eps = 1e-4
    damp = 1e-6
    max_iter = 200
    dt = 0.1

    for _ in range(max_iter):
      pin.forwardKinematics(self.model_pin, self.data_pin, q)
      pin.updateFramePlacements(self.model_pin, self.data_pin)

      err = target_pos - self.data_pin.oMf[ee_id].translation
      if np.linalg.norm(err) < eps:
        break

      J = pin.computeFrameJacobian(
          self.model_pin, self.data_pin, q, ee_id,
          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
      )[:3, :]  # Position only

      # Damped least squares
      v = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
      q = pin.integrate(self.model_pin, q, v * dt)

      # Clamp to joint limits
      q = np.clip(q, self.model_pin.lowerPositionLimit,
                  self.model_pin.upperPositionLimit)

    # Reachability check
    pin.forwardKinematics(self.model_pin, self.data_pin, q)
    pin.updateFramePlacements(self.model_pin, self.data_pin)
    final_pos = self.data_pin.oMf[ee_id].translation
    err_norm = np.linalg.norm(target_pos - final_pos)

    if err_norm > self.IK_REACHABILITY_THRESHOLD:
      print(f"[{side}] REJECTED: target {target_pos} unreachable (err={err_norm:.3f}m)")
      arm_qpos = self.left_arm_qpos if side == "left" else self.right_arm_qpos
      return self.data.qpos[arm_qpos].copy()

    print(f"[{side}] ACCEPTED: target reachable (err={err_norm:.4f}m)")
    # Extract arm joint angles from Pinocchio q
    arm_pin_nv = self.left_arm_pin_nv if side == "left" else self.right_arm_pin_nv
    pin_q_idx = [nv + 6 for nv in arm_pin_nv]  # qpos_idx = nv_idx + 6 for single-DOF
    # Actually, need to use joint idx_q for each arm joint
    arm_joint_ids = self.pin_arm_joint_ids[side]
    pin_q_idx = [self.model_pin.joints[jid].idx_q for jid in arm_joint_ids]
    return q[pin_q_idx]

  def step(self):
    """Execute one control + simulation step."""
    # Compute full-system torques (Pinocchio impedance for each arm)
    ctrl_left = self._compute_impedance_torques("left")
    ctrl_right = self._compute_impedance_torques("right")

    # Merge: left arm from left, right arm from right, rest from left
    ctrl = ctrl_left.copy()
    ctrl[self.right_arm_ctrl] = ctrl_right[self.right_arm_ctrl]

    self.data.ctrl[:] = ctrl

    mujoco.mj_step(self.model, self.data)


def handle_user_input(robot: G1Robot):
  """Terminal command interface."""
  print("\n--- G1 Arm Impedance Controller ---")
  print('Commands: "home", "exit", "left X Y Z", "right X Y Z"')
  print('Example: left 0.3 0.2 0.5\n')

  while True:
    try:
      raw = input(">> ").strip().lower()
      if raw == "exit":
        break
      if raw == "home":
        robot.q_target_left = np.copy(robot.HOME_POSE)
        robot.q_target_right = np.copy(robot.HOME_POSE)
        robot.is_autonomous = {"left": True, "right": True}
        robot.visual_goal_xyz = {"left": None, "right": None}
        print("Arms returning to home pose.")
        continue

      parts = raw.split()
      if len(parts) == 4 and parts[0] in ("left", "right"):
        side = parts[0]
        coords = np.array([float(x) for x in parts[1:]])
        robot.visual_goal_xyz[side] = coords
        q_target = robot.compute_ik(coords, side)
        if side == "left":
          robot.q_target_left = q_target
        else:
          robot.q_target_right = q_target
        robot.is_autonomous[side] = True
      else:
        print('Error: Use "home", "exit", or "left/right X Y Z".')
    except (ValueError, EOFError):
      print("Error: Invalid input.")


def main():
  scene_xml = os.path.join(_REPO_ROOT, "assets/robots/unitree_g1/scene_g1_23dof_upper.xml")
  robot_xml = os.path.join(_REPO_ROOT, "assets/robots/unitree_g1/g1_23dof_upper.xml")

  robot = G1Robot(scene_xml, robot_xml)

  print("G1 Impedance Control Active.")
  print("Fixed base, upper body only. Arms at home pose (all zeros).")
  print(f"Gains: Kp={robot.KP}, Kd={robot.KD}")

  threading.Thread(target=handle_user_input, args=(robot,), daemon=True).start()

  with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    while viewer.is_running():
      step_start = time.time()
      robot.step()

      # Draw goal markers
      ngeom = 0
      for side, color in [("left", [0, 0, 1, 0.5]), ("right", [1, 0, 0, 0.5])]:
        if robot.visual_goal_xyz[side] is not None:
          mujoco.mjv_initGeom(
              viewer.user_scn.geoms[ngeom],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.02, 0, 0],
              pos=robot.visual_goal_xyz[side],
              mat=np.eye(3).flatten(),
              rgba=color,
          )
          ngeom += 1
      viewer.user_scn.ngeom = ngeom

      viewer.sync()
      elapsed = time.time() - step_start
      if elapsed < robot.TIMESTEP:
        time.sleep(robot.TIMESTEP - elapsed)


if __name__ == "__main__":
  main()
