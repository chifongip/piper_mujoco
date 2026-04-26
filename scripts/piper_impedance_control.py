"""A module for reachability-validated IK for the Piper robot.

This version implements strict error checking after the IK solver 
to prevent the robot from attempting to reach unreachable poses.
"""

import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class PiperRobot:
  """Manages the Piper robot simulation with strict reachability validation.

  Attributes:
    model_mj: The MuJoCo model instance.
    data_mj: The MuJoCo data instance.
    model_pin: The Pinocchio model instance.
    data_pin: The Pinocchio data instance.
    ee_frame_id: Frame ID of the end-effector in Pinocchio.
    q_target_arm: Desired joint positions for the 6-DOF arm.
    is_autonomous: Flag to bypass drag thresholds during commanded moves.
    visual_goal_xyz: The 3D coordinate used for the visual marker.
  """

  # --- Constants ---
  TIMESTEP = 0.002
  DRAG_THRESHOLD = 2.5
  HOME_POSE = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0])
  IK_REACHABILITY_THRESHOLD = 0.05  # 5cm tolerance

  # Impedance Gains
  KP_ARM = np.array([250.0, 450.0, 300.0, 100.0, 100.0, 50.0])
  KD_ARM = np.array([15.0, 25.0, 15.0, 4.0, 4.0, 1.5])
  KP_GRIPPER = np.array([500.0, 500.0])
  KD_GRIPPER = np.array([10.0, 10.0])

  def __init__(self, mjcf_path, robot_mjcf_path):
    """Initializes MuJoCo and Pinocchio models and internal states."""
    self.model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
    self.data_mj = mujoco.MjData(self.model_mj)
    self.model_mj.opt.timestep = self.TIMESTEP

    self.model_pin = pin.buildModelFromMJCF(robot_mjcf_path)
    self.data_pin = self.model_pin.createData()
    self.model_pin.gravity.linear = self.model_mj.opt.gravity

    self.ee_frame_id = self.model_pin.getFrameId(self.model_pin.frames[-1].name)

    self.q_target_arm = np.copy(self.HOME_POSE)
    self.gripper_width_target = 0.0
    self.is_autonomous = False
    self.visual_goal_xyz = None

    self.data_mj.qpos[:6] = self.HOME_POSE
    mujoco.mj_forward(self.model_mj, self.data_mj)

  def compute_ik(self, target_pos, max_iter=200, dt=0.1):
    """Calculates joint angles with strict physical reachability validation."""
    q = np.copy(self.data_mj.qpos[:self.model_pin.nq])
    eps = 1e-4
    damp = 1e-6

    for _ in range(max_iter):
      pin.forwardKinematics(self.model_pin, self.data_pin, q)
      pin.updateFramePlacements(self.model_pin, self.data_pin)

      o_m_ee = self.data_pin.oMf[self.ee_frame_id]
      err = target_pos - o_m_ee.translation

      if np.linalg.norm(err) < eps:
        break

      jacobian = pin.computeFrameJacobian(
          self.model_pin, self.data_pin, q, self.ee_frame_id,
          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]

      # Damped Least Squares update
      v = jacobian.T @ np.linalg.solve(
          jacobian @ jacobian.T + damp * np.eye(3), err)
      q = pin.integrate(self.model_pin, q, v * dt)
      
      # Enforce joint limits to prevent unrealistic "stretching"
      q = np.clip(q, self.model_pin.lowerPositionLimit, 
                  self.model_pin.upperPositionLimit)

    # --- REACHABILITY VALIDATION ---
    pin.forwardKinematics(self.model_pin, self.data_pin, q)
    pin.updateFramePlacements(self.model_pin, self.data_pin)
    final_ee_pos = self.data_pin.oMf[self.ee_frame_id].translation
    real_distance_error = np.linalg.norm(target_pos - final_ee_pos)

    if real_distance_error > self.IK_REACHABILITY_THRESHOLD:
      print(f'REJECTED: Target {target_pos} is unreachable.')
      print(f'Physical distance error: {real_distance_error:.3f}m')
      self.visual_goal_xyz = None  # Hide visual marker on failure
      return self.data_mj.qpos[:6]  # Return current pose to stay put

    print(f'ACCEPTED: Target reachable (Error: {real_distance_error:.4f}m)')
    return q[:6]

  def _compute_impedance_torques(self):
    """Computes Joint-Space Impedance Control torques."""
    q_full = self.data_mj.qpos[:self.model_pin.nq]
    v_full = self.data_mj.qvel[:self.model_pin.nv]

    pin.crba(self.model_pin, self.data_pin, q_full)
    m_mat = self.data_pin.M[:6, :6]
    bias = pin.rnea(self.model_pin, self.data_pin, q_full, v_full,
                    np.zeros(self.model_pin.nv))

    q_error = self.q_target_arm - q_full[:6]
    v_error = -v_full[:6]
    a_des = (self.KP_ARM * q_error) + (self.KD_ARM * v_error)

    if self.is_autonomous:
      if np.linalg.norm(q_error) < 0.02:
        self.is_autonomous = False
    else:
      if np.linalg.norm(a_des) > self.DRAG_THRESHOLD:
        self.q_target_arm = np.copy(q_full[:6])
        a_des = self.KD_ARM * v_error

    return (m_mat @ a_des) + bias[:6]

  def _compute_gripper_torques(self):
    """Calculates PD torques for the gripper fingers."""
    q_target = np.array([self.gripper_width_target, -self.gripper_width_target])
    error = q_target - self.data_mj.qpos[6:8]
    v_error = -self.data_mj.qvel[6:8]
    return (self.KP_GRIPPER * error) + (self.KD_GRIPPER * v_error)

  def step(self):
    """Performs a single control and simulation step."""
    arm_tau = self._compute_impedance_torques()
    gripper_tau = self._compute_gripper_torques()
    self.data_mj.ctrl[:6] = arm_tau
    self.data_mj.ctrl[6:8] = gripper_tau
    mujoco.mj_step(self.model_mj, self.data_mj)


def handle_user_input(robot):
  """Handles terminal commands in a background thread."""
  print('\n--- Piper Controller (Safety Enabled) ---')
  print('Usage: "home", "exit", or "X Y Z" (e.g., 0.3 0 0.4)')

  while True:
    try:
      raw_input = input('>> ').strip().lower()
      if raw_input == 'exit':
        break
      if raw_input == 'home':
        robot.is_autonomous = True
        robot.q_target_arm = np.copy(robot.HOME_POSE)
        robot.visual_goal_xyz = None
        continue

      coords = np.array([float(x) for x in raw_input.split()])
      if len(coords) == 3:
        robot.visual_goal_xyz = coords
        robot.q_target_arm = robot.compute_ik(coords)
        robot.is_autonomous = True
      else:
        print('Error: Enter 3 coordinates.')
    except (ValueError, EOFError):
      print('Error: Invalid format.')


def main():
  """Main entry point for simulation and rendering."""
  mjcf_path = '../assets/robots/piper/scene.xml'
  robot_mjcf_path = '../assets/robots/piper/piper.xml'
  robot = PiperRobot(mjcf_path, robot_mjcf_path)

  threading.Thread(target=handle_user_input, args=(robot,), daemon=True).start()

  with mujoco.viewer.launch_passive(robot.model_mj, robot.data_mj) as viewer:
    while viewer.is_running():
      step_start = time.time()
      robot.step()

      if robot.visual_goal_xyz is not None:
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=robot.visual_goal_xyz,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 0.5])
      else:
        viewer.user_scn.ngeom = 0

      viewer.sync()
      elapsed = time.time() - step_start
      if elapsed < robot.TIMESTEP:
        time.sleep(robot.TIMESTEP - elapsed)


if __name__ == '__main__':
  main()