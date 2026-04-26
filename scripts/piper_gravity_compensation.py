"""A module for PD control and lead-through programming using Pinocchio dynamics."""

import time
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class PiperRobot:
    """Manages the Piper robot simulation with Pinocchio-based dynamics.
    
    Attributes:
        model_mj: The MuJoCo model instance.
        data_mj: The MuJoCo data instance.
        model_pin: The Pinocchio model instance.
        data_pin: The Pinocchio data instance.
    """

    # --- Constants ---
    TIMESTEP: float = 0.002
    DRAG_THRESHOLD: float = 2.5
    HOME_POSE: np.ndarray = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0])

    # PD Gains
    KP_ARM: np.ndarray = np.array([250.0, 450.0, 300.0, 100.0, 100.0, 50.0])
    KD_ARM: np.ndarray = np.array([15.0, 25.0, 15.0, 4.0, 4.0, 1.5])
    KP_GRIPPER: np.ndarray = np.array([500.0, 500.0])
    KD_GRIPPER: np.ndarray = np.array([10.0, 10.0])

    def __init__(self, mjcf_path: str, robot_mjcf_path: str):
        """Initializes MuJoCo and Pinocchio models.

        Args:
            mjcf_path: Path to the full MuJoCo scene XML.
            robot_mjcf_path: Path to the robot MJCF for Pinocchio.
        """
        # MuJoCo Setup
        self.model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data_mj = mujoco.MjData(self.model_mj)
        self.model_mj.opt.timestep = self.TIMESTEP

        # Pinocchio Setup (Using MJCF loader)
        self.model_pin = pin.buildModelFromMJCF(robot_mjcf_path)
        self.data_pin = self.model_pin.createData()
        
        # Ensure gravity is synced (MuJoCo default is [0, 0, -9.81])
        self.model_pin.gravity.linear = self.model_mj.opt.gravity

        # Control targets
        self.q_target_arm = np.copy(self.HOME_POSE)
        self.gripper_width_target = 0.0

        # Initialize physical pose
        self.data_mj.qpos[:6] = self.HOME_POSE
        mujoco.mj_forward(self.model_mj, self.data_mj)

    def _compute_bias_forces(self) -> np.ndarray:
        """Calculates Coriolis, Centripetal, and Gravity forces using Pinocchio.

        Returns:
            A 1D array of bias torque values.
        """
        # Ensure we only pass the dimensions Pinocchio expects
        # Pinocchio model_pin.nq might differ from mj.nq if the scene has extra objects
        nq = self.model_pin.nq
        nv = self.model_pin.nv

        q = self.data_mj.qpos[:nq]
        v = self.data_mj.qvel[:nv]
        a = np.zeros(nv)

        # RNEA computes: C(q, v)v + g(q) when acceleration is zero
        return pin.rnea(self.model_pin, self.data_pin, q, v, a)

    def _compute_arm_torques(self) -> np.ndarray:
        """Computes PD torques for the arm with lead-through logic."""
        q_current = self.data_mj.qpos[:6]
        v_current = self.data_mj.qvel[:6]

        arm_error = self.q_target_arm - q_current
        arm_v_error = 0 - v_current
        
        effort = (self.KP_ARM * arm_error) + (self.KD_ARM * arm_v_error)

        if np.linalg.norm(effort) > self.DRAG_THRESHOLD:
            self.q_target_arm = np.copy(q_current)
            effort = self.KD_ARM * arm_v_error

        return effort

    def _compute_gripper_torques(self) -> np.ndarray:
        """Computes symmetric PD torques for the gripper fingers."""
        q_target = np.array([self.gripper_width_target, -self.gripper_width_target])
        error = q_target - self.data_mj.qpos[6:8]
        v_error = 0 - self.data_mj.qvel[6:8]
        return (self.KP_GRIPPER * error) + (self.KD_GRIPPER * v_error)

    def step(self) -> None:
        """Executes a single simulation step."""
        arm_pd = self._compute_arm_torques()
        gripper_pd = self._compute_gripper_torques()
        
        # Calculate bias via Pinocchio
        bias = self._compute_bias_forces()

        # Apply bias to arm and PD to all
        # We assume bias[:6] maps to the arm joints
        self.data_mj.ctrl[:6] = arm_pd + bias[:6]
        self.data_mj.ctrl[6:8] = gripper_pd

        mujoco.mj_step(self.model_mj, self.data_mj)


def main():
    """Main simulation loop."""
    # Paths (Update these to your local directory structure)
    mjcf_path = '../assets/robots/piper/scene.xml'
    robot_mjcf_path = '../assets/robots/piper/piper.xml'

    robot = PiperRobot(mjcf_path, robot_mjcf_path)

    print(f"Lead-through active. Drag threshold: {robot.DRAG_THRESHOLD}")

    with mujoco.viewer.launch_passive(robot.model_mj, robot.data_mj) as viewer:
        while viewer.is_running():
            step_start = time.time()

            robot.step()
            viewer.sync()

            # Maintain real-time frequency
            elapsed = time.time() - step_start
            if elapsed < robot.TIMESTEP:
                time.sleep(robot.TIMESTEP - elapsed)


if __name__ == '__main__':
    main()