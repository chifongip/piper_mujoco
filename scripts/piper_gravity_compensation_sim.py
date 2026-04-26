"""A module for PD control and lead-through programming of the Piper robot."""

import mujoco
import mujoco.viewer
import numpy as np
import time


class PiperRobot:
    """Manages the Piper robot simulation and control logic."""

    # --- Constants (Google Style: UPPER_CASE) ---
    TIMESTEP: float = 0.002
    DRAG_THRESHOLD: float = 2.5
    HOME_POSE: np.ndarray = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0])
    
    # PD Gains
    KP_ARM: np.ndarray = np.array([250.0, 450.0, 300.0, 100.0, 100.0, 50.0])
    KD_ARM: np.ndarray = np.array([15.0, 25.0, 15.0, 4.0, 4.0, 1.5])
    KP_GRIPPER: np.ndarray = np.array([500.0, 500.0])
    KD_GRIPPER: np.ndarray = np.array([10.0, 10.0])

    def __init__(self, model_path: str):
        """Initializes the robot model, data, and targets.

        Args:
            model_path: Path to the MuJoCo scene XML file.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.TIMESTEP

        # Control targets
        self.q_target_arm = np.copy(self.HOME_POSE)
        self.gripper_width_target = 0.0

        # Initialize robot physical pose
        self.data.qpos[:6] = self.HOME_POSE
        mujoco.mj_forward(self.model, self.data)

    def _compute_arm_torques(self) -> np.ndarray:
        """Computes PD torques for the arm with drag detection.

        Returns:
            A 1D array of 6 torque values for the arm joints.
        """
        arm_error = self.q_target_arm - self.data.qpos[:6]
        arm_v_error = 0 - self.data.qvel[:6]
        effort = (self.KP_ARM * arm_error) + (self.KD_ARM * arm_v_error)

        # Torque Deviation Drag Detection
        if np.linalg.norm(effort) > self.DRAG_THRESHOLD:
            self.q_target_arm = np.copy(self.data.qpos[:6])
            effort = np.zeros(6)

        return effort

    def _compute_gripper_torques(self) -> np.ndarray:
        """Computes symmetric PD torques for the gripper.

        Returns:
            A 1D array of 2 torque values for the gripper fingers.
        """
        q_target = np.array([self.gripper_width_target, -self.gripper_width_target])
        error = q_target - self.data.qpos[6:8]
        v_error = 0 - self.data.qvel[6:8]
        effort = (self.KP_GRIPPER * error) + (self.KD_GRIPPER * v_error)

        return effort

    def step(self) -> None:
        """Executes one control step and physics update."""
        arm_effort = self._compute_arm_torques()
        gripper_effort = self._compute_gripper_torques()

        # Combine PD effort and add Gravity/Coriolis compensation
        total_pd = np.concatenate([arm_effort, gripper_effort])
        self.data.ctrl[:8] = total_pd + self.data.qfrc_bias[:8]

        mujoco.mj_step(self.model, self.data)


def main():
    """Main entry point for the simulation."""
    model_path = '../assets/robots/piper/scene.xml'
    robot = PiperRobot(model_path)

    print(f"Robot initialized at Home: {robot.HOME_POSE}")
    print("Drag the arm with the mouse to move. Release to hold.")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            robot.step()
            viewer.sync()

            # Real-time synchronization
            elapsed = time.time() - step_start
            if elapsed < robot.TIMESTEP:
                time.sleep(robot.TIMESTEP - elapsed)


if __name__ == '__main__':
    main()