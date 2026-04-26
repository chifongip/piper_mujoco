"""A module for Impedance Control and lead-through programming using Pinocchio."""

import time
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class PiperRobot:
    """Manages the Piper robot simulation with Joint-Space Impedance Control.

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

    # Impedance Gains (Virtual Stiffness and Damping)
    # These act as the 'K' and 'D' in the virtual mass-spring-damper system.
    KP_ARM: np.ndarray = np.array([250.0, 450.0, 300.0, 100.0, 100.0, 50.0])
    KD_ARM: np.ndarray = np.array([15.0, 25.0, 15.0, 4.0, 4.0, 1.5])
    KP_GRIPPER: np.ndarray = np.array([500.0, 500.0])
    KD_GRIPPER: np.ndarray = np.array([10.0, 10.0])

    def __init__(self, mjcf_path: str, robot_mjcf_path: str):
        """Initializes MuJoCo and Pinocchio models."""
        # MuJoCo Setup
        self.model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data_mj = mujoco.MjData(self.model_mj)
        self.model_mj.opt.timestep = self.TIMESTEP

        # Pinocchio Setup
        self.model_pin = pin.buildModelFromMJCF(robot_mjcf_path)
        self.data_pin = self.model_pin.createData()
        self.model_pin.gravity.linear = self.model_mj.opt.gravity

        # Control targets
        self.q_target_arm = np.copy(self.HOME_POSE)
        self.gripper_width_target = 0.0

        # Initialize physical pose
        self.data_mj.qpos[:6] = self.HOME_POSE
        mujoco.mj_forward(self.model_mj, self.data_mj)

    def _compute_impedance_torques(self) -> np.ndarray:
        """Computes Joint-Space Impedance Control torques.

        The control law is: tau = M(q) * (Kp*e + Kd*edot) + b(q, v)
        """
        nq = self.model_pin.nq
        nv = self.model_pin.nv

        # Use full vectors for Pinocchio algorithms
        q_full = self.data_mj.qpos[:nq]
        v_full = self.data_mj.qvel[:nv]

        # 1. Update Mass Matrix M(q) using full q
        pin.crba(self.model_pin, self.data_pin, q_full)

        # 2. Compute Bias forces b(q, v) using full q and v
        # Passing zero acceleration for the full model size
        a_zero_full = np.zeros(nv)
        bias_full = pin.rnea(self.model_pin, self.data_pin, q_full, v_full, a_zero_full)

        # 3. Define desired virtual dynamics (acceleration) for the ARM ONLY
        # We slice back to 6 for our specific arm control logic
        q_arm = q_full[:6]
        v_arm = v_full[:6]

        q_error = self.q_target_arm - q_arm
        v_error = 0 - v_arm

        a_des_arm = (self.KP_ARM * q_error) + (self.KD_ARM * v_error)

        # 4. Lead-through Logic (Drag Detection)
        if np.linalg.norm(a_des_arm) > self.DRAG_THRESHOLD:
            self.q_target_arm = np.copy(q_arm)
            a_des_arm = self.KD_ARM * v_error

        # 5. Final Impedance Law
        # Multiply the 6x6 arm submask of the Mass Matrix by our 6-dim a_des
        # M[:6, :6] @ a_des_arm gives the required torques for the arm joints
        tau_arm = (self.data_pin.M[:6, :6] @ a_des_arm) + bias_full[:6]

        return tau_arm

    def _compute_gripper_torques(self) -> np.ndarray:
        """Standard PD for the gripper fingers."""
        q_target = np.array([self.gripper_width_target, -self.gripper_width_target])
        error = q_target - self.data_mj.qpos[6:8]
        v_error = 0 - self.data_mj.qvel[6:8]
        return (self.KP_GRIPPER * error) + (self.KD_GRIPPER * v_error)

    def step(self) -> None:
        """Executes one simulation step with Impedance Control."""
        # Calculate specialized arm torques and standard gripper torques
        arm_tau = self._compute_impedance_torques()
        gripper_tau = self._compute_gripper_torques()

        # Apply to control buffer
        self.data_mj.ctrl[:6] = arm_tau
        self.data_mj.ctrl[6:8] = gripper_tau

        mujoco.mj_step(self.model_mj, self.data_mj)


def main():
    """Main simulation loop."""
    mjcf_path = "../assets/robots/piper/scene.xml"
    robot_mjcf_path = "../assets/robots/piper/piper.xml"

    robot = PiperRobot(mjcf_path, robot_mjcf_path)

    print("Impedance Control Active.")
    print("The robot should feel consistent and compliant across all joints.")

    with mujoco.viewer.launch_passive(robot.model_mj, robot.data_mj) as viewer:
        while viewer.is_running():
            step_start = time.time()
            robot.step()
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < robot.TIMESTEP:
                time.sleep(robot.TIMESTEP - elapsed)


if __name__ == "__main__":
    main()
