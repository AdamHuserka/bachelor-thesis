import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot, ASSIST_ACTIVATION_THRESHOLD, CONSTRAINT_VIOLATION_THRESHOLD
from igibson.external.pybullet_tools.utils import get_constraint_violation

PALM_BASE_POS = np.array([ 0.025,   0.00,   0.00])
PALM_CENTER_POS = np.array([ 0.035,   0.00,   0.00])
FINGER_TIP_POS = np.array([0.05, 0, 0])
#THUMB_TIP_POS = np.array([0.0353, 0.0353, 0.02])

class Nico(ManipulationRobot, ActiveCameraRobot):
    """
    Nico Robot
    """

    def __init__(
        self,
        grasping_mode="sticky",
        **kwargs,
    ):
        """
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        :param **kwargs: see ManipulationRobot
        """
        # Run super init
        super().__init__(grasping_mode=grasping_mode, **kwargs)

        self.right_arm = self.arm_names[0]
        self.left_arm = self.arm_names[1]

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Nico"

    def _create_discrete_action_space(self):
        # Nico does not support discrete actions
        raise ValueError("Nico does not support discrete actions!")

    def load(self, simulator):
        # Run super method
        ids = super().load(simulator)

        assert len(ids) == 1, "Nico robot is expected to have only one body ID."

        # Extend super method by increasing laterial friction for EEF
        #for link in self.finger_joint_ids[self.default_arm]:
        #    p.changeDynamics(self.base_link.body_id, link, lateralFriction=500)

        return ids

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        controllers = ["camera"]
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        controllers["camera"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "JointController"

        return controllers
    
    
    @property
    def _default_camera_joint_controller_config(self):
        """
        :return: Dict[str, Any] Default camera joint controller config to control this robot's camera
        """
        # Always run super method first
        cfg = super()._default_camera_joint_controller_config
        joint_lower_limits, joint_upper_limits = self.control_limits["position"][0][self.camera_control_idx], self.control_limits["position"][1][self.camera_control_idx]
        ranges = joint_upper_limits - joint_lower_limits
        output_ranges = (-ranges, ranges)
        cfg["motor_type"] = "position"
        cfg["command_output_limits"] = output_ranges

        return cfg
    
    @property
    def _default_gripper_joint_controller_configs(self):
        # Always run super method first
        dic = super()._default_gripper_joint_controller_configs

        for arm in self.arm_names:
            # Compute the command output limits that would allow -1 to fully open and 1 to fully close.
            joint_lower_limits = self.control_limits["position"][0][self.gripper_control_idx[arm]]
            joint_upper_limits = self.control_limits["position"][1][self.gripper_control_idx[arm]]
            ranges = joint_upper_limits - joint_lower_limits
            output_ranges = (-ranges, ranges)
            dic[arm].update(
                {
                    "motor_type": "position",
                    "parallel_mode": True,
                    "inverted": True if arm == "right" else False,
                    "command_input_limits": (0, 1),
                    "command_output_limits": output_ranges,
                    "use_delta_commands": True,
                }
            )
        return dic
    
    @property
    def _default_controller_config(self):
        # Always run super method first
        controllers = super()._default_controller_config

        controllers.update(
            {
                "gripper_%s"
                % arm: {
                    #"NullGripperController": self._default_gripper_null_controller_configs[arm],
                    #"MultiFingerGripperController": self._default_gripper_multi_finger_controller_configs[arm],
                    "JointController": self._default_gripper_joint_controller_configs[arm],
                }
                for arm in self.arm_names
            }
        )
        return controllers

    # override
    def _handle_assisted_grasping(self, control, control_type):
        """
        Handles assisted grasping.

        :param control: Array[float], raw control signals that are about to be sent to the robot's joints
        :param control_type: Array[ControlType], control types for each joint
        """
        # Loop over all arms
        for arm in self.arm_names:
            joint_idxes = self.gripper_control_idx[arm]
            current_positions = self.joint_positions[joint_idxes]
            assert np.all(
                control_type[joint_idxes] == ControlType.POSITION
            ), "Assisted grasping only works with position control."
            desired_positions = control[joint_idxes] * (-1 if arm == "left" else 1)
            activation_thresholds = (
                (1 - ASSIST_ACTIVATION_THRESHOLD) * self.joint_lower_limits
                + ASSIST_ACTIVATION_THRESHOLD * self.joint_upper_limits
            )[joint_idxes] * (-1 if arm == "left" else 1)

            # We will clip the current positions just near the limits of the joint. This is necessary because the
            # desired joint positions can never reach the unclipped current positions when the current positions
            # are outside the joint limits, since the desired positions are clipped in the controller.
            if arm == "right":
                clipped_current_positions = np.clip(
                    current_positions,
                    self.joint_lower_limits[joint_idxes] + 1e-3,
                    self.joint_upper_limits[joint_idxes] - 1e-3,
                )
            else:
                clipped_current_positions = np.clip(
                    current_positions * -1,
                    (self.joint_upper_limits[joint_idxes] - 1e-3) * -1,
                    self.joint_lower_limits[joint_idxes] - 1e-3,
                )
            if self._ag_obj_in_hand[arm] is None:
                # We are not currently assisted-grasping an object and are eligible to start.
                # We activate if the desired joint position is above the activation threshold, regardless of whether or
                # not the desired position is achieved e.g. due to collisions with the target object. This allows AG to
                # work with large objects.
                if np.any(desired_positions < activation_thresholds):
                    self._ag_data[arm] = self._calculate_in_hand_object(arm=arm)
                    self._establish_grasp(arm=arm, ag_data=self._ag_data[arm])

            # Otherwise, decide if we will release the object.
            else:
                # If we are not already in the process of releasing, decide if we want to start.
                # We should release an object in two cases: if the constraint is violated, or if the desired hand
                # position in this frame is more open than both the threshold and the previous current hand position.
                # This allows us to keep grasping in cases where the hand was frozen in a too-open position
                # possibly due to the grasped object being large.
                if self._ag_release_counter[arm] is None:
                    constraint_violated = (
                        get_constraint_violation(self._ag_obj_cid[arm]) > CONSTRAINT_VIOLATION_THRESHOLD
                    )
                    thresholds = np.maximum(clipped_current_positions, activation_thresholds)
                    releasing_grasp = np.all(desired_positions > thresholds)
                    if constraint_violated or releasing_grasp:
                        self._release_grasp(arm=arm)

                # Otherwise, if we are already in the release window, continue releasing.
                else:
                    self._handle_release_window(arm=arm)


    @property
    def default_joint_pos(self):
        pos = np.zeros(20)
        # Camera (head_z and head_y joints in that order)
        pos[self.camera_control_idx] = np.array([0.0, 0.0])
        # Right arm
        pos[self.arm_control_idx[self.right_arm]] = np.array(
                [   0,  #r_shoulder_z_rjoint
                            #lower="-0.4363" upper="1.3963"
                    0,  #r_shoulder_y_rjoint
                            #lower="-0.5236" upper="3.1416"
                    0,  #r_upperarm_x_rjoint
                            #lower="0" upper="1.2217"
                    1.5,  #r_elbow_y_rjoint
                            #lower="0.8726" upper="3.1416"
                    0,  #r_wrist_z_rjoint
                            #lower="-1.571" upper="1.571"
                    0,  #r_wrist_x_rjoint
                            #lower="-.78" upper="0.78"
                ]
                 
            )
        # Right gripper
        pos[self.gripper_control_idx[self.right_arm]] = np.array(
                [
                    0,  #gripper_rjoint (index finger)
                            #lower="-2.57" upper="0"        
                    0,  #middlefinger_rjoint
                            #lower="-2.57" upper="0"
                    0,  #litlefinger_rjoint
                            #lower="-2.57" upper="0"
                ]
            )
        # Left arm
        pos[self.arm_control_idx[self.left_arm]] = np.array(
                [   0,  #l_shoulder_z_rjoint
                            #lower="-1.3963" upper="0.4363"
                    0,  #l_shoulder_y_rjoint
                            #lower="-0.5236" upper="3.1416"
                    0,  #l_upperarm_x_rjoint
                            #lower="0" upper="1.2217"
                  1.5,  #l_elbow_y_rjoint
                            #lower="0.8726" upper="3.1416"
                    0,  #l_wrist_z_rjoint
                            #lower="-1.571" upper="1.571"
                    0,  #l_wrist_x_rjoint
                            #lower="-.78" upper="0.78"
                ]
                 
            )
        # Left gripper
        pos[self.gripper_control_idx[self.left_arm]] = np.array(
                [
                    0,  #grippel_rjoint (index finger)
                            #lower="0" upper="2.57"        
                    0,  #middlefingel_rjoint
                            #lower="0" upper="2.57"
                    0,  #litlefingel_rjoint
                            #lower="0" upper="2.57"
                ]
            )
        return pos

    @property
    def gripper_link_to_grasp_point(self):
        return {arm: PALM_CENTER_POS * (1 if arm == "right_hand" else -1) for arm in self.arm_names}

    @property
    def assisted_grasp_start_points(self):
        side_coefficients = {"left_hand": np.array([1, -1, 1]), "right_hand": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_BASE_POS),
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_CENTER_POS * side_coefficients[arm]),
                #GraspingPoint(link_name="%s_palm" % arm, position=(PALM_CENTER_POS + THUMB_TIP_POS) * side_coefficients[arm]),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        side_coefficients = {"left_hand": np.array([1, -1, 1]), "right_hand": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s" % finger, position=FINGER_TIP_POS * side_coefficients[arm])
                for finger in self.finger_link_names[arm]
            ]
            for arm in self.arm_names
        }

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([0, 1])
    
    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.right_arm: np.array([2, 3, 4, 5, 6, 7]), self.left_arm: np.array([11, 12, 13, 14, 15, 16])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {self.right_arm: np.array([8, 9, 10]), self.left_arm: np.array([17, 18, 19])}

    @property
    def disabled_collision_pairs(self):
        return [
            # ["head", "neck"],
            # ["head", "right_shoulder"],
            # ["head", "right_collarbone"],
            # ["head", "right_upper_arm"],
            # ["neck", "right_shoulder"],
            # ["neck", "right_collarbone"],
            # ["neck", "right_upper_arm"],
            # ["gripper", "littlefinger"],
            # ["middlefinger", "gripper"],
            # ["middlefinger", "littlefinger"],
            # ["right_palm", "gripper"],
            # ["right_palm", "middlefinger"],
            # ["right_palm", "littlefinger"],
            # ["right_palm", "right_wrist"],
            # ["right_wrist", "right_lower_arm"],
            # ["right_lower_arm", "right_upper_arm"],
            # ["right_upper_arm", "right_collarbone"],
            # ["right_collarbone", "right_shoulder"],

            # ["head", "left_shoulder"],
            # ["head", "left_collarbone"],
            # ["head", "left_upper_arm"],
            ["neck", "left_shoulder"],
            # ["neck", "left_collarbone"],
            # ["neck", "left_upper_arm"],
            # ["grippel", "littlefingel"],
            # ["middlefingel", "grippel"],
            # ["middlefingel", "littlefingel"],
            # ["left_palm", "grippel"],
            # ["left_palm", "middlefingel"],
            # ["left_palm", "littlefingel"],
            # ["left_palm", "left_wrist"],
            # ["left_wrist", "left_lower_arm"],
            # ["left_lower_arm", "left_upper_arm"],
            # ["left_upper_arm", "left_collarbone"],
            # ["left_collarbone", "left_shoulder"],
            ]
    
    @property
    def arm_names(self):
        return ["right", "left"]

    @property
    def eef_link_names(self):
        return {self.right_arm: "right_palm", self.left_arm: "left_palm"}

    @property
    def finger_link_names(self):
        return {self.right_arm: ["gripper", "middlefinger", "littlefinger"], self.left_arm: ["grippel", "middlefingel", "littlefingel"]}

    @property
    def finger_joint_names(self):
        return {self.right_arm: ["gripper_rjoint", "middlefinger_rjoint", "littlefinger_rjoint"], self.left_arm: ["grippel_rjoint", "middlefingel_rjoint", "littlefingel_rjoint"]}

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/nico/nico_upper_head_rh6d_dual.urdf")