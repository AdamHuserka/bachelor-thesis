import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key


class Nico(ManipulationRobot):
    """
    Nico Robot
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        :param **kwargs: see ManipulationRobot
        """
        # Run super init
        super().__init__(**kwargs)

        self.right_arm = self.arm_names[0]
        self.left_arm = self.arm_names[1]
        self.arm_controller = "InverseKinematicsController"
        self.gripper_controller = "NullGripperController"

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

        controllers = []
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        #controllers["base"] = "DifferentialDriveController"
        #controllers["camera"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = self.arm_controller
            controllers["gripper_{}".format(arm)] = self.gripper_controller
        return controllers


    @property
    def default_joint_pos(self):
        pos = np.zeros(16)
        #pos[self.camera_control_idx] = np.array([0.0, 0.45])
        print("Default joint positions")
        print("self.right_arm:", self.right_arm)
        print("Indexes for the position array given by gripper_control_idx[self.right_arm]", self.gripper_control_idx[self.right_arm])
        pos[self.gripper_control_idx[self.right_arm]] = np.array(
                [
                    0.00,   #r_wrist_x_rjoint
                            #Minimum: -0.78, Maximum: 0.78, Average: 0          
                   -1.285    #gripper_rjoint
                            #Minimum: -2.57, Maximum: 0, Average: -1.285
                ]
            )
        print("Indexes for the position array given by arm_control_idx[self.right_arm]", self.arm_control_idx[self.right_arm])
        pos[self.arm_control_idx[self.right_arm]] = np.array(
                [   0,   #Torso - Shoulder
                            #Minimum: -0.4363, Maximum: 1.3963, Average: 0.48
                    0,   #Shoulder - Collarbone
                            #Minumim: -0.5236, Maximum: 3.1416, Average: 1.309
                    0,   #Collarbone - Upper arm
                            #Minumim: 0, Maximum: 1.2217, Average: 0.61085
                  0.8726,   #Upper arm - Lower arm
                            #Minumim: 0.8726, Maximum: 3.1416, Average: 2.0071
                       0    #Lower arm - Wrist
                            #Minumim: -1.571, Maximum: 1.571, Average: 0
                ]
                 
            )
        pos[self.gripper_control_idx[self.left_arm]] = np.array(
                [
                    0.00,   #l_wrist_x_rjoint
                            #Minimum: -0.78, Maximum: 0.78, Average: 0          
                   -1.285    #grippel_rjoint
                            #Minimum: -2.57, Maximum: 0, Average: -1.285
                ]
            )
        pos[self.arm_control_idx[self.left_arm]] = np.array(
                [   -0.2,   #Torso - Shoulder
                            #Minimum: -0.4363, Maximum: 1.3963, Average: 0.48
                    0,   #Shoulder - Collarbone
                            #Minumim: -0.5236, Maximum: 3.1416, Average: 1.309
                    0,   #Collarbone - Upper arm
                            #Minumim: 0, Maximum: 1.2217, Average: 0.61085
                  0.8726,   #Upper arm - Lower arm
                            #Minumim: 0.8726, Maximum: 3.1416, Average: 2.0071
                       -1.571    #Lower arm - Wrist
                            #Minumim: -1.571, Maximum: 1.571, Average: 0
                ]
                 
            )
        return pos

    @property
    def gripper_link_to_grasp_point(self):
        return {self.right_arm: np.array([0.1, 0, 0]), self.left_arm: np.array([-0.1, 0, 0])}

    # @property
    # def assisted_grasp_start_points(self):
    #     return {
    #         self.default_arm: [
    #             GraspingPoint(link_name="r_gripper_finger_link", position=[0.04, -0.012, 0.014]),
    #             GraspingPoint(link_name="r_gripper_finger_link", position=[0.04, -0.012, -0.014]),
    #             GraspingPoint(link_name="r_gripper_finger_link", position=[-0.04, -0.012, 0.014]),
    #             GraspingPoint(link_name="r_gripper_finger_link", position=[-0.04, -0.012, -0.014]),
    #         ]
    #     }

    # @property
    # def assisted_grasp_end_points(self):
    #     return {
    #         self.default_arm: [
    #             GraspingPoint(link_name="l_gripper_finger_link", position=[0.04, 0.012, 0.014]),
    #             GraspingPoint(link_name="l_gripper_finger_link", position=[0.04, 0.012, -0.014]),
    #             GraspingPoint(link_name="l_gripper_finger_link", position=[-0.04, 0.012, 0.014]),
    #             GraspingPoint(link_name="l_gripper_finger_link", position=[-0.04, 0.012, -0.014]),
    #         ]
    #     }

    # @property
    # def camera_control_idx(self):
    #     """
    #     :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
    #     """
    #     return np.array([0, 1])
    
    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.right_arm: np.array([2, 3, 4, 5, 6]), self.left_arm: np.array([9, 10, 11, 12, 13])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {self.right_arm: np.array([7, 8]), self.left_arm: np.array([14, 15])}

    @property
    def disabled_collision_pairs(self):
        return []
    
        # [
        #     ["torso_lift_link", "shoulder_lift_link"],
        #     ["torso_lift_link", "torso_fixed_link"],
        #     ["caster_wheel_link", "estop_link"],
        #     ["caster_wheel_link", "laser_link"],
        #     ["caster_wheel_link", "torso_fixed_link"],
        #     ["caster_wheel_link", "l_wheel_link"],
        #     ["caster_wheel_link", "r_wheel_link"],
        # ]
    
    @property
    def arm_names(self):
        return ["right", "left"]

    @property
    def eef_link_names(self):
        return {self.right_arm: "gripper", self.left_arm: "grippel"}

    @property
    def finger_link_names(self):
        return {self.right_arm: ["endeffector"], self.left_arm: ["endeffectol"]}

    @property
    def finger_joint_names(self):
        return {self.right_arm: ["gripper_rjoint"], self.left_arm: ["grippel_rjoint"]} #in nico_upper_rh6d.urdf the endeffector_joint joint is fixed

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/nico/nico_upper_head_rh6d_dual.urdf")
