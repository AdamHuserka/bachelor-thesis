import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key

#THUMB_2_POS = np.array([0, -0.02, -0.05])
#THUMB_1_POS = np.array([0, -0.015, -0.02])
PALM_CENTER_POS = np.array([ 0.035,   0.00,   0.00])
PALM_BASE_POS = np.array([ 0.025,   0.00,   0.00])
FINGER_TIP_POS = np.array([0.05, 0, 0])

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

        # We use multi finger gripper and IK controllers as default
        controllers["camera"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "JointController"
        # print("*"*100)
        # print("Default controllers", controllers)
        # print("*"*100)
        # try:
        #     print("Printing names and data of controllers")
        #     for name, controller in controllers.items():
        #         print(name,":", controller)#, "controller dim", controller.command_dim)
        # except Exception as e:
        #     print("Nah, I'd continue without printing")
        #     print("EXCEPTION:", e)

        return controllers
    
    
    @property
    def _default_camera_joint_controller_config(self):
        """
        :return: Dict[str, Any] Default camera joint controller config to control this robot's camera
        """
        # Always run super method first
        cfg = super()._default_camera_joint_controller_config

        cfg["motor_type"] = "position"
        cfg["compute_delta_in_quat_space"] = [(3, 4, 5)]
        cfg["command_output_limits"] = None

        return cfg
    
    @property
    def _default_gripper_joint_controller_configs(self):
        # The use case for the joint controller for the BehaviorRobot is supporting the VR action space. We configure
        # this accordingly.
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
                    "inverted": True,
                    "command_input_limits": (0, 1),
                    "command_output_limits": output_ranges,
                    "use_delta_commands": True,
                }
            )
        return dic
    
    @property
    def _default_controller_config(self):
        # controllers = {
        #     "camera": {"JointController": self._default_camera_controller_configs},
        # }
        # controllers.update(
        #     {"arm_%s" % arm: {"JointController": self._default_arm_controller_configs[arm]} for arm in self.arm_names}
        # )
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


    @property
    def default_joint_pos(self):
        pos = np.zeros(20)
        # CAMERA (head_z and head_y joints in that order)
        pos[self.camera_control_idx] = np.array([0.0, 0.0])
        # RIGHT ARM
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
        # LEFT ARM
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
                # GraspingPoint(
                #     link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_1_POS * side_coefficients[arm]
                # ),
                # GraspingPoint(
                #     link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_2_POS * side_coefficients[arm]
                # ),
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
