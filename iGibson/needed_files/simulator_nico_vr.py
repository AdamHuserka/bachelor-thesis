import ctypes
import logging
import platform
import time
from time import sleep

import numpy as np
import pybullet as p

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR, VrSettings
from igibson.render.viewer import ViewerVR
from igibson.robots.behavior_robot import BODY_ANGULAR_VELOCITY, BODY_LINEAR_VELOCITY, HAND_BASE_ROTS
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.robots.robot_base import BaseRobot
from igibson.simulator_vr import SimulatorVR
from igibson.utils.vr_utils import VR_CONTROLLERS, VR_DEVICES, VrData, calc_offset, calc_z_rot_from_right

log = logging.getLogger(__name__)

ATTACHMENT_BUTTON_TIME_THRESHOLD = 1  # second


class SimulatorNicoVR(SimulatorVR):
    """
    Simulator class for robot Nico is a wrapper of physics simulatorVR (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
        self,
        gravity=9.8,
        physics_timestep=1 / 120.0,
        render_timestep=1 / 30.0,
        solver_iterations=100,
        mode="vr",
        image_width=128,
        image_height=128,
        vertical_fov=90,
        device_idx=0,
        rendering_settings=MeshRendererSettings(),
        vr_settings=VrSettings(),
        use_pb_gui=False,
    ):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param solver_iterations: number of solver iterations to feed into pybullet, can be reduced to increase speed.
            pybullet default value is 50.
        :param use_variable_step_num: whether to use a fixed (1) or variable physics step number
        :param mode: choose mode from headless, headless_tensor, gui_interactive, gui_non_interactive
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param rendering_settings: settings to use for mesh renderer
        :param vr_settings: settings to use for VR in simulator and MeshRendererVR
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        super().__init__(
            gravity,
            physics_timestep,
            render_timestep,
            solver_iterations,
            mode,
            image_width,
            image_height,
            vertical_fov,
            device_idx,
            rendering_settings,
            vr_settings,
            use_pb_gui,
        )

        self.debug = True
        self.log_writes = 1000
        if self.debug:
            self.log = open("C:\\Users\\huser\\teleop_log.txt", "a")

        self.writing_time = time.time()

        self.robot_origin = {'right': None, 'left': None}
        self.vr_origin = {'right': None, 'left': None}
        self.vr_state = {'right': None, 'left': None}
        self.reset_origin = {'right': False, 'left': False}


    def vr_system_update(self):
        """
        Updates the VR system for a single frame. This includes moving the vr offset,
        adjusting the user's height based on button input, and triggering haptics.
        """
        # Update VR offset using appropriate controller
        if self.vr_settings.touchpad_movement:
            vr_offset_device = "{}_controller".format(self.vr_settings.movement_controller)
            is_valid, _, _ = self.get_data_for_vr_device(vr_offset_device)
            if is_valid:
                _, touch_x, touch_y, _ = self.get_button_data_for_controller(vr_offset_device)
                new_offset = calc_offset(
                    self, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device
                )
                self.set_vr_offset(new_offset)

        # Adjust user height based on y-axis (vertical direction) touchpad input
        vr_height_device = "left_controller" if self.vr_settings.movement_controller == "right" else "right_controller"
        is_height_valid, _, _ = self.get_data_for_vr_device(vr_height_device)
        if is_height_valid:
            curr_offset = self.get_vr_offset()
            hmd_height = self.get_hmd_world_pos()[2]
            _, _, height_y, _ = self.get_button_data_for_controller(vr_height_device)
            if height_y < -0.7:
                vr_z_offset = -0.01
                if hmd_height + curr_offset[2] + vr_z_offset >= self.vr_settings.height_bounds[0]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
            elif height_y > 0.7:
                vr_z_offset = 0.01
                if hmd_height + curr_offset[2] + vr_z_offset <= self.vr_settings.height_bounds[1]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

        # Update haptics for body and hands
        if self.main_vr_robot:
            robot_body_id = self.main_vr_robot.base_link.body_id
            vr_hands = [
                ("left_controller", "left"),
                ("right_controller", "right"),
            ]

            # Check for body haptics
            wall_ids = [bid for x in self.scene.objects_by_category["walls"] for bid in x.get_body_ids()]
            for c_info in p.getContactPoints(robot_body_id):
                if wall_ids and (c_info[1] in wall_ids or c_info[2] in wall_ids):
                    for controller in ["left_controller", "right_controller"]:
                        is_valid, _, _ = self.get_data_for_vr_device(controller)
                        if is_valid:
                            # Use 90% strength for body to warn user of collision with wall
                            self.trigger_haptic_pulse(controller, 0.9)

            # Check for hand haptics
            for hand_device, hand_name in vr_hands:
                is_valid, _, _ = self.get_data_for_vr_device(hand_device)
                if is_valid:
                    if (
                        len(p.getContactPoints(self.main_vr_robot.eef_links[hand_name].body_id))  # TODO: Generalize
                        or self.main_vr_robot.is_grasping(hand_name) == IsGraspingState.TRUE
                    ):
                        # Only use 30% strength for normal collisions, to help add realism to the experience
                        self.trigger_haptic_pulse(hand_device, 0.3)

        self.vr_data_available = True

        

    def gen_vr_robot_action(self):

        """
        Generates an action for the robot Nico to perform based on VrData collected this frame.

        Action space
        Eye:
        - 2DOF pose delta - relative to body frame
        Right hand, left hand (in that order):
        - 6DOF pose delta - relative to body frame (same as above)
        - Trigger fraction delta

        Total size: 16
        """
        # Actions are stored as 1D numpy array
        action = np.zeros((16,))

        if not self.vr_data_available:
            print("Sending empty action")
            return action

        # Get VrData for the current frame
        v = self.gen_vr_data()

        # Update body action space
        hmd_is_valid, hmd_pos, hmd_orn, hmd_r = v.query("hmd")[:4]
        #torso_is_valid, torso_pos, torso_orn = v.query("torso_tracker")
        robot_body = self.main_vr_robot.base_link
        prev_body_pos, prev_body_orn = robot_body.get_position_orientation()
        # inv_prev_body_pos, inv_prev_body_orn = p.invertTransform(prev_body_pos, prev_body_orn)

        # if hmd_is_valid:
        #     des_body_pos, des_body_orn = prev_body_pos, p.getQuaternionFromEuler([0, 0, calc_z_rot_from_right(hmd_r)])
        # else:
        #     des_body_pos, des_body_orn = prev_body_pos, prev_body_orn

        # body_delta_pos, body_delta_orn = p.multiplyTransforms(
        #     inv_prev_body_pos, inv_prev_body_orn, des_body_pos, des_body_orn
        # )
        # body_delta_rot = p.getEulerFromQuaternion(body_delta_orn)

        # # Get new body position so we can calculate correct relative transforms for other VR objects
        # clipped_body_delta_pos = np.clip(body_delta_pos, -BODY_LINEAR_VELOCITY, BODY_LINEAR_VELOCITY).tolist()
        # clipped_body_delta_orn = p.getQuaternionFromEuler(
        #     np.clip(body_delta_rot, -BODY_ANGULAR_VELOCITY, BODY_ANGULAR_VELOCITY).tolist()
        # )
        # new_body_pos, new_body_orn = p.multiplyTransforms(
        #     prev_body_pos, prev_body_orn, clipped_body_delta_pos, clipped_body_delta_orn
        # )
        # # Also calculate its inverse for further local transform calculations
        # inv_new_body_pos, inv_new_body_orn = p.invertTransform(new_body_pos, new_body_orn)
        
        if hmd_is_valid:
            y = hmd_orn[1]
            z = hmd_orn[2]
            controller_name = "camera"
            action[self.main_vr_robot.controller_action_idx[controller_name]] = np.array([z, y])

        #VR parts
        vr_parts = {"right" : self.main_vr_robot.eef_links["right"], "left" : self.main_vr_robot.eef_links["left"]}

        right_controller_prev_world_pos, right_controller_prev_world_orn = right_controller_vr_part.get_position_orientation()
        right_controller_prev_local_pos, right_controller_prev_local_orn = right_controller_vr_part.get_local_position_orientation()
        _, right_controller_inv_prev_local_orn = p.invertTransform(right_controller_prev_local_pos, right_controller_prev_local_orn)

        left_controller_prev_world_pos, left_controller_prev_world_orn = left_controller_vr_part.get_position_orientation()
        left_controller_prev_local_pos, left_controller_prev_local_orn = left_controller_vr_part.get_local_position_orientation()
        _, left_controller_inv_prev_local_orn = p.invertTransform(left_controller_prev_local_pos, left_controller_prev_local_orn)

        for arm in self.main_vr_robot.arm_names:
            controller_name = "right_controller" if arm == "right" else "left_controller"
            if self.get_action_button_state(controller_name, "reset_agent", v):
                # Query the world position and orientation of the @controller_name
                valid, controller_pos, _ = v.query(controller_name)[:3]
                controller_orn = v.query(controller_name)[6]

                # Keep in same world position as last frame if controller/tracker data is not valid
                if not valid:
                    continue
                #Get the robot base link position and rotation
                robot_pos, robot_orn = robot_body.get_position_orientation()
                if self.reset_origin[arm]:
                    self.robot_origin[arm] = {"pos": robot_pos, "quat": robot_orn}
                    self.vr_origin[arm] = {"pos": controller_pos, "quat": controller_orn}
                    self.reset_origin[arm] = False

                # Calculate Positional Action
                robot_pos_offset = robot_pos - self.robot_origin[arm]["pos"]
                target_pos_offset = controller_pos - self.vr_origin[arm]["pos"]
                pos_action = target_pos_offset - robot_pos_offset

                # Calculate Euler Action
                robot_quat_offset = quat_diff(robot_quat, self.robot_origin[arm]["quat"])
                target_quat_offset = quat_diff(controller_orn, self.vr_origin[arm]["quat"])
                quat_action = quat_diff(target_quat_offset, robot_quat_offset)
                euler_action = quat_to_euler(quat_action)

                delta_action = np.concatenate((pos_action, euler_action))

                controller = "arm_right"
                action[self.main_vr_robot.controller_action_idx[controller]] = delta_action

                    #world_pos, world_orn = self.prev_right_controller_pos, self.prev_right_controller_orn
                    #self.prev_right_controller_pos, self.prev_right_controller_orn = world_pos, world_orn
                    
                # _, inv_prev_local_orn = p.invertTransform(self.prev_right_controller_pos, self.prev_right_controller_orn)
                # _, delta_local_orn = p.multiplyTransforms(
                #     [0, 0, 0],
                #     world_orn,
                #     [0, 0, 0],
                #     inv_prev_local_orn,
                # )
                # delta_local_orn = np.array(delta_local_orn)
                # delta_local_orn *= 10
                # delta_local_orn = np.clip(delta_local_orn, -1, 1)
                # print(delta_local_orn)
                # delta_local_orn = p.getEulerFromQuaternion(delta_local_orn.tolist())
                # # Get the delta local position in the reference frame of the body
                # #delta_local_pos = np.array(des_local_pos) - np.array(right_controller_prev_local_pos)
                # delta_local_pos = np.array(world_pos) - np.array(self.prev_right_controller_pos)
                # delta_local_pos *= 10
                if self.debug:
                    print()
                    if self.log_writes > 0:
                        print("WRITING LOG")
                        self.log.writelines("Write time diff:" + str(time.time() - self.writing_time) + "\n")
                        self.writing_time = time.time()
                        self.log.writelines("Delta local pos:" + str(delta_local_pos) + "\n")
                        self.log.writelines("Delta local orn:" + str(delta_local_orn) + "\n")
                        #self.log.writelines("Desired local pos:" + str(des_local_pos) + "\n")
                        #self.log.writelines("right_controller_prev_local_pos:" + str(right_controller_prev_local_pos) + "\n")
                        self.log.writelines("World pos:" + str(world_pos) + "\n")
                        self.log.writelines("World orn:" + str(world_orn) + "\n")
                        self.log.writelines("\n")
                        self.log_writes -= 1
                    else:
                        self.log.close()
                        self.debug = False


            #gripper
            part_name = "right"
            fingers = self.main_vr_robot.gripper_control_idx[part_name]

            # The normalized joint positions are inverted and scaled to the (0, 1) range to match VR controller.
            # Note that we take the minimum (e.g. the most-grasped) finger - this means if the user releases the
            # trigger, *all* of the fingers are guaranteed to move to the released position.
            current_trig_frac = 1 - (np.min(self.main_vr_robot.joint_positions_normalized[fingers]) + 1) / 2

            if valid:
                button_name = "{}_controller_button".format(part_name)
                trig_frac = v.query(button_name)[0]
                delta_trig_frac = trig_frac - current_trig_frac
                print("Deltra grasp:", delta_trig_frac)
            else:
                # Use the last trigger fraction if no valid input was received from controller.
                delta_trig_frac = 0

            grip_controller_name = "gripper_" + part_name

            action[self.main_vr_robot.controller_action_idx[grip_controller_name]] = delta_trig_frac
            print("Whole action", action)
            print("Right grasp controller idx:", self.main_vr_robot.controller_action_idx[grip_controller_name])
        


        if self.get_action_button_state("left_controller", "reset_agent", v):
            controller_name = "left_controller"
            valid, world_pos, _ = v.query(controller_name)[:3]
            world_orn = v.query(controller_name)[6]

            # Keep in same world position as last frame if controller/tracker data is not valid
            if not valid:
                world_pos, world_orn = left_controller_prev_world_pos, left_controller_prev_world_orn

            # Get desired local position and orientation transforms
            des_local_pos, des_local_orn = p.multiplyTransforms(
                inv_new_body_pos, inv_new_body_orn, world_pos, world_orn
            )

            # Get the delta local orientation in the reference frame of the body
            _, delta_local_orn = p.multiplyTransforms(
                [0, 0, 0],
                des_local_orn,
                [0, 0, 0],
                left_controller_inv_prev_local_orn,
            )
            delta_local_orn = p.getEulerFromQuaternion(delta_local_orn)

            # Get the delta local position in the reference frame of the body
            delta_local_pos = np.array(des_local_pos) - np.array(left_controller_prev_local_pos)
            
            if self.debug:
                print()
                if self.log_writes > 0:
                    print("WRITING LOG")
                    self.log.writelines("Write time diff:" + str(time.time() - self.writing_time))
                    self.writing_time = time.time()
                    self.log.writelines("Delta local pos:" + str(delta_local_pos) + "\n")
                    self.log.writelines("Delta local orn:" + str(delta_local_orn) + "\n")
                    self.log.writelines("World pos:" + str(world_pos) + "\n")
                    self.log.writelines("World pos:" + str(world_orn) + "\n")
                    self.log.writelines("\n")
                    self.log_writes -= 1
                else:
                    self.log.close()
                    self.debug = False

            controller_name = "arm_left"
            action[self.main_vr_robot.controller_action_idx[controller_name]] = np.concatenate(
                [delta_local_pos, delta_local_orn[:3]]
            )

            #gripper
            part_name = "left"
            fingers = self.main_vr_robot.gripper_control_idx[part_name]

            # The normalized joint positions are inverted and scaled to the (0, 1) range to match VR controller.
            # Note that we take the minimum (e.g. the most-grasped) finger - this means if the user releases the
            # trigger, *all* of the fingers are guaranteed to move to the released position.
            current_trig_frac = 1 - (np.min(self.main_vr_robot.joint_positions_normalized[fingers]) + 1) / 2

            if valid:
                button_name = "{}_controller_button".format(part_name)
                trig_frac = v.query(button_name)[0]
                delta_trig_frac = trig_frac - current_trig_frac
            else:
                # Use the last trigger fraction if no valid input was received from controller.
                delta_trig_frac = 0

            grip_controller_name = "gripper_" + part_name
            action[self.main_vr_robot.controller_action_idx[grip_controller_name]] = delta_trig_frac
            
        print(action)
        return action