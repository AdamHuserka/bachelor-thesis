import logging
import os
import pybullet as p
import numpy as np
import cv2
import math
import igibson
import sys

from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
    get_joint_state,
    get_joint_info,
    get_pose,
    get_link_pose,
    get_link_name,
)
from igibson.external.pybullet_tools.transformations import rotation_matrix
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.viewer import Viewer
from igibson.robots import REGISTERED_ROBOTS
from igibson.robots.nico import Nico
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.utils.utils import l2_distance, parse_config, restoreState


from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path


last_joint_poses = []
def main(selection="user", headless=False, short_exec=False):
    """
    Loads robot NICO in an empty scene, tests IK methods
    """

    # Create simulator, scene and robot (nico)
    config = parse_config(os.path.join(igibson.configs_path, "robots", "nico.yaml"))
    s = Simulator(mode="headless", gravity=0, use_pb_gui=True if not headless else False)
    scene = EmptyScene()
    s.import_scene(scene)

    robot_config = config["robot"]
    robot_config.pop("name")

    nico = Nico(**robot_config)
    s.import_object(nico)

    #print("Imported the robot into the simulator")
    body_ids = nico.get_body_ids()
    assert len(body_ids) == 1, "nico robot is expected to be single-body."
    robot_id = body_ids[0]

    right_arm_default_joint_positions = (
            0,   #Torso - Shoulder
                    #Minimum: -0.4363, Maximum: 1.3963, Average: 0.48
            0,   #Shoulder - Collarbone
                    #Minumim: -0.5236, Maximum: 3.1416, Average: 1.309
            0,   #Collarbone - Upper arm
                    #Minumim: 0, Maximum: 1.2217, Average: 0.61085
            0.8726,   #Upper arm - Lower arm
                    #Minumim: 0.8726, Maximum: 3.1416, Average: 2.0071
                0    #Lower arm - Wrist
                    #Minumim: -1.571, Maximum: 1.571, Average: 0
    )
    left_arm_default_joint_positions = (
            -0.2,   #Torso - Shoulder
                    #Minimum: -0.4363, Maximum: 1.3963, Average: 0.48
            0,   #Shoulder - Collarbone
                    #Minumim: -0.5236, Maximum: 3.1416, Average: 1.309
            0,   #Collarbone - Upper arm
                    #Minumim: 0, Maximum: 1.2217, Average: 0.61085
            0.8726,   #Upper arm - Lower arm
                    #Minumim: 0.8726, Maximum: 3.1416, Average: 2.0071
                0    #Lower arm - Wrist
                    #Minumim: -1.571, Maximum: 1.571, Average: 0
    )

    robot_default_joint_positions = (
        list(left_arm_default_joint_positions)
        + [0, -1.285]
        + list(right_arm_default_joint_positions)
        + [0, -1.285]
    )

    robot_joint_names = [joint for joint in nico.joints]
    right_arm_joints_names = robot_joint_names[:5]
    right_eef_joints_names = robot_joint_names[5:7]
    left_arm_joints_names = robot_joint_names[7:12]
    left_eef_joints_names = robot_joint_names[12:14]
    # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
    robot_left_arm_indices = nico.arm_control_idx[nico.left_arm]
    robot_right_arm_indices = nico.arm_control_idx[nico.right_arm]

    # PyBullet ids of the joints corresponding to the joints of the arm
    right_arm_joint_ids = joints_from_names(robot_id, right_arm_joints_names)
    right_eef_joint_ids = joints_from_names(robot_id, right_eef_joints_names)
    all_joint_ids = joints_from_names(robot_id, robot_joint_names)
    right_arm_and_eef_joint_ids = right_arm_joint_ids + right_eef_joint_ids

    set_joint_positions(robot_id, right_arm_joint_ids, right_arm_default_joint_positions)

    # Set robot base
    nico.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    nico.keep_still()

    # Get initial EE position
    x, y, z = [0.25, 0, 0.25]

    # Define the limits (max and min, range), and resting position for the joints
    max_limits = get_max_limits(robot_id, all_joint_ids)
    min_limits = get_min_limits(robot_id, all_joint_ids)
    rest_position = robot_default_joint_positions
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    joint_damping = [0.1 for _ in joint_range]

    def calculate_forward_kinematics(robot_id, start_link_id, joint_ids, joint_poses, visuals=False):
        print("FD CALLED", "*"*80)
        # Get joint states
        joints_info = {joint: get_joint_info(robot_id, joint) for joint in joint_ids}
        # joints_state = {joint: get_joint_state(robot_id, joint) for joint in joint_ids}
        joints_angles = {joint: joint_poses[i] for i, joint in enumerate(joint_ids)}
        # Resulting homogenious transformation
        #H = identity_matrix()
        #H[:3, 3] = get_link_pose(robot_id, start_link_id)[0]

        #translation = translation_from_matrix(H)
        #computedEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[0, 0, 1, 0.5])
        #s.import_object(computedEEFmarker)
        #computedEEFmarker.set_position(translation) #BLUE MARKER OF THE COMPUTED POSITION OF THE TRANSLATION

        joint_translation = np.array(get_link_pose(robot_id, start_link_id)[0])
        real_joint_translation = np.array(get_link_pose(robot_id, start_link_id)[0])

        dx = 0
        dy = 0
        dz = 0
        counter = 0
        # print("Translation at the beginning:", joint_translation)
        # print("Real Translation at the beginning:", real_joint_translation)
        joints_angles[3] = get_joint_state(robot_id, 3).jointPosition + 0.5
        for joint in joint_ids:
            counter += 1/len(joint_ids)
            # print("Joint:", joints_info[joint].jointName, "and his ID:", joint)
            joint_angle = joints_angles[joint]
            joint_axis = joints_info[joint].jointAxis
            # print("IS?", is_x, is_y, is_z)
            if joint != joint_ids[-1]:
                vector = np.array(get_link_pose(robot_id, joint + 1)[0]) - np.array(get_link_pose(robot_id, joint)[0])
                M = rotation_matrix(joint_angle, joint_axis)
                # print(M)
                R = M[:3, :3]
                # print(R)
                dx, dy, dz = np.dot(R, vector)
            real_joint_translation += vector
            joint_translation += np.array([dx, dy, dz])
            if visuals:
                computedEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[0, 1, 1, 0.5])
                s.import_object(computedEEFmarker)
                computedEEFmarker.set_position(joint_translation) #BLUE MARKER OF THE COMPUTED POSITION OF THE TRANSLATION
            # print("Computed Translation:", joint_translation)
            # print("It should be:", real_joint_translation)
            ## print("Rotation:", rotation)
        print("FD CALCULATED", "*"*80)
        rotation = (0, 0, 0)
        return joint_translation, rotation
    

    def accurate_calculate_inverse_kinematics(robot_id, eef_link_id, target_pos, threshold, max_iter):
        global last_joint_poses
        print("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 1
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            print("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(robot_id, right_arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(robot_id, right_arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                )
                joint_poses = np.array(joint_poses)[robot_right_arm_indices]

                set_joint_positions(robot_id, right_arm_joint_ids, joint_poses)

                dist = l2_distance(nico.get_eef_position(), target_pos)
                #calculate_forward_kinematics(robot_id, start_link_id=right_arm_joint_ids[0], joint_ids=right_arm_joint_ids, joint_poses=joint_poses),
                if dist < threshold:

                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                print("Solution found at iter: " + str(it) + ", residual: " + str(dist))
                last_joint_poses.append(joint_poses.copy())
                print("LAST SAVED JOINT POSES:", last_joint_poses)
                break
            else:
                print("Attempt failed. Retry")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses

    nico.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
    s.import_object(marker)
    marker.set_position([0.25, 0, 0.25])

    threshold = 0.03
    max_iter = 1
    print_message()
    quit_now = False
    while True:
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            # if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
            #     y -= 0.01
            # if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
            #     y += 0.01
            # if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
            #     x += 0.01
            # if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
            #     x -= 0.01
            # if k == ord("z") and (v & p.KEY_IS_DOWN):
            #     z += 0.01
            # if k == ord("x") and (v & p.KEY_IS_DOWN):
            #     z -= 0.01
            if k == ord(" "):
                print("Querying joint configuration to current marker position")
                joint_pos = accurate_calculate_inverse_kinematics(
                    robot_id, nico.eef_links[nico.default_arm].link_id, [x, y, z], threshold, max_iter
                )
                if last_joint_poses:
                    print("WE ARE DOING IT!")
                    calculate_forward_kinematics(robot_id, start_link_id=right_arm_joint_ids[0], joint_ids=right_arm_joint_ids, joint_poses=last_joint_poses[-1], visuals=True)
                if joint_pos is not None and len(joint_pos) > 0:
                    print("Solution found. Setting new arm configuration.")
                    set_joint_positions(robot_id, right_arm_joint_ids, joint_pos)
                    print_message()
                    print("Position of the target:", (x, y, z))
                    print("Joint positions that reach it:", [math.degrees(joint) for joint in joint_pos])
                else:
                    print(
                        "No configuration to reach that point. Move the marker to a different configuration and try again."
                    )
            if k == ord("q"):
                print("Quit.")
                quit_now = True
                break

        if quit_now:
            break

        marker.set_position([x, y, z])
        nico.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
        nico.keep_still()
        s.step()

    s.disconnect()

def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press SPACE")
    print("Up/Down arrows: move marker further away or closer to the robot")
    print("Left/Right arrows: move marker to the left or the right of the robot")
    print("z/x: move marker up and down")
    print("q: quit")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
