import logging
import os
import pybullet as p
import numpy as np
import cv2
import tkinter

import igibson
import sys

from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
    get_joint_position,
    get_joint_state,
    get_joint_info,
    get_pose,
    get_link_pose,
    get_link_name,
)
from igibson.external.pybullet_tools.transformations import *
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.viewer import Viewer
from igibson.robots import REGISTERED_ROBOTS
from igibson.robots.nico import Nico
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.utils.utils import *


from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path


#problem_marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[1,0,0,0.5])
def main(selection="user", headless=False, short_exec=False):
    """
    Loads robot NICO in an empty scene, generate random actions
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
    print("Nico's _joints:")
    for k in nico._joints:
        print("Key:", k)
    print("...")
    print("Nico's base link:", nico.base_name)
    #print("Imported the robot into the simulator")
    body_ids = nico.get_body_ids()
    assert len(body_ids) == 1, "nico robot is expected to be single-body."
    robot_id = body_ids[0]
    print("Nico's 3rd link:", get_link_name(robot_id, 3))
    print("Nico's shoulder link world position:", get_link_pose(robot_id, 3))
    nico.set_position_orientation([0, 0, 0], [0, 0, 0, 1])

    robot_joint_names = [joint for joint in nico.joints]
    #all_joint_ids = joints_from_names(robot_id, robot_joint_names)
    print("Joint name: state")
    for k, v in nico.joints.items():
        print(str(k) + ":", v.get_state()[0])

    robot_joint_names = [joint for joint in nico.joints]
    right_arm_joints_names = robot_joint_names[2:7]
    right_eef_joints_names = robot_joint_names[7:9]
    left_arm_joints_names = robot_joint_names[9:14]
    left_eef_joints_names = robot_joint_names[14:16]
    # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
    robot_left_arm_indices = nico.arm_control_idx[nico.left_arm]
    robot_right_arm_indices = nico.arm_control_idx[nico.right_arm]

    # PyBullet ids of the joints corresponding to the joints of the arm
    right_arm_joint_ids = joints_from_names(robot_id, right_arm_joints_names)
    right_eef_joint_ids = joints_from_names(robot_id, right_eef_joints_names)
    all_joint_ids = joints_from_names(robot_id, robot_joint_names)
    print("All joint's names", robot_joint_names)
    print()
    print("Nico's all joints ids:", all_joint_ids)
    print()
    right_arm_and_eef_joint_ids = right_arm_joint_ids + right_eef_joint_ids
    right_arm_link_lenghts = [l2_distance(get_link_pose(robot_id, joint)[0], get_link_pose(robot_id, joint - 1)[0]) 
                              if joint != right_arm_and_eef_joint_ids[0] 
                              else l2_distance(get_link_pose(robot_id, joint)[0], get_link_pose(robot_id, -1)[0])
                              for joint in right_arm_and_eef_joint_ids]
    print(right_arm_link_lenghts)
    print("Nico's right arm and EEF joint ids:", right_arm_and_eef_joint_ids)

    marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
    s.import_object(marker)
    marker.set_position([0.25, 0, 0.25])
    markers = [VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02) for _ in range(len(right_arm_joint_ids))]
    markers_whatifs = [VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[1,1,1,0.5]) for _ in range(len(right_arm_joint_ids))]
    for m in markers:
        s.import_object(m)
    for m in markers_whatifs[:3]:
        s.import_object(m)
    def calculate_right_arm_forward_kinematics(robot_id, start_link_id, eef_link_id, joint_ids, joint_poses, starting_link_positions=None):
        print("FD CALLED", "*"*80)
        # Get joint info
        joints_info = {joint: get_joint_info(robot_id, joint) for joint in joint_ids}
        # joints_state = {joint: get_joint_state(robot_id, joint) for joint in joint_ids}
        joints_angles = {joint: joint_poses[i] for i, joint in enumerate(joint_ids)}
        # Resulting homogenious transformation
        H = identity_matrix()
        t = np.array(get_link_pose(robot_id, start_link_id)[0])
        if starting_link_positions is not None:
            starting_link_positions[start_link_id] = t[:]
        H[:3, 3] = t[:]
        color = [0, 1, 1, 0.5]


        origin = np.array(get_link_pose(robot_id, start_link_id)[0])
        m = markers_whatifs[0]
        m.rgba_color = [0,0,0,0.5]
        m.set_position(origin)
        vec = np.array([0.1,0.1,0.1])
        vec += origin
        m = markers_whatifs[1]
        m.rgba_color = [0,1,0,0.5]
        m.set_position(vec)

        for joint in joint_ids:
            print("-"*80)
            print("Joint:", joints_info[joint].jointName, "and his ID:", joint)
            print("LINK POSITION:", np.array(get_link_pose(robot_id, joint)[0]))
            joint_angle = joints_angles[joint]
            print("Angle:", np.degrees(joint_angle))
            #if joint in (joint_ids[-2],): #right-hand rule not working???
            #    joint_angle -= np.sqrt(3)/2#0.87076240919479543#np.radians(45)
            print("Angle repaired:", np.degrees(joint_angle))
            joint_axis = joints_info[joint].jointAxis   
            #if joint == joint_ids[-2]:
            #    joint_axis = (0.0, -1.0, 0.0)
            if abs(joint_axis[0]) > 0:
                color = [1,0,0,0.5]
            if abs(joint_axis[1]) > 0:
                color = [0,1,0,0.5]
            if abs(joint_axis[2]) > 0:
                color = [0,0,1,0.5]
            print("Angle", joint_angle)
            print("Axis", joint_axis)
            #if joint != joint_ids[-1]:
            print("Link poses of:", joint,"and", joint - 1)
            if starting_link_positions is not None:
                link2 = starting_link_positions[joint]
                link1 = starting_link_positions[joint - 1]
            else:
                link2 = get_link_pose(robot_id, joint)[0]
                link1 = get_link_pose(robot_id, joint - 1)[0]
            print("Link joint:", link2)
            print("Link joint - 1:", link1)
            vector = np.array(link2) - np.array(link1)
            print("Vector:", vector)
            print("H\n",H)
            print()
            #t = H[:3, 3]
            #if joint != joint_ids[0]:
            M = rotation_matrix(joint_angle, joint_axis)
            print("Get rot matrix angle:", rotation_from_matrix(M)[0])
            print("Get rot matrix axis:", rotation_from_matrix(M)[1])
            M[:3, 3] = vector[:]
            H = np.dot(H, M)
            t = H[:3, 3]

            print("H after H°R\n",H)
            print("EXPERIMENT TRANSLATION:", t)
            computedEEFmarker = markers[joint - joint_ids[0]]
            markers[joint - joint_ids[0]].rgba_color = color
            computedEEFmarker.set_position(t) #MARKER OF THE COMPUTED POSITION OF THE TRANSLATION
            color = [0, 1, 1, 1.0]
        
        print("FD CALCULATED", "*"*80)
        rotation = (0, 0, 0)
        return t, rotation

    quit_now = False
    counter = 0
    flag = True
    starting_link_poses = {joint: get_link_pose(robot_id, joint)[0] for joint in right_arm_joint_ids}
    def show(val):
        jp = []
        for joint in right_arm_joint_ids:
            jp.append((scale_vars[joint].get() + scale_configs[joint]["offset"])/precision)
            labels[joint].config(text="VALUE:" + str(jp[-1]))
        result = calculate_right_arm_forward_kinematics(robot_id, 
                                                        2, 
                                                        nico.eef_links[nico.default_arm].link_id, 
                                                        right_arm_joint_ids, 
                                                        jp,
                                                        starting_link_poses
                                                        )
    def plus(joint):
        if scale_configs[joint]["to"] > scale_vars[joint].get():
            scale_vars[joint].set(scale_vars[joint].get() + 1)
        show(None)
    def minus(joint):
        if scale_configs[joint]["from_"] < scale_vars[joint].get():
            scale_vars[joint].set(scale_vars[joint].get() - 1)
        show(None)

    canvas = tkinter.Canvas()
    scale_configs = dict()
    precision = 1000
    for joint in right_arm_joint_ids:
        info = get_joint_info(robot_id, joint)
        lower_limit = int(info.jointLowerLimit * precision)
        upper_limit = int(info.jointUpperLimit * precision)
        print("Ul", upper_limit)
        print("Ll", lower_limit)
        scale_configs[joint] = { "from_": 0, "to": upper_limit - lower_limit, "offset": lower_limit}

    scale_vars = {joint: tkinter.DoubleVar() for joint in right_arm_joint_ids}
    jp = [0.75312394, 0.36425076, 1.02149217, 1.23143874, 0.27490885]
    for i, joint in enumerate(right_arm_joint_ids):
        scale_vars[joint].set(int(jp[i] * precision) - scale_configs[joint]["offset"])
    scales = {joint: tkinter.Scale( canvas, 
                                    variable=scale_vars[joint],
                                    from_=scale_configs[joint]["from_"],
                                    to=scale_configs[joint]["to"],
                                    command=show, 
                                    orient="horizontal"
                                    ) 
                                    for joint in right_arm_joint_ids}
    labels = {joint: tkinter.Label(canvas) for joint in right_arm_joint_ids}
    plus_buttons = {joint: tkinter.Button(canvas, text="+1", command=lambda m=joint : plus(m)) for joint in right_arm_joint_ids}
    minus_buttons = {joint: tkinter.Button(canvas, text="-1", command=lambda m=joint : minus(m)) for joint in right_arm_joint_ids}
    for label in labels.values():
        label.config(text = "Horizontal Scale Value = ")


    canvas.pack()
    for joint in right_arm_joint_ids:
        scales[joint].pack()
        labels[joint].pack()
        minus_buttons[joint].pack()
        plus_buttons[joint].pack()
    #joint 6 from_=0, to=2269, offset = 872

    while True:
        counter += 1
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            if k == ord("q"):
                print("Quit.")
                quit_now = True
                break
        if counter >= 0 and flag:
            print("WE DONE THE KINEMATICS")
            #jp_before = [get_joint_state(robot_id, joint).jointPosition for joint in right_arm_joint_ids]
            jp = [0.75312394, 0.36425076, 1.02149217, 1.23143874, 0.27490885]
            result = calculate_right_arm_forward_kinematics(robot_id, 
                                                    2, 
                                                    nico.eef_links[nico.default_arm].link_id, 
                                                    right_arm_joint_ids,
                                                    joint_poses = 
                                                        jp
                                                    # [get_joint_state(robot_id, joint).jointPosition
                                                    #  - get_joint_state(robot_id, joint).jointPosition
                                                    #  for joint in right_arm_joint_ids]
                                                     )
            # set_joint_positions(robot_id, right_arm_joint_ids, [-0.15678173,  0.17113347,  0.00170381,  0.86940125,  1.33662929])
            # array([0.70110679, 0.26622198, 1.01962282, 1.11010884, 0.3100647 ]), 
            # array([ 0.61957018,  0.27837879,  0.1274856 ,  1.29629587, -1.36055436]), 
            # array([0.75312394, 0.36425076, 1.02149217, 1.23143874, 0.27490885])
            set_joint_positions(robot_id, right_arm_joint_ids, jp)
            for joint in right_arm_joint_ids:
                pose = get_link_pose(robot_id, joint)[0]
                poseBefore = get_link_pose(robot_id, joint - 1)[0]
                angle = get_joint_state(robot_id, joint).jointPosition
                print("Joint", joint, "position:", pose)
                print("Joint", joint - 1, "position:", poseBefore)
                print("Joint", joint, "angle:", angle)
                computedEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[1,1,1,0.5])
                s.import_object(computedEEFmarker)
                computedEEFmarker.set_position(pose) #MARKER OF THE COMPUTED POSITION OF THE TRANSLATION
                computedEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.02, rgba_color=[1,1,0,0.5])
                s.import_object(computedEEFmarker)
                computedEEFmarker.set_position(poseBefore)
            pose = np.array(get_link_pose(robot_id, 7)[0])
            poseBefore = np.array(get_link_pose(robot_id, 6)[0])
            v1 = pose - poseBefore
            prilahla = np.linalg.norm(v1)
            v2 = pose - np.array(result[0])
            protilahla = np.linalg.norm(v2)
            print(v1)
            print(v2)
            print(pose)
            print(prilahla)
            print(protilahla)
            at = np.arctan(protilahla/prilahla)
            print("arctan:", at)
            print("In degrees:", np.degrees(at))
            eef_position = result[0]
            eef_rotation = result[1]
            eef_angle = eef_rotation[0]
            eef_directiom = eef_rotation[1]
            eef_point = eef_rotation[2]
            print("These are the results of FK:\n")
            print("REAL EEF POSITION:", nico.get_eef_position())
            print("REAL EEF ROTATION:", nico.get_eef_orientation())
            print("CALC EEF position:", eef_position)
            print("CALC EEF rotation:", eef_rotation)
            print("CALC EEF angle:", eef_angle)
            print("CALC EEF direction:", eef_directiom)
            print("CALC EEF point:", eef_point)

            # realEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06, rgba_color=[0, 1, 0, 0.5])
            # s.import_object(realEEFmarker)
            # realEEFmarker.set_position(nico.get_eef_position()) #GREEN MARKER OF THE REAL POSITION OF THE EEF

            #computedEEFmarker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06, rgba_color=[1, 0, 0, 0.5])
            #s.import_object(computedEEFmarker)
            #computedEEFmarker.set_position(eef_position) #GREEN MARKER OF THE COMPUTED POSITION OF THE EEF
            #set_joint_positions(robot_id, right_arm_joint_ids, jp_before)
            tkinter.mainloop()
            flag = False

        if quit_now:
            break
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
