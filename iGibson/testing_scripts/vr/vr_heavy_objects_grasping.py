""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""
import logging
import os

import pybullet as p
import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import BehaviorRobot, Nico
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

# HDR files for PBR rendering
from igibson.simulator_nico_vr import SimulatorNicoVR
from igibson.utils.utils import parse_config

hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main(selection="user", headless=False, short_exec=False):
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = SimulatorNicoVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))

    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    

    objects = [
        ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),

        # ("jenga/jenga.urdf", (0.660000,  0.050000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        # ("jenga/jenga.urdf", (0.650000,  -0.050000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        # ("duck_vhacd.urdf", (0.650000, -0.150000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
        # ("sphere_small.urdf", (0.650000, 0.200000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    obj = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "basket",
            "e3bae8da192ab3d4a17ae19fa77775ff",
            "e3bae8da192ab3d4a17ae19fa77775ff.urdf",
        ),
        scale=1,
    )
    s.import_object(obj)
    obj.set_position_orientation([0.85, 0.00000, 1.0], [0, 0, 1, 1])
    obj = ArticulatedObject(os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "ball",
            "ball_000",
            "ball_000.urdf",
        ),
        scale = 0.25    
    )
    s.import_object(obj)
    obj.set_position_orientation([0.650000,  -0.050000, 0.750000], [0, 0, 1, 1])
    
    config = parse_config(os.path.join(igibson.configs_path, "robots", "nico.yaml"))
    nico = Nico(**config["robot"])
    s.import_object(nico)
    nico.set_position_orientation([0.5, 0, 0.7], [0, 0, 0, 1])

    # Main simulation loop
    print("FINISHED ALL LOADING, STARTING LOOP")
    while True:
        s.step()

        event_data = s.get_vr_events()
        # print(event_data)
        if event_data:
            print("----- Next set of event data (on current frame): -----")
            for event in event_data:
                readable_event = ["left_controller" if event[0] == 0 else "right_controller", event[1], event[2]]
                print("Event (controller, button_idx, press_id): {}".format(readable_event))
            print("------------------------------------------------------")

        nico.apply_action(s.gen_vr_robot_action())

        # End demo by pressing overlay toggle
        #if s.query_vr_event("left_controller", "overlay_toggle"):
        #    print("PRESSED THE BUTTON, ENDING THE SIMULATION")
        #    break

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()