# bakalarna-praca

Needed files for loading robot NICO in iGibson:
- nico.py
- nico.yaml
- nico_upper_head_rh6d_dual.urdf
- meshes in .STL and .OBJ format

Needed files for using nico in VR:
- simulator_nico_vr.py
- vr_config.yaml

We are using modified nico_upper_head_rh6d_dual.urdf file. You can find all NICO's URDF files and meshes in https://github.com/incognite-lab/myGym/tree/nico-sim2real/myGym/envs/robots/nico.

Meshes need to be also in the .OBJ format as iGibson cannot parse shapes from the .STL format.

Testing scripts can be found in the testing_scripts directory.
