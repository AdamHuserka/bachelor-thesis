# bakalarna-praca

Needed files for loading robot NICO in iGibson:
- nico.py
- nico.yaml
- nico_upper_head_rh6d_dual.urdf
- meshes in .STL and .OBJ format

You can find all NICO's URDF files and meshes in https://github.com/incognite-lab/myGym/tree/nico-sim2real/myGym/envs/robots/nico,
we are using the nico_upper_head_rh6d_dual.urdf file, meshes are only in the .STL format, you need to convert them to .OBJ format. 

You can use for example https://imagetostl.com/convert/file/stl/to/obj.

Meshes need to be also in the .OBJ format as iGibson cannot parse shapes from the .STL format.

Testing scripts can be found in the testing_scripts directory.
