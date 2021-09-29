These MiMiC input files are taken from the Acetone MiMiC tutorial as of July 2020.
- ``cpmd.inp`` has been modified to perform a single step of MD with very small timestep and save the force trajectory.
- ``equilibrated.gro`` are the coordinates after the GROMACS equilibration (i.e., using the force field, no QM region).
- ``mimic.pdb`` is the last frame of the production run (``mimic.gro``) converted to pdb format.
