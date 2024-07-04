These MiMiC input files are taken from the Acetone MiMiC tutorial as of July 2020.
- ``cpmd.inp`` has been modified to perform a single step of MD with very small timestep and save the force trajectory.
- ``gromacs.mdp`` this is the GROMACS ``mdp`` file used to run MiMiC's calculations.
- ``gromacs-only.mdp`` this is the GROMACS ``mdp`` file used to run pure GROMACS calculations (no QM).
- ``equilibrated.gro`` are the coordinates after the GROMACS equilibration (i.e., using the force field, no QM region).
- ``equilibrated-forces-gromacs-only.trr`` GROMACS trr file holding the forces of the ``equilibrated.gro`` frame computed
  with ``gromacs-only.mdp``.
- ``mimic.pdb`` is the last frame of the production run (``mimic.gro``) converted to pdb format.
- ``mimic-forces-gromacs-only.trr`` GROMACS trr file holding the forces of the ``equilibrated.gro`` frame computed with
  ``gromacs-only.mdp``.
