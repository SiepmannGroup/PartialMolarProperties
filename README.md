# PartialMolarProperties
Supporting Information for "Partial molar properties from molecular simulation using multiple linear regression," DOI: 10.1080/00268976.2019.1648898, available at https://www.tandfonline.com/doi/full/10.1080/00268976.2019.1648898

Contains the source code for the version of Monte Carlo for Complex Chemical Systems (MCCCS-MN) used in this work, as well as sample input files (fort.4, fort.77, and topmon.inp), submit script (run.sh), and output files (run.production1, run.production2, etc., and fort12.production1, fort12.production2, etc.) for one independent simulation of the ternary methane/n-butane/n-decane mixture at 333 K and 16220 kPa in the NpT-Gibbs ensemble. 

Also contains a Python program to calculate partial molar properties from simulation trajectory data using linear regression, quadratic regression, and Gaussian process.. The format of the simulation trajecories is the same as `fort12.*` files in the example data folder, without the header line (see `data-55/` folder). The program is invoked as:
```
partial_molar_property.py [-h] [-n N] -b NBOX [-p PRESSURE] [-i INTERVAL] [--train TRAIN] path
positional arguments:
  path                  Path to trajectory files, using fort.12 format: [box lengths (angstrom)] [energy (K)] [pressure (kPa)] [molecule
                        numbers], if the path contains nested directory then each subdirectory represents one state point

optional arguments:
  -h, --help            show this help message and exit
  -n N                  Number of independent simulations to load, use 0 to read all trajectories in [path].
  -b NBOX, --nbox NBOX  Number of simulation boxes
  -p PRESSURE, --pressure PRESSURE
                        Set pressure of NpT simulation in MPa
  -i INTERVAL, --interval INTERVAL
                        Keep every [i] cycles in the trajectory, i should be a multuple of the pressure calculation interval to obtain the
                        correct enthalpy.
  --train TRAIN         Fraction ot training samples in the trajectory.
```
