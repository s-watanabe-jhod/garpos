# GARPOS

<img src="https://github.com/s-watanabe-jhod/garpos/assets/68180987/ed955d3c-4c3b-4ca3-91d5-f57b876cfa7b" width=400 alt="GARPOS">

"GARPOS" (GNSS-Acoustic Ranging combined POsitioning Solver) is an analysis tool for GNSS-Acoustic seafloor positioning.

### Version
Latest version is GARPOS v1.2.0 (Jul. 22. 2026)

#### Major change(s)
* v1.2.0: to delete the use of Scikit-sparse, and to add metadata in initcfg.ini and obs.csv. 
* v1.1.0: correcting a bug in second-derivative of B-spline and add first derivative constraint (can be selected in Settings.ini)
* v1.0.2: to apply a mode for "array take-over" (solve each transponder's position and parallel disp. simultaneously)
* v1.0.2: parameter "invtyp" is deleted. Users can set zero in config files to solve limited parameter(s), instead.
* v1.0.1: to set B-spline's knots by time interval (also need to change "Setting.ini" file)
* v1.0.1: to use Cholesky decomposition (module "sksparse" is needed)

# Citation

### for methodology

Watanabe, S., Ishikawa, T., Yokota, Y., & Nakamura, Y. (2020). GARPOS: analysis software for the GNSS-A seafloor positioning with simultaneous estimation of sound speed structure, Front. Earth Sci. (https://doi.org/10.3389/feart.2020.597532).

### for code

Shun-ichi Watanabe, Tadashi Ishikawa, Yuto Nakamura & Yusuke Yokota. (2024). GARPOS: Analysis tool for GNSS-Acoustic seafloor positioning (Version 1.0.2). Zenodo. (https://doi.org/10.5281/zenodo.12620693)

## Corresponding author

* Shun-ichi Watanabe
* Hydrographic and Oceanographic Department, Japan Coast Guard
* Website : https://www1.kaiho.mlit.go.jp/KOHO/chikaku/kaitei/sgs/index.html (in Japanese)


# License

"GARPOS" is distributed under the [GPL 3.0] (https://www.gnu.org/licenses/gpl-3.0.html) license.


# Requirements (tested environment)
* python=3.12
  - numpy>=1.22
  - scipy>=1.8
  - pandas>=1.4
  - matplotlib>=3.5

* Fortran 90 compiler (e.g., gfortran)

Environments under [Anaconda for Linux](https://www.anaconda.com/distribution/) is tested.


### Compilation of Fortran90-based library

For the calculation of travel time, a Fortran90-based library is needed.
For example, the library can be compiled in f90lib via gfortran as,

```bash
cd bin/garpos_v120/f90lib/
make all
```

Path to the library should be indicated in "Settings.ini".


# Usage

When using GARPOS, you should prepare the following files.
* Initial site-parameter file (e.g., *initcfg.ini)
* Acoustic observation data csv file
* Reference sound speed data csv file
* Settings file (e.g., Settings.ini)

"bin/solveSingleEpoch.py" is a driver code. 
Two observation epochs are stored in "sample" directory as demo data.

```bash
cd sample
./demo.sh
```

or run the program manually. 

```bash
cd sample

# to solve position for each transponder (for epoch SAGA.1903)
solveSingleEpoch.py -i Settings-prep.ini -f initcfg/SAGA/SAGA.1903.kaiyo_k4-initcfg.ini -d demo_prep/SAGA
# to solve position for each transponder (for epoch SAGA.1905)
solveSingleEpoch.py -i Settings-prep.ini -f initcfg/SAGA/SAGA.1905.meiyo_m5-initcfg.ini -d demo_prep/SAGA

# to make the averaged array
makeAveFixCfg.py -d cfgfix --res_singles "demo_prep/SAGA/*res.dat"

# to solve in array-constraint condition (for epoch SAGA.1903)
solveSingleEpoch.py -i Settings-fix.ini -f cfgfix/SAGA/SAGA.1903.kaiyo_k4-fix.ini -d demo_res/SAGA
# to solve in array-constraint condition (for epoch SAGA.1905)
solveSingleEpoch.py -i Settings-fix.ini -f cfgfix/SAGA/SAGA.1905.meiyo_m5-fix.ini -d demo_res/SAGA
```

The following files will be created in the directory (specified with "-d" option).
* Estimated site-parameter file (*res.dat)
* Modified acoustic observation data csv file (*obs.csv)
* Model parameter list file (*m.p.dat)
* A posteriori covariance matrix file (*var.dat)


# Note

Please be aware of your storage when searching hyperparameters,
since it will create result files for all combinations of hyperparameters.


### Index list of obs.csv data
| Index       | Description |
|:-----------:| :--- |
| SET         | Names of subset in each observation (typically S01, S02,...) |
| LN          | Names of survey lines in each observation (typically L01, L02,...) |
| MT          | ID of mirror transponder (should be consistent with Site-parameter file) |
| TT          | Observed travel time |
| ResiTT      | Residuals of travel time (observed - calculated) |
| TakeOff     | Takeoff angle of ray path (in degrees, Zenith direction = 180 deg.) |
| Azimuth     | Azimuth of ray path (in degrees) |
| gamma       | Correction term setting in the observation equations |
| flag        | True: data of this acoustic shot is not used as data |
| ST          | Transmission time of acoustic signal |
| ant_e0      | GNSS antenna position (eastward) at ST |
| ant_n0      | GNSS antenna position (northward) at ST |
| ant_u0      | GNSS antenna position (upward) at ST |
| head0       | Heading at ST (in degree) |
| pitch0      | Pitch at ST (in degree) |
| roll0       | Roll at ST (in degree) |
| RT          | Reception time of acoustic signal |
| ant_e1      | GNSS antenna position (eastward) at RT |
| ant_n1      | GNSS antenna position (northward) at RT |
| ant_u1      | GNSS antenna position (upward) at RT |
| head1       | Heading at RT (in degree) |
| pitch1      | Pitch at RT (in degree) |
| roll1       | Roll at RT (in degree) |
| ping_id     | ID of acoustic ping in each dataset |
| dV0         | Sound speed variation (for dV0) |
| gradV1e     | Sound speed variation (for east component of grad(V1)) |
| gradV1n     | Sound speed variation (for north component of grad(V1)) |
| gradV2e     | Sound speed variation (for east component of grad(V2)) |
| gradV2n     | Sound speed variation (for north component of grad(V2)) |
| dV          | Correction term transformed into sound speed variation (gamma x V0) |
| LogResidual | Actual residuals in estimation (log(TT) - log(calculated TT)) |
|             |             | |

*Some parameters will be updated after the analysis.

