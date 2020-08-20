# GARPOS

"GARPOS" (GNSS-Acoustic Ranging combined POsitioning Solver) is an analysis tool for GNSS-Acoustic seafloor positioning.

### Version
Latest version is GARPOS v0.1.0 (Jul. 01. 2020)


# Citation

## for methodology

Watanabe, S., Ishikawa, T., Yokota, Y., and Nakamura, Y., (2020), GARPOS: analysis software for the GNSS-A seafloor positioning with simultaneous estimation of sound speed structure

## for code
Shun-ichi Watanabe, Tadashi Ishikawa, Yusuke Yokota, & Yuto Nakamura. (2020, August 20). GARPOS v0.1.0: Analysis tool for GNSS-Acoustic seafloor positioning (Version 0.1.0). Zenodo. http://doi.org/10.5281/zenodo.3992688

### Corresponding author

* Shun-ichi Watanabe
* Hydrographic and Oceanographic Department, Japan Coast Guard
* Website : https://www1.kaiho.mlit.go.jp/KOHO/chikaku/kaitei/sgs/index.html (in Japanese)


# License

"GARPOS" is distributed under the [GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) license.


### Algorithm and documentation

Please see Watanabe, S., Ishikawa, T., Yokota, Y., and Nakamura, Y., (2020)


# Requirements

* Python 3.7.3
* Packages NumPy, Scipy, Pandas, and Matplotlib are also required.
* Fortran 90 compiler (e.g., gfortran)

Environments under [Anaconda for Linux](https://www.anaconda.com/distribution/) is tested.


### Compilation of Fortran90-based library

For the calculation of travel time, a Fortran90-based library is needed.
For example, the library can be compiled via gfortran as,

```bash
gfortran -shared -fPIC -fopenmp -O3 -o lib_raytrace.so sub_raytrace.f90 lib_raytrace.f90
```

Path to the library should be indicated in "Settings.ini".


# Usage

When using GARPOS, you should prepare the following files.
* Initial site-parameter file (e.g., *initcfg.ini)
* Acoustic observation data csv file
* Reference sound speed data csv file
* Settings file (e.g., Settings.ini)

Attached "sample/demo.py" is a sample driver code. 2 observation epochs are stored as demo data.

```bash
cd sample

# to solve position for each transponder (for epoch SAGA.1903)
python demo.py -i Settings-prep.ini -f initcfg/SAGA/SAGA.1903.kaiyo_k4-initcfg.ini -d demo_prep/SAGA
# to solve position for each transponder (for epoch SAGA.1905)
python demo.py -i Settings-prep.ini -f initcfg/SAGA/SAGA.1905.meiyo_m5-initcfg.ini -d demo_prep/SAGA

# to make the averaged array
python makeFixCfg.py -d cfgfix --res_singles "demo_prep/SAGA/*res.dat"

# to solve in array-constraint condition (for epoch SAGA.1903)
python demo.py -i Settings-fix.ini -f cfgfix/SAGA/SAGA.1903.kaiyo_k4-fix.ini -d demo_res/SAGA
# to solve in array-constraint condition (for epoch SAGA.1905)
python demo.py -i Settings-fix.ini -f cfgfix/SAGA/SAGA.1905.meiyo_m5-fix.ini -d demo_res/SAGA
```

The following files will be created in the directory (specified with "-d" option).
* Estimated site-parameter file (*res.dat)
* Modified acoustic observation data csv file (*obs.csv)
* Model parameter list file (*m.p.dat)
* A posteriori covariance matrix file (*var.dat)


# Note

Please be aware of your storage when searching hyperparameters,

since it will create result files for all combinations of hyperparameters.


### List of functions

+ drive_garpos (in garpos_main.py)
 + parallelrun (in garpos_main.py)
   + MAPestimate (in map_estimation.py)
     + init_position (in setup_model.py)
     + make_splineknots (in setup_model.py)
     + derivative2 (in setup_model.py)
     + data_correlation (in setup_model.py)
     + calc_forward (in forward.py)
       + corr_attitude (in coordinate_trans.py)
       + calc_traveltime (in traveltime.py)
     + calc_gamma (in forward.py)
     + jacobian_pos (in forward.py)
       + corr_attitude (in coordinate_trans.py)
       + calc_traveltime (in traveltime.py)
     + outresults (in output.py)

### Index list of obs.csv data

| No. | Index       | Description |
|:---:|:-----------:| :--- |
| 00  | SET         | Names of subset in each observation (typically S01, S02,...) |
| 01  | LN          | Names of survey lines in each observation (typically L01, L02,...) |
| 02  | MT          | ID of mirror transponder (should be consistent with Site-parameter file) |
| 03  | TT          | Observed travel time |
| 04  | ResiTT      | Residuals of travel time (observed - calculated) |
| 05  | TakeOff     | Takeoff angle of ray path (in degrees, Zenith direction = 180 deg.) |
| 06  | gamma       | Correction term setting in the observation equations |
| 07  | flag        | True: data of this acoustic shot is not used as data |
| 08  | ST          | Transmission time of acoustic signal |
| 09  | ant_e0      | GNSS antenna position (eastward) at ST |
| 10  | ant_n0      | GNSS antenna position (northward) at ST |
| 11  | ant_u0      | GNSS antenna position (upward) at ST |
| 12  | head0       | Heading at ST (in degree) |
| 13  | pitch0      | Pitch at ST (in degree) |
| 14  | roll0       | Roll at ST (in degree) |
| 15  | RT          | Reception time of acoustic signal |
| 16  | ant_e1      | GNSS antenna position (eastward) at RT |
| 17  | ant_n1      | GNSS antenna position (northward) at RT |
| 18  | ant_u1      | GNSS antenna position (upward) at RT |
| 19  | head1       | Heading at RT (in degree) |
| 20  | pitch1      | Pitch at RT (in degree) |
| 21  | roll1       | Roll at RT (in degree) |
| 22  | dV0         | Sound speed variation (for dV0) |
| 23  | gradV1e     | Sound speed variation (for east component of grad(V1)) |
| 24  | gradV1n     | Sound speed variation (for north component of grad(V1)) |
| 25  | gradV1e     | Sound speed variation (for east component of grad(V2)) |
| 26  | gradV1n     | Sound speed variation (for north component of grad(V2)) |
| 27  | dV          | Correction term transformed into sound speed variation (gamma x V0) |
| 28  | LogResidual | Actual residuals in estimation (log(TT) - log(calculated TT) |
|     |             | |

*Indices #04-#07, #22-#28 will be updated after the estimation.

### EOF
