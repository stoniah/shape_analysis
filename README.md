# shape_analysis
Extracting cardiac shape features from echocardiographic records

## Extracting basic features
shape_features_script.py extracts features from LV contours into a table
  >usage: python shape_features_script.py [-h] [-i N] [-m] [-f F] [-r R] [-o OUT] dir
  >
  >Left Ventricle Functional Geometry. Extracts functional geometry indexes into a table
  >
  >positional arguments:
  >  dir                   directory with segmented left ventricle records
  >
  >options:
  >  -h, --help            show this help message and exit
  >  -i N, --interpolate N
  >                        perform time interpolation setting record frame count to N (default: N=30)
  >  -m, --max_interpolate
  >                        set N for --interpolate flag as max record length in dir, overrides the --interpolate flag
  >  -f F, --fourier F     perform time interpolation with F fourier harmonics to ensure continuity between last and first frames
  >  -r R, --regions R     LV is subdivided into R regions for calculation of heterogeneity indexes (default is 20)
  >  -o OUT, --out OUT     filename where to save the table with indexes

Needed packages for the script:
  scipy, numpy, shapely, pandas, tqdm
  
It is recommended to use separate python virtual environment for the scripts.
Creating, activating the envrionment, and running the script can be done with the following commands through conda

  ```
  conda create --name shape_analysis python=3.9 numpy scipy shapely pandas tqdm
  conda activate shape_analysis
  python shape_features_script.py [path_to_input_dir]
```
## Extracting statistical shape analysis features
shape_analysis_script.py

>usage: lvssa [-h] [-i N] [-m] [-f F] [-s] [-o OUT] dir
>
>Left Ventricle Statistical Shape Analysis - LV SSA. Performs SSA on LV cardiac cycles to extract features describing LV shape and LV deformation throughout cardiac cycle. Shape space components (SS) describe differences
>in shape, trajectory space components (TS) describe differences in deformation.
>
>positional arguments:
>  dir                   directory with segmented left ventricle records
>
>options:
>  -h, --help            show this help message and exit
>  -i N, --interpolate N
>                        perform time interpolation setting record frame count to N (default: N=30)
>  -m, --max_interpolate
>                        set N for --interpolate flag as max record length in dir, overrides the --interpolate flag
>  -f F, --fourier F     perform time interpolation with F fourier harmonics to ensure continuity between last and first frames
>  -s, --scale           remove scale differences during shape analysis (recommended)
>  -o OUT, --out OUT     directory where to save the output files

