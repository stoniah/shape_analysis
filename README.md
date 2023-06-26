# shape_analysis
Extracting cardiac shape features from echocardiographic records

shape_features_script.py extracts features from LV contours into a table
It can be run from command line as 

  usage: python shape_features_script.py [-h] [-i N] [-m] [-f F] [-r R] [-o OUT] dir
  
  Left Ventricle Functional Geometry. Extracts functional geometry indexes into a table
  
  positional arguments:
    dir                   directory with segmented left ventricle records
  
  options:
    -h, --help            show this help message and exit
    -i N, --interpolate N
                          perform time interpolation setting record frame count to N (default: N=30)
    -m, --max_interpolate
                          set N for --interpolate flag as max record length in dir, overrides the --interpolate flag
    -f F, --fourier F     perform time interpolation with F fourier harmonics to ensure continuity between last and first frames
    -r R, --regions R     LV is subdivided into R regions for calculation of heterogeneity indexes (default is 20)
    -o OUT, --out OUT     filename where to save the table with indexes

Needed packages for the script:
  scipy, numpy, shapely, pandas, tqdm
  
It is recommended to use separate python virtual environment for the scripts.
It can be created as
  conda create --name shape_analysis python=3.9 numpy scipy shapely pandas tqdm
Then to run the programs activate the environment
  conda activate shape_analysis
Then run the script with
  python shape_features_script.py
