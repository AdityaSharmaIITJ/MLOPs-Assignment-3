MLOPs Assignment 3

(base) apple@Mac MLOPs-Assignment-3 % conda create -n ml-env python=3.10
2 channel Terms of Service accepted
Retrieving notices: done
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /opt/anaconda3/envs/ml-env

  added / updated specs:
    - python=3.10


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    openssl-3.0.17             |       h4ee41c1_0         4.3 MB
    python-3.10.18             |       h19e8193_0        12.9 MB
    setuptools-78.1.1          |  py310hca03da5_0         1.7 MB
    wheel-0.45.1               |  py310hca03da5_0         116 KB
    ------------------------------------------------------------
                                           Total:        19.0 MB

The following NEW packages will be INSTALLED:

  bzip2              pkgs/main/osx-arm64::bzip2-1.0.8-h80987f9_6 
  ca-certificates    pkgs/main/osx-arm64::ca-certificates-2025.2.25-hca03da5_0 
  expat              pkgs/main/osx-arm64::expat-2.7.1-h313beb8_0 
  libcxx             pkgs/main/osx-arm64::libcxx-17.0.6-he5c5206_4 
  libffi             pkgs/main/osx-arm64::libffi-3.4.4-hca03da5_1 
  ncurses            pkgs/main/osx-arm64::ncurses-6.4-h313beb8_0 
  openssl            pkgs/main/osx-arm64::openssl-3.0.17-h4ee41c1_0 
  pip                pkgs/main/noarch::pip-25.1-pyhc872135_2 
  python             pkgs/main/osx-arm64::python-3.10.18-h19e8193_0 
  readline           pkgs/main/osx-arm64::readline-8.2-h1a28f6b_0 
  setuptools         pkgs/main/osx-arm64::setuptools-78.1.1-py310hca03da5_0 
  sqlite             pkgs/main/osx-arm64::sqlite-3.50.2-h79febb2_1 
  tk                 pkgs/main/osx-arm64::tk-8.6.14-h6ba3021_1 
  tzdata             pkgs/main/noarch::tzdata-2025b-h04d1e81_0 
  wheel              pkgs/main/osx-arm64::wheel-0.45.1-py310hca03da5_0 
  xz                 pkgs/main/osx-arm64::xz-5.6.4-h80987f9_1 
  zlib               pkgs/main/osx-arm64::zlib-1.2.13-h18a0788_1 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                     
Preparing transaction: done                                                                                          
Verifying transaction: done                                                                                          
Executing transaction: done                                                                                          
#
# To activate this environment, use
#
#     $ conda activate ml-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(base) apple@Mac MLOPs-Assignment-3 % conda activate ml-env
(ml-env) apple@Mac MLOPs-Assignment-3 % pip3 install -r requirements.txt
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
(ml-env) apple@Mac MLOPs-Assignment-3 % pip3 install -r src/requirements.txt
Collecting scikit-learn==1.3.0 (from -r src/requirements.txt (line 1))
  Downloading scikit_learn-1.3.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (11 kB)
Collecting numpy==1.24.3 (from -r src/requirements.txt (line 2))
  Downloading numpy-1.24.3-cp310-cp310-macosx_11_0_arm64.whl.metadata (5.6 kB)
Collecting pandas==2.0.3 (from -r src/requirements.txt (line 3))
  Downloading pandas-2.0.3-cp310-cp310-macosx_11_0_arm64.whl.metadata (18 kB)
Collecting torch==2.0.1 (from -r src/requirements.txt (line 4))
  Downloading torch-2.0.1-cp310-none-macosx_11_0_arm64.whl.metadata (23 kB)
Collecting joblib==1.3.1 (from -r src/requirements.txt (line 5))
  Downloading joblib-1.3.1-py3-none-any.whl.metadata (5.4 kB)
Collecting scipy>=1.5.0 (from scikit-learn==1.3.0->-r src/requirements.txt (line 1))
  Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl.metadata (61 kB)
Collecting threadpoolctl>=2.0.0 (from scikit-learn==1.3.0->-r src/requirements.txt (line 1))
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting python-dateutil>=2.8.2 (from pandas==2.0.3->-r src/requirements.txt (line 3))
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas==2.0.3->-r src/requirements.txt (line 3))
  Using cached pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.1 (from pandas==2.0.3->-r src/requirements.txt (line 3))
  Using cached tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting filelock (from torch==2.0.1->-r src/requirements.txt (line 4))
  Downloading filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)
Collecting typing-extensions (from torch==2.0.1->-r src/requirements.txt (line 4))
  Using cached typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy (from torch==2.0.1->-r src/requirements.txt (line 4))
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx (from torch==2.0.1->-r src/requirements.txt (line 4))
  Downloading networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch==2.0.1->-r src/requirements.txt (line 4))
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas==2.0.3->-r src/requirements.txt (line 3))
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch==2.0.1->-r src/requirements.txt (line 4))
  Downloading MarkupSafe-3.0.2-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.0 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy->torch==2.0.1->-r src/requirements.txt (line 4))
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Downloading scikit_learn-1.3.0-cp310-cp310-macosx_12_0_arm64.whl (9.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.5/9.5 MB 453.6 kB/s eta 0:00:00
Downloading numpy-1.24.3-cp310-cp310-macosx_11_0_arm64.whl (13.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.9/13.9 MB 2.3 MB/s eta 0:00:00
Downloading pandas-2.0.3-cp310-cp310-macosx_11_0_arm64.whl (10.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.8/10.8 MB 2.4 MB/s eta 0:00:00
Downloading torch-2.0.1-cp310-none-macosx_11_0_arm64.whl (55.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.8/55.8 MB 2.4 MB/s eta 0:00:00
Downloading joblib-1.3.1-py3-none-any.whl (301 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl (22.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22.4/22.4 MB 2.4 MB/s eta 0:00:00
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Using cached tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Downloading filelock-3.18.0-py3-none-any.whl (16 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading MarkupSafe-3.0.2-cp310-cp310-macosx_11_0_arm64.whl (12 kB)
Downloading networkx-3.4.2-py3-none-any.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 2.4 MB/s eta 0:00:00
Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 2.4 MB/s eta 0:00:00
Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 2.5 MB/s eta 0:00:00
Using cached typing_extensions-4.14.1-py3-none-any.whl (43 kB)
Installing collected packages: pytz, mpmath, tzdata, typing-extensions, threadpoolctl, sympy, six, numpy, networkx, MarkupSafe, joblib, filelock, scipy, python-dateutil, jinja2, torch, scikit-learn, pandas
Successfully installed MarkupSafe-3.0.2 filelock-3.18.0 jinja2-3.1.6 joblib-1.3.1 mpmath-1.3.0 networkx-3.4.2 numpy-1.24.3 pandas-2.0.3 python-dateutil-2.9.0.post0 pytz-2025.2 scikit-learn-1.3.0 scipy-1.15.3 six-1.17.0 sympy-1.14.0 threadpoolctl-3.6.0 torch-2.0.1 typing-extensions-4.14.1 tzdata-2025.2
(ml-env) apple@Mac MLOPs-Assignment-3 % python src/train.py
Starting California Housing model training...
Dataset shape: (20640, 8)
Target shape: (20640,)
Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Training set size: 16512
Test set size: 4128
Model coefficients shape: (8,)
Model intercept: -37.0232777060639
R² Score: 0.5758
MSE: 0.5559
Model saved to models/california_housing_model.joblib
Test data saved to models/test_data.joblib
Training completed successfully!
(ml-env) apple@Mac MLOPs-Assignment-3 % 
