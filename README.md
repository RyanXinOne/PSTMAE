# P-STMAE

This repository includes training and evaluation codes for P-STMAE model and RNN baselines. The specific version of code for different datasets can be found in the corresponding branches.

## Running Experiments

Navigate to the desired branch of dataset and run the training script. For example, to train the P-STMAE model:

```sh
export PYTHONPATH='./'
python models/timae/train.py
```

## Dataset Availability

The shallow water dataset is simulated with the script `utils/simulate_sw.py`. The diffusion reaction and compressible Navier-Stokes dataset are downloaded from [PDEBench](https://github.com/pdebench/PDEBench) project. The NOAA sea surface temperature dataset can be accessed from [NOAA website](https://www.ncei.noaa.gov/products/optimum-interpolation-sst).
