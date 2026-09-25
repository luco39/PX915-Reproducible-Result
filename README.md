# PX915-Reproducible-Result

## Reproducible Result

The main result to reproduce here is the converged barrier height of the MACE ensemble-run NEB calculation.

This should output a forward barrier of 187.542 kJ/mol, with an ensemble uncertainty of +- 3.6 kJ/mol.

The scripts are intended to be run locally on your laptop, and should take roughly 10 mins on a Mac.

## Installation and Setup

To copy the repository to your local machine

```
git clone https://github.com/luco39/PX915-Reproducible-Result
cd PX915-Reproducible-Result
```

Then setup up the Python environment. This codebase should work for Python 3.9.6 - other versions are untested

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Obtaining the Result

To reproduce the exact result from the project report, simply run

```
python run_reproducible_result.py
```
! A warning about a library "cuequivariance" may pop up - ignore this, it is only relevant when using GPU

The results directory will be found under `neb_runs`. The progress of the NEB can be followed along here, in the file `neb_runs/output.log`. 

Additionally, after the run is finished, further details about the run can be found under `neb_runs/results/`, including plots visualising some of the NEB metrics.


