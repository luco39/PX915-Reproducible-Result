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

## Playing Around 

If you go into the file `run_reproducible_result.py`, you will see the following code block

```
command = [
    "python",
    f"{home_dir}/scripts/run_goeneb_mace_ensemble.py",
    "--jobdir", neb_dir,                                    # Where you want the NEB results to end up
    "--starttraj", start_traj,                              # Interpolated reaction path to start the NEB from
    "--model-paths", model_path_list,                       # Paths to all models in the ensemble
    "--threshold", "100",                                   # Uncertainty threshold at which to terminate the NEB (as we are testing the NEB to full convergence here, the threshold is set arbitrarily high)
    "--climbing-image",                                     # Turns Climbing Image mode on (helps push the images to the saddle point of the path)
    "--use-vark",                                           # Allows the spring constant to vary between image pairs (lets the images be loose near the saddle point, allowing exploration)
    "--k-const", "0.02",                                    # Maximum value of the spring constant (this is not empirical, just a result of testing - see report for effect of changing this value)
    "--uncertainty-metric", "energy",                       # Which metric (energy/force) to use for the uncertainty threshold (here energy is used to get error bars for the energy barrier height)
    "--batch-images",                                       # Batches all NEB images into a single MACE call (just speeds things up)
    "--maxiter", "1000",                                    # Max iterations of the NEB, set high here to drive it to convergence
]
```

You can try playing around with some of these settings. One suggestion could be to alter the value of `k-const`, as it was noted in the report that even a small change to this value can greatly impact the convergence behaviour of the NEB. Other parameters that could influence this are `climbing-image` (turned off with the flag `--no-climbing-image`) and `use-vark` (turned off by deleting this flag).

The other parameters are not actually that interesting to change.


