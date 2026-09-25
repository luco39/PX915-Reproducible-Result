# PX915-Reproducible-Result


## Installation and Setup

To copy the repository to your local machine

```
git clone https://github.com/luco39/PX915-Reproducible-Result
cd PX915-Reproducible-Result
```

Then setup up the Python environment. This codebase should work for any Python version between 3.9.6 and 3.14.4

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

