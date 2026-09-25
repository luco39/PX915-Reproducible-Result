from pathlib import Path
import os
import subprocess

home_dir = Path(__file__).parent.resolve()                  # Returns the directory containing this script

start_traj = home_dir / "Interpolated_reaction_path.xyz"    # Interpolated reaction path to start the NEB from

neb_dir = home_dir / f"neb_runs/run1"                       # Where you want the NEB results to end up

model_path_list = ",".join(
    f"Models/used_models_compiled/invariant16_{i:02d}_compiled.model"
    for i in range(1, 6)
)

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

subprocess.run(command, check=True)

os.system(f'python {home_dir}/goeneb/optplot.py -f {neb_dir}/results/optlog.csv -s {neb_dir}/results/visual.png')           # Uses GoeNEB in-built plotting script to visualise the NEB run
os.system(f'python {home_dir}/scripts/plot_neb_convergence.py {neb_dir}/results --save {neb_dir}/results/diagnostic.png')   # Quick plot of the NEB barrier convergence to check the run
