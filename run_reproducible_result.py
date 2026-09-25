from pathlib import Path
import os
import subprocess

home_dir = Path(__file__).parent.resolve()      # Returns the directory containing this script

start_traj = home_dir / "Interpolated_reaction_path.xyz"

neb_dir = home_dir / f"neb_runs/run1"

model_path_list = ",".join(
    f"Models/used_models_compiled/invariant16_{i:02d}_compiled.model"
    for i in range(1, 6)
)

command = [
    "python",
    "run_goeneb_mace_ensemble.py",
    "--jobdir", neb_dir,
    "--starttraj", start_traj,
    "--model-paths", model_path_list,
    "--threshold", "100",
    "--climbing-image",
    "--use-vark",
    "--k-const", "0.02",
    "--uncertainty-metric", "energy",
    "--batch-images",
    "--maxiter", "1000",
]

subprocess.run(command, check=True)

os.system(f'python {home_dir}/goeneb/optplot.py -f {neb_dir}/results/optlog.csv -s {neb_dir}/results/visual.png')
os.system(f'python {home_dir}/plot_neb_convergence.py {neb_dir}/results --save {neb_dir}/results/diagnostic.png')
