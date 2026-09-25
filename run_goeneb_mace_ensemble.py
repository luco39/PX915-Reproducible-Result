"""Run a GoNEB NEB calculation driven by an ensemble of MACE models, for
active-learning uncertainty quantification.

Unlike run_goeneb_mace.py (single model), this loads N independently-trained
MACE checkpoints and evaluates every image with every model each iteration.
The NEB path itself is driven by the *mean* energy/force across all N models
(a single shared path, not N independent ones -- see
goeneb/interfaces/mace_ensemble_interface.py's module docstring for why).
Each model's own per-image energies are de-meaned before comparing across
models, so a constant reference-energy offset between independently trained
models isn't mistaken for genuine disagreement. If the per-image energy
disagreement (std across models, worst image) ever exceeds
--threshold (eV), the NEB stops immediately and the current path is written
out -- meant to be relabelled with a higher level of theory and used as new
training data, not treated as a converged reaction path.

Climbing image is fully compatible with this mode: there is only ever one
path (driven by the mean), so CI doesn't cause per-model paths to diverge
the way it would if each model ran its own independent NEB.

GoNEB is not pip-installed; its source directory is added to sys.path (via
goeneb_runner_common). Must be run with an interpreter that has mace-torch +
ase installed (chemcoord too, if using --interp-mode internal), not the
sandbox/system python:

Example:
    MACE_venv/bin/python run_goeneb_mace_ensemble.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --model-paths model_0.model,model_1.model,model_2.model,model_3.model,model_4.model \\
        --threshold 0.05 \\
        --climbing-image
"""

import argparse
import json
from pathlib import Path
import numpy as np

np.random.seed(343)

from goeneb_runner_common import resolve_jobdir, run_and_get_barrier


def max_step_type(s):
    """argparse type for --max-step: accepts a float, or 'none' to disable
    step clipping entirely (goeneb treats max_step=None that way)."""
    if s.lower() == 'none':
        return None
    return float(s)


def build_mace_ensemble_model_lines(model_paths, threshold_eV, device='cpu', dtype='float64',
                                    batch_images=False, max_batch_atoms=0,
                                    uncertainty_metric='energy'):
    """Build the mace_ensemble_* lines of the ini. model_paths is a list of
    checkpoint paths (at least 2, goeneb itself enforces this and raises
    MissingKeyword otherwise)."""
    if len(model_paths) < 2:
        raise ValueError(f"Need at least 2 model paths for an ensemble, got {len(model_paths)}.")
    joined_paths = ','.join(str(p) for p in model_paths)
    lines = (f"mace_ensemble_model_paths = {joined_paths}\n"
            f"mace_ensemble_uncertainty_threshold = {threshold_eV}\n"
            f"mace_ensemble_device = {device}\n"
            f"mace_ensemble_dtype = {dtype}\n"
            f"mace_ensemble_uncertainty_metric = {uncertainty_metric}\n"
            f"mace_ensemble_batch_images = {batch_images}\n"
            f"mace_ensemble_max_batch_atoms = {int(max_batch_atoms or 0)}\n")
    return lines


def run_mace_ensemble_neb(start_xyz=None, end_xyz=None, model_paths=None,
                          threshold_eV=None, jobdir=None, *,
                          starttraj=None,
                          interp_mode='internal', n_images=11,
                          climbing_image=True, relaxed_neb=False,
                          sidpp=False, idpp=None,
                          step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                          spring_gradient='difference', k_const=0.003,
                          use_vark=False, vark_min_fac=0.1,
                          use_analytical_springpos=False, tangents='henkjon',
                          device='cpu', dtype='float64', maxiter=500, verbose='debug',
                          batch_images=False, max_batch_atoms=0,
                          uncertainty_metric='energy'):
    """Run a GoNEB NEB calculation using an ensemble of MACE models and
    return the resulting barrier plus uncertainty-trigger info.

    model_paths: list of at least 2 MACE checkpoint paths (a common ensemble
    size for query-by-committee-style active learning is 5, but this is
    entirely up to what checkpoints you have trained -- goeneb only requires
    at least 2 to compute a meaningful disagreement).

    threshold_eV: stop the NEB as soon as the max per-image energy
    disagreement (std across models, after de-meaning each model's own path)
    exceeds this, in eV. Required.

    Structures, interp_mode, n_images, climbing_image, relaxed_neb, sidpp,
    idpp, step_pred_method, max_step, stepsize_fac, spring_gradient,
    k_const, use_vark, vark_min_fac, use_analytical_springpos, tangents:
    same as run_goeneb_mace.run_mace_neb() -- see that function's docstring
    and goeneb_runner_common.write_common_ini()'s docstring for full
    details. Climbing image is safe to use here (see this module's
    docstring).

    Returns a dict with everything barrier_from_results() returns
    (path_energies_Eh, left/right barrier, jobdir, resultdir, output_log),
    plus:
    - 'uncertainty_triggered': bool, whether the run stopped early due to
      the uncertainty threshold rather than converging or reaching maxiter
    - 'uncertainty_info': the parsed ensemble_uncertainty.json contents if
      triggered, else None

    batch_images: pack every image of the path into a single forward pass per
    model, instead of calling the calculator once per image. An iteration then
    costs n_models calls rather than n_models x n_images. The models cannot be
    batched with each other (different weights), only the images within each.
    Measured ~7x faster on CPU for an 11-image, 15-atom path, with energies
    bit-identical to the per-image loop and gradients agreeing to 1e-18.

    Unlike the UMA equivalent, a batch that raises is not fatal: that model
    falls back to the per-image loop for the rest of the run, so a single
    pathological geometry costs speed rather than the whole path.

    max_batch_atoms: cap the atoms in one forward, so a long path or a large
    complex is split across several batches instead of exhausting GPU memory.
    0 (default) means one batch for the whole path.
    """
    if threshold_eV is None:
        raise ValueError("threshold_eV is required (the uncertainty threshold, in eV).")
    if model_paths is None or len(model_paths) < 2:
        raise ValueError("model_paths must be a list of at least 2 MACE checkpoint paths.")

    if start_xyz is not None:
        start_xyz = Path(start_xyz).resolve()
    if end_xyz is not None:
        end_xyz = Path(end_xyz).resolve()
    if starttraj is not None:
        starttraj = Path(starttraj).resolve()
    model_paths = [Path(p).resolve() for p in model_paths]

    jobdir = resolve_jobdir(jobdir, 'goeneb_mace_ensemble_runs')
    model_lines = build_mace_ensemble_model_lines(model_paths, threshold_eV, device, dtype,
                                                 batch_images, max_batch_atoms,
                                                 uncertainty_metric)

    result = run_and_get_barrier(jobdir, 'mace_ensemble', model_lines,
                                 start_xyz=start_xyz, end_xyz=end_xyz, starttraj=starttraj,
                                 interp_mode=interp_mode, n_images=n_images,
                                 climbing_image=climbing_image, relaxed_neb=relaxed_neb,
                                 sidpp=sidpp, idpp=idpp,
                                 step_pred_method=step_pred_method, max_step=max_step,
                                 stepsize_fac=stepsize_fac,
                                 spring_gradient=spring_gradient, k_const=k_const,
                                 use_vark=use_vark, vark_min_fac=vark_min_fac,
                                 use_analytical_springpos=use_analytical_springpos,
                                 tangents=tangents,
                                 maxiter=maxiter, verbose=verbose)

    # The marker is now written whenever the running maximum improves, not only
    # on a trigger, so that a run finishing at maxiter still reports how close it
    # came. Its 'triggered' field decides, not its existence. Marker files
    # written before that change always carried triggered=True, so defaulting to
    # True keeps them reading correctly.
    marker_path = result['resultdir'] / 'ensemble_uncertainty.json'
    if marker_path.exists():
        info = json.loads(marker_path.read_text())
        result['uncertainty_triggered'] = bool(info.get('triggered', True))
        result['uncertainty_info'] = info
    else:
        result['uncertainty_triggered'] = False
        result['uncertainty_info'] = None

    return result


def print_ensemble_report(result: dict):
    """CLI printout distinguishing "stopped due to uncertainty" from
    "converged" or "reached maxiter" -- barrier numbers from a
    driver-stopped run are NOT a converged barrier and shouldn't be read as
    one; the path itself (finaltraj.xyz) is the useful output in that case,
    meant for relabelling."""
    kJmol_per_eV = 96.4853075
    kJmol_in_kcalmol = 1 / 4.184

    print(f"\nResults directory: {result['resultdir']}")
    print(f"Full log:           {result['output_log']}")

    if result['uncertainty_triggered']:
        info = result['uncertainty_info']
        print(f"\n*** STOPPED EARLY: ensemble uncertainty threshold exceeded ***")
        print(f"Iteration:                {info['iteration']}")
        print(f"Worst-disagreement image: {info['image_index']}")
        print(f"Max uncertainty (eV):     {info['max_uncertainty_eV']:.4f}")
        print(f"Threshold (eV):           {info['threshold_eV']:.4f}")
        print(f"Ensemble size:            {info['n_models']} models")
        print(f"\nThe path at this point (results/finaltraj.xyz) is NOT converged -- "
             f"this is expected. Relabel it at a higher level of theory and add it "
             f"to the training set. The barrier numbers below describe this "
             f"intermediate, unconverged path only.")
    else:
        print(f"\nNo uncertainty trigger -- the run either converged normally or "
             f"reached maxiter without any model disagreement exceeding threshold.")

    left_eV = result['left_barrier_kJmol'] / kJmol_per_eV
    right_eV = result['right_barrier_kJmol'] / kJmol_per_eV
    print(f"\nBarrier w.r.t. left end:  {result['left_barrier_kJmol']:.3f} kJ/mol "
         f"({result['left_barrier_kJmol'] * kJmol_in_kcalmol:.3f} kcal/mol, {left_eV:.4f} eV)")
    print(f"Barrier w.r.t. right end: {result['right_barrier_kJmol']:.3f} kJ/mol "
         f"({result['right_barrier_kJmol'] * kJmol_in_kcalmol:.3f} kcal/mol, {right_eV:.4f} eV)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, help='start structure .xyz (with --end)')
    p.add_argument('--end', type=Path, help='end structure .xyz (with --start)')
    p.add_argument('--starttraj', type=Path,
                   help='existing multi-frame xyz path to start NEB from directly, '
                        'instead of --start/--end (e.g. a labelled_data/*.xyz file)')

    p.add_argument('--model-paths', required=True,
                   help='comma-separated list of at least 2 MACE checkpoint paths '
                        '(a common ensemble size is 5, but this is up to you)')
    p.add_argument('--threshold', type=float, required=True,
                   help='uncertainty threshold in eV: stop the NEB as soon as the '
                        'worst-image energy disagreement (std across models, after '
                        'de-meaning each model\'s own path) exceeds this')

    p.add_argument('--jobdir', type=Path, default=None,
                   help='where to run the job (default: timestamped dir under '
                        './goeneb_mace_ensemble_runs)')
    p.add_argument('--interp-mode', default='internal',
                   choices=['internal', 'cartesian', 'geodesic'],
                   help='initial path interpolation method (default: internal/z-matrix)')
    p.add_argument('--n-images', type=int, default=11)
    p.add_argument('--climbing-image', action='store_true', default=True)
    p.add_argument('--no-climbing-image', dest='climbing_image', action='store_false')
    p.add_argument('--relaxed-neb', action='store_true', default=False)
    p.add_argument('--sidpp', action='store_true', default=False,
                   help='build the initial path with Sequential IDPP instead of '
                        '--interp-mode + plain IDPP (ignored with --starttraj)')
    p.add_argument('--idpp', dest='idpp', action='store_true', default=None,
                   help='force a plain IDPP pre-optimization pass on the initial path '
                        '(default: on when interpolating from --start/--end, off when '
                        'using --starttraj)')
    p.add_argument('--no-idpp', dest='idpp', action='store_false',
                   help='force-disable the IDPP pre-optimization pass')
    p.add_argument('--step-pred-method', default='amgd',
                   choices=['amgd', 'sd', 'nr', 'rfo', 'l-nr', 'l-rfo', 'sct'],
                   help="step predictor (default: 'amgd')")
    p.add_argument('--max-step', type=max_step_type, default=0.05,
                   help="max per-image step norm per iteration (default: 0.05; "
                        "pass 'none' to disable clipping entirely)")
    p.add_argument('--stepsize-fac', type=float, default=0.2,
                   help="scales the raw step every predictor takes (default: 0.2)")
    p.add_argument('--spring-gradient', default='difference',
                   choices=['difference', 'projected', 'raw'],
                   help="spring force definition (default: 'difference', Henkelman & "
                        "Jonsson's improved version)")
    p.add_argument('--k-const', type=float, default=0.003,
                   help='spring force constant in Eh/Angstrom^2 (default: 0.003; if '
                        '--use-vark, this is the maximum spring constant instead)')
    p.add_argument('--use-vark', action='store_true', default=False,
                   help="use Henkelman & Jonsson's variable-k spring scheme (stiffer "
                        "springs near high-energy/TS region) instead of a fixed --k-const")
    p.add_argument('--vark-min-fac', type=float, default=0.1,
                   help='only used with --use-vark: minimum spring constant is '
                        'this fraction of --k-const (default: 0.1)')
    p.add_argument('--use-analytical-springpos', action='store_true', default=False,
                   help="use the analytical spring positioning scheme instead of "
                        "force-based image spacing (not compatible with --step-pred-method sct)")
    p.add_argument('--tangents', default='henkjon', choices=['henkjon', 'simple'],
                   help="tangent estimation method (default: 'henkjon', Henkelman & "
                        "Jonsson's improved estimate)")
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    p.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    p.add_argument('--uncertainty-metric', default='energy', choices=['energy', 'force'],
                   help="what the --threshold measures: 'energy' (default) is the std "
                        "across models of the per-image energy after de-meaning each "
                        "model's own path, in eV. 'force' is the worst atom's RMS force "
                        "disagreement, in eV/Angstrom -- needs no de-meaning, is intensive "
                        "so one value means the same for small and large complexes, and is "
                        "the quantity the NEB optimiser actually moves the path with.")
    p.add_argument('--batch-images', dest='batch_images', action='store_true', default=False,
                   help='pack every image into one forward pass per model per iteration '
                        '(n_models calls instead of n_models x n_images -- markedly faster, '
                        'especially on GPU). A batch that raises falls back to the per-image '
                        'loop for that model, so it is not all-or-nothing.')
    p.add_argument('--max-batch-atoms', type=int, default=0,
                   help='with --batch-images, cap the atoms per forward pass so a long path '
                        'or large complex is split across several batches (default: 0, one '
                        'batch for the whole path)')
    p.add_argument('--maxiter', type=int, default=500)
    p.add_argument('--verbose', default='debug',
                   choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = p.parse_args()

    have_ends = args.start is not None and args.end is not None
    if have_ends == (args.starttraj is not None):
        p.error("specify either both --start and --end, or --starttraj, not both/neither")

    model_paths = [s.strip() for s in args.model_paths.split(',') if s.strip()]

    result = run_mace_ensemble_neb(args.start, args.end, model_paths, args.threshold, args.jobdir,
                                   starttraj=args.starttraj,
                                   interp_mode=args.interp_mode,
                                   n_images=args.n_images, climbing_image=args.climbing_image,
                                   relaxed_neb=args.relaxed_neb, sidpp=args.sidpp, idpp=args.idpp,
                                   step_pred_method=args.step_pred_method, max_step=args.max_step,
                                   stepsize_fac=args.stepsize_fac,
                                   spring_gradient=args.spring_gradient, k_const=args.k_const,
                                   use_vark=args.use_vark, vark_min_fac=args.vark_min_fac,
                                   use_analytical_springpos=args.use_analytical_springpos,
                                   tangents=args.tangents,
                                   device=args.device,
                                   dtype=args.dtype, maxiter=args.maxiter, verbose=args.verbose,
                                   batch_images=args.batch_images,
                                   max_batch_atoms=args.max_batch_atoms,
                                   uncertainty_metric=args.uncertainty_metric)

    print_ensemble_report(result)


if __name__ == '__main__':
    main()
