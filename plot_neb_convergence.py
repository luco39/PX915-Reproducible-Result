"""
Plot convergence of the NEB run, for quick diagnostic checks
"""

import argparse
from pathlib import Path

from goeneb_runner_common import (plot_convergence_diagnostics,
                                  DEFAULT_MAX_RMSF_TOL, DEFAULT_MAX_ABSF_TOL,
                                  DEFAULT_CI_RMSF_TOL, DEFAULT_CI_ABSF_TOL)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path,
                   help='results directory (containing optlog.csv) or a direct path to optlog.csv')
    p.add_argument('--save', type=Path, default=None,
                   help='save the figure to this path (e.g. diagnostics.png)')
    p.add_argument('--show', action='store_true', default=False,
                   help='display the figure interactively (blocks until closed)')
    p.add_argument('--max-rmsf-tol', type=float, default=DEFAULT_MAX_RMSF_TOL,
                   help=f'path RMSF tolerance to draw (default: goeneb stock default {DEFAULT_MAX_RMSF_TOL})')
    p.add_argument('--max-absf-tol', type=float, default=DEFAULT_MAX_ABSF_TOL,
                   help=f'path AbsF tolerance to draw (default: {DEFAULT_MAX_ABSF_TOL})')
    p.add_argument('--ci-rmsf-tol', type=float, default=DEFAULT_CI_RMSF_TOL,
                   help=f'climbing image RMSF tolerance to draw (default: {DEFAULT_CI_RMSF_TOL})')
    p.add_argument('--ci-absf-tol', type=float, default=DEFAULT_CI_ABSF_TOL,
                   help=f'climbing image AbsF tolerance to draw (default: {DEFAULT_CI_ABSF_TOL})')
    args = p.parse_args()

    if args.save is None and not args.show:
        p.error('nothing to do: pass --save PATH and/or --show')

    plot_convergence_diagnostics(args.path, savepath=args.save, show=args.show,
                                 Max_RMSF_tol=args.max_rmsf_tol, Max_AbsF_tol=args.max_absf_tol,
                                 CI_RMSF_tol=args.ci_rmsf_tol, CI_AbsF_tol=args.ci_absf_tol)

    if args.save is not None:
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
