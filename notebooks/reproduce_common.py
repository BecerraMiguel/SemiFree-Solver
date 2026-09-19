"""Shared helpers for the reproducibility notebooks in this folder.

The five notebooks (physical validation, computational validation for Nx=0 and
Nx=1, and the two FreeGSNKE/freegs comparisons) import this module after
cloning the repository, so that they stay short and all use the same code.

Conventions
-----------
* ``repo``  : path of the cloned repository.
* ``work``  : scratch directory where runs and figures are written
              (``<work>/Nx0/<mesh>``, ``<work>/Nx1/<mesh>``, ``<work>/freegs_*``).
* ``tag``   : mesh label, one of ``MESH_TAGS`` (e.g. ``'151x261'``).
* Every expensive step is skipped when its outputs already exist in ``work``,
  so a notebook can be re-run (or resumed after a disconnection, if ``work`` is
  on Google Drive) without repeating finished runs.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

MESH_TAGS = ['82x142', '100x172', '151x261', '213x368', '301x521']
QUICK_TAGS = ['82x142', '100x172', '151x261']
XPOINT_RZ = (1.45, -1.30)
FREEGS_COMMIT = '9fbec2056b4322ea253bb5614896d549365163f3'

# Generous wall-clock limits (hours) for one solver run on Colab.
SEMIFREE_TIMEOUT_H = {'82x142': 2, '100x172': 2, '151x261': 3, '213x368': 6, '301x521': 11}
# Grid used by freegs (2**k+1 points per axis are required by its Romberg quadrature),
# and wall-clock limit (hours) for the Picard loop.
FREEGS_GRID = {'82x142': (65, 129), '100x172': (129, 129), '151x261': (129, 257),
               '213x368': (257, 257), '301x521': (257, 513)}
FREEGS_TIMEOUT_H = {'82x142': 1, '100x172': 1.5, '151x261': 3, '213x368': 6, '301x521': 12}

# ---------------------------------------------------------------------------
# Reference values from the validation runs (used only to print side-by-side
# comparisons; nothing here feeds the computations).
# ---------------------------------------------------------------------------
REFERENCE = {
    'Nx0': {
        'psi_p': {'full': (1.555, 0.889), 'interior': (2.751, 0.968), 'excl_coils': (2.959, 0.988)},
        'coil_p': (2.96, 0.946),
        'coil_rel': [(5.12e-2, 1.14e-2), (6.90e-3, 2.19e-3), (8.04e-3, 1.57e-3), (7.28e-4, 2.41e-4)],
    },
    'Nx1': {
        'psi_p': {'full': (0.899, 0.953), 'interior': (2.476, 0.996), 'excl_coils': (1.620, 0.936)},
        'coil_p': (1.40, 0.795),
        'coil_rel': [(4.02e-2, 7.35e-3), (1.21e-1, 2.82e-2), (2.89e-2, 6.39e-3), (1.81e-2, 4.14e-3)],
        'xpoint': {  # psi [Wb/rad], B_R [T], B_Z [T] at the X-point
            '82x142': (-1.55e-6, 6.40e-6, 2.51e-3), '100x172': (9.31e-7, -2.49e-6, 5.85e-3),
            '151x261': (-7.39e-7, 1.26e-6, -4.86e-3), '213x368': (-1.38e-7, 7.84e-8, -1.99e-3),
            '301x521': (-1.14e-8, 2.32e-8, -1.85e-4)},
    },
    # Time [min] through psi_check.txt (reference run on Colab).
    'psi_minutes': {
        'Nx0': {'82x142': 0.65, '100x172': 0.95, '151x261': 2.91, '213x368': 8.62, '301x521': 28.57},
        'Nx1': {'82x142': 1.14, '100x172': 1.67, '151x261': 4.65, '213x368': 12.76, '301x521': 40.57},
    },
    # freegs Picard time [s] (reference run on Colab).
    'freegs_seconds': {
        'unsym': {'82x142': 59.0, '100x172': 294.5, '151x261': 753.9, '213x368': 1827.6, '301x521': 5399.1},
        'sym': {'82x142': 59.2, '100x172': 119.8, '151x261': 308.1, '213x368': 412.6, '301x521': 1130.6},
    },
    # Mean |psi_freegs - psi_semifree| [Wb/rad], interior / exterior, symmetrized run.
    'freegs_sym_diff': {
        '82x142': (9.49e-4, 7.23e-4), '100x172': (6.26e-4, 6.28e-4), '151x261': (3.78e-4, 3.60e-4),
        '213x368': (3.60e-4, 4.70e-4), '301x521': (4.32e-4, 4.29e-4)},
    'physical': {'boundary_max': 9.43e-3, 'boundary_mean': 1.94e-3,
                 'currents_kA': (-255.0, 64.3), 'mean_abs_current_kA': 130.9,
                 'far_field_dB': 7e-4, 'interior_dB': 0.114},
}


# ---------------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------------
def default_work_dir(use_drive=False):
    """Return the working directory; with ``use_drive`` it lives on Google Drive
    (survives a Colab disconnection and is shared between the notebooks)."""
    env = os.environ.get('REPRODUCE_WORK')
    if env:
        return env
    if use_drive:
        from google.colab import drive
        drive.mount('/content/drive')
        return '/content/drive/MyDrive/semifree_reproduction'
    return '/content/work'


def sh(cmd, cwd=None, check=True):
    """Run a command, echo its combined output tail, return (returncode, output)."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (res.stdout or '') + (res.stderr or '')
    if check and res.returncode != 0:
        print(out[-4000:])
        raise RuntimeError(f'command failed ({res.returncode}): {" ".join(cmd)}')
    return res.returncode, out


def environment_report():
    print(sh(['g++', '--version'])[1].splitlines()[0])
    print('CPUs:', os.cpu_count())
    try:
        print(sh(['python3', '--version'])[1].strip())
    except Exception:
        pass


def cfg_path(repo, tag):
    return f'{repo}/configs/DIII-D_{tag}.json'


def load_cfg(repo, tag):
    with open(cfg_path(repo, tag)) as f:
        return json.load(f)


def npoints(repo, tag):
    m = load_cfg(repo, tag)['mesh']
    return m['npr'] * m['npz']


def h_of(repo, tag):
    """Radial mesh step h = (r_max - r_min)/(npr - 1), used in the order fits."""
    m = load_cfg(repo, tag)['mesh']
    return (m['r_max'] - m['r_min']) / (m['npr'] - 1)


def mesh_table(repo, tags):
    print(f'{"Mesh":>10} {"npr":>5} {"npz":>5} {"N":>9} {"h [m]":>12}')
    for t in tags:
        m = load_cfg(repo, t)['mesh']
        print(f'{t:>10} {m["npr"]:>5} {m["npz"]:>5} {m["npr"]*m["npz"]:>9,} {h_of(repo, t):>12.4e}')


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
_PATCH_BEGIN = "// BZ_check.txt / BR_check.txt: closed-form"
_PATCH_END = "fclose(out3);"


def patch_no_bfield(text):
    """Return SemiFree_Solver.cpp source with the dense B_R/B_Z sweeps disabled.

    Only psi and the coil currents are needed by the convergence studies; the
    two closed-form B sweeps roughly triple the run time. The repository copy
    is never modified: the notebooks compile this patched *copy*.
    """
    lines = text.split('\n')
    begin = [i for i, l in enumerate(lines) if _PATCH_BEGIN in l]
    end = [i for i, l in enumerate(lines) if _PATCH_END in l]
    if len(begin) != 1 or len(end) != 1 or end[0] < begin[0]:
        raise RuntimeError('SemiFree_Solver.cpp no longer has the expected B-field block; '
                           'cannot build the psi-only variant.')
    lines.insert(end[0] + 1, '#endif')
    lines.insert(begin[0], '#if 0')
    return '\n'.join(lines)


def build_semifree(repo, work, skip_bfield=False):
    """Compile the semi-free solver with -fopenmp and return the binary path.

    ``skip_bfield=False`` uses the repository ``Makefile`` (full output, including
    BR_check.txt / BZ_check.txt). ``skip_bfield=True`` compiles a patched copy that
    omits the B sweeps (psi_check.txt and the coil currents are unchanged).
    """
    t0 = time.time()
    if not skip_bfield:
        sh(['make', '-C', repo, 'build/semifree_solver'])
        binary = f'{repo}/build/semifree_solver'
    else:
        os.makedirs(f'{work}/build', exist_ok=True)
        src = open(f'{repo}/src/semifree_boundary/SemiFree_Solver.cpp').read()
        patched = f'{work}/build/SemiFree_Solver_psi_only.cpp'
        with open(patched, 'w') as f:
            f.write(patch_no_bfield(src))
        binary = f'{work}/build/semifree_solver_psi_only'
        sh(['g++', '-O3', '-fopenmp', '-std=c++11', f'-I{repo}/src/third_party',
            f'-I{repo}/src/semifree_boundary', '-w', '-o', binary, patched, '-lm'])
    linked = 'gomp' in sh(['ldd', binary])[1]
    print(f'Binary: {binary}  (built in {time.time()-t0:.0f} s, OpenMP: {"OK" if linked else "NOT LINKED"})')
    if not linked:
        raise RuntimeError('The binary is not linked against OpenMP (-fopenmp).')
    return binary


# ---------------------------------------------------------------------------
# Running the semi-free solver
# ---------------------------------------------------------------------------
_RE_PSI_MIN = re.compile(r'Elapsed time through psi_check\.txt[^:]*:\s*([0-9.eE+-]+)\s*minutes')
_RE_TOTAL_MIN = re.compile(r'El codigo se demoro:\s*([0-9.eE+-]+)\s*minutos')


def stage_case(repo, work, mode, tag, subdir=None):
    """Copy a public case into ``<work>/<subdir>/<tag>`` under the file names the solver reads.

    Nx0: Dshape.txt, Jt.txt, coils.txt.
    Nx1: the X-point boundary and its consistent J_phi are renamed to Dshape.txt / Jt.txt
         (the solver is then run with boundary-transformation method 0, i.e. as-is).
    """
    src = f'{repo}/cases/DIII-D_{tag}'
    dst = f'{work}/{subdir or mode}/{tag}'
    os.makedirs(dst, exist_ok=True)
    names = ({'Dshape.txt': 'Dshape.txt', 'Jt.txt': 'Jt.txt', 'coils.txt': 'coils.txt'} if mode == 'Nx0' else
             {'Dshape_xpoint.txt': 'Dshape.txt', 'Jt_xpoint.txt': 'Jt.txt', 'coils.txt': 'coils.txt'})
    for s, d in names.items():
        shutil.copy(f'{src}/{s}', f'{dst}/{d}')
    return dst


def read_timing(case_dir, log_tag):
    out = dict(elapsed_s=None, psi_min=None, total_min=None)
    tpath = f'{case_dir}/_{log_tag}_timing.txt'
    if os.path.exists(tpath):
        for line in open(tpath):
            if line.startswith('elapsed_seconds='):
                out['elapsed_s'] = float(line.strip().split('=')[1])
    lpath = f'{case_dir}/_{log_tag}_stdout.log'
    if os.path.exists(lpath):
        text = open(lpath).read()
        m1, m2 = _RE_PSI_MIN.search(text), _RE_TOTAL_MIN.search(text)
        out['psi_min'] = float(m1.group(1)) if m1 else None
        out['total_min'] = float(m2.group(1)) if m2 else None
    return out


def run_semifree(binary, repo, work, mode, tag, subdir=None, timeout_h=None, force=False,
                 need_bfield=False):
    """Run the semi-free solver on one public case (Nx0: no X-point; Nx1: X-point at XPOINT_RZ).

    Skipped (results reused) when the outputs of a previous run are already in ``work``.
    Returns the timing dictionary.
    """
    subdir = subdir or mode
    case = stage_case(repo, work, mode, tag, subdir)
    log_tag = f'semifree_{subdir}_{tag}'
    needed = ['psi_check.txt', 'corrientes.txt', 'sample_points_check.txt', f'_{log_tag}_timing.txt']
    if need_bfield:
        needed += ['BR_check.txt', 'BZ_check.txt']
    if not force and all(os.path.exists(f'{case}/{n}') for n in needed):
        t = read_timing(case, log_tag)
        print(f'[{tag}] existing results reused ({subdir}); '
              f'recorded time through psi_check: {t["psi_min"]} min')
        return t

    cfg = cfg_path(repo, tag)
    stdin = f'{cfg}\n0\n' if mode == 'Nx0' else f'{cfg}\n1\n{XPOINT_RZ[0]}\n{XPOINT_RZ[1]}\n0\n'
    timeout_s = (timeout_h or SEMIFREE_TIMEOUT_H[tag]) * 3600
    print(f'>>> [{tag}] {subdir}: running the solver (limit {timeout_s/3600:.1f} h) ...')
    t0 = time.time()
    timed_out = False
    res = None
    try:
        res = subprocess.run([binary], input=stdin, cwd=case, capture_output=True, text=True,
                             timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.time() - t0
    stdout = '' if res is None else res.stdout
    open(f'{case}/_{log_tag}_stdout.log', 'w').write(stdout)
    with open(f'{case}/_{log_tag}_timing.txt', 'w') as f:
        f.write(f'elapsed_seconds={elapsed:.6f}\ntimed_out={timed_out}\n'
                f'returncode={-1 if res is None else res.returncode}\n')
    if timed_out:
        raise RuntimeError(f'[{tag}] the solver exceeded the time limit ({timeout_s/3600:.1f} h).')
    if res.returncode != 0:
        print(res.stderr[-3000:])
        raise RuntimeError(f'[{tag}] the solver exited with code {res.returncode}.')
    t = read_timing(case, log_tag)
    print(f'>>> [{tag}] finished in {elapsed/60:.2f} min (through psi_check: {t["psi_min"]} min)')
    return t


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------
def load_run(repo, work, mode, tag, subdir=None):
    """Load one finished run (psi, currents, boundary, sample points, timing) or None."""
    subdir = subdir or mode
    d = f'{work}/{subdir}/{tag}'
    if not os.path.exists(f'{d}/psi_check.txt') or not os.path.exists(f'{d}/corrientes.txt'):
        return None
    m = load_cfg(repo, tag)['mesh']
    Rg = np.linspace(m['r_min'], m['r_max'], m['npr'])
    Zg = np.linspace(m['z_min'], m['z_max'], m['npz'])
    psi = np.loadtxt(f'{d}/psi_check.txt')
    assert psi.shape == (m['npz'], m['npr']), f'{tag}: psi shape {psi.shape}'
    run = dict(tag=tag, dir=d, Rg=Rg, Zg=Zg, psi=psi,
               currents=np.loadtxt(f'{d}/corrientes.txt'),   # columns: R, Z, I [A]
               boundary=np.loadtxt(f'{d}/Dshape.txt'),
               timing=read_timing(d, f'semifree_{subdir}_{tag}'))
    for name, fn in [('samples', 'sample_points_check.txt'), ('BRg', 'BR_check.txt'),
                     ('BZg', 'BZ_check.txt'), ('boundary_check', 'psi_boundary_check.txt')]:
        p = f'{d}/{fn}'
        run[name] = np.loadtxt(p) if os.path.exists(p) else None
    return run


def load_runs(repo, work, mode, tags, subdir=None):
    runs = {t: load_run(repo, work, mode, t, subdir) for t in tags}
    missing = [t for t, r in runs.items() if r is None]
    if missing:
        print('Meshes without results (skipped):', missing)
    return {t: r for t, r in runs.items() if r is not None}


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------
def loglog_fit(hs, errs):
    """Least-squares fit ln(e) = p ln(h) + c. Returns (p, c, R^2)."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    ok = np.isfinite(hs) & np.isfinite(errs) & (errs > 0) & (hs > 0)
    if ok.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    x, y = np.log(hs[ok]), np.log(errs[ok])
    p, c = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (p * x + c)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(p), float(c), float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def psi_convergence(repo, runs, tags, disc_radius=0.02):
    """L2 error of psi on the finest mesh, in three regions, and fitted orders p.

    Each coarser psi is interpolated onto the finest mesh with a bivariate spline of order 2.
    Regions: full domain, plasma interior (inside the run's own boundary), and full domain
    without discs of radius ``disc_radius`` [m] around every PF coil.
    """
    from matplotlib.path import Path
    from scipy.interpolate import RectBivariateSpline
    tags = [t for t in tags if t in runs]
    finest = runs[tags[-1]]
    RR, ZZ = np.meshgrid(finest['Rg'], finest['Zg'])
    inside = Path(finest['boundary']).contains_points(
        np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
    dcoil = np.min([np.hypot(RR - c[0], ZZ - c[1]) for c in finest['currents'][:, :2]], axis=0)
    masks = {'full': np.ones_like(inside), 'interior': inside, 'excl_coils': dcoil > disc_radius}
    hs = [h_of(repo, t) for t in tags[:-1]]
    errs = {k: [] for k in masks}
    for t in tags[:-1]:
        r = runs[t]
        interp = RectBivariateSpline(r['Zg'], r['Rg'], r['psi'], kx=2, ky=2)(finest['Zg'], finest['Rg'])
        for k, msk in masks.items():
            errs[k].append(float(np.sqrt(np.mean((interp[msk] - finest['psi'][msk]) ** 2))))
    fits = {k: loglog_fit(hs, errs[k]) for k in masks}
    return dict(tags=tags[:-1], h=hs, errors=errs, fits=fits, reference=tags[-1])


REGION_LABELS = {'full': 'Full domain', 'interior': 'Plasma interior',
                 'excl_coils': 'Full domain, excluding coils'}


def print_psi_table(res, mode):
    ref = REFERENCE[mode]['psi_p']
    print(f'{"Region":<40} {"p":>7} {"R^2":>7}   {"ref. p":>8} {"ref. R^2":>10}')
    for k, lab in REGION_LABELS.items():
        p, _, r2 = res['fits'][k]
        print(f'{lab:<40} {p:>7.3f} {r2:>7.3f}   {ref[k][0]:>8.3f} {ref[k][1]:>10.3f}')


def plot_psi_convergence(res, mode, fname=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (k, lab) in zip(axes, REGION_LABELS.items()):
        x, y = np.log(res['h']), np.log(res['errors'][k])
        p, c, r2 = res['fits'][k]
        ax.plot(x, y, 'o', color='#d95f02', label='data')
        if np.isfinite(p):
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, p * xx + c, '-', color='#1b9e77', label=f'fit: p={p:.3f}, $R^2$={r2:.3f}')
        ax.set_xlabel(r'$\ln(h)$')
        ax.set_ylabel(r'$\ln(e_k)$, $L_2$ norm of $\psi$')
        ax.set_title(lab, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'Convergence of $\\psi$ with respect to the {res["reference"]} mesh ({mode})')
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def coil_convergence(repo, runs, tags):
    """Relative change of the coil currents between consecutive meshes, and order fit vs. finest."""
    tags = [t for t in tags if t in runs]
    rows = []
    for a, b in zip(tags[:-1], tags[1:]):
        Ia, Ib = runs[a]['currents'][:, 2], runs[b]['currents'][:, 2]
        rel = np.abs(Ib - Ia) / np.abs(Ib)
        rows.append(dict(pair=f'{a} -> {b}', max=float(rel.max()), mean=float(rel.mean())))
    Iref = runs[tags[-1]]['currents'][:, 2]
    hs = [h_of(repo, t) for t in tags[:-1]]
    errs = [float(np.linalg.norm(runs[t]['currents'][:, 2] - Iref)) for t in tags[:-1]]
    return dict(rows=rows, h=hs, errors=errs, fit=loglog_fit(hs, errs), reference=tags[-1])


def print_coil_table(res, mode):
    ref = REFERENCE[mode]
    print(f'{"Mesh pair":<24} {"max":>11} {"mean":>11}   {"ref. max":>10} {"ref. mean":>12}')
    for i, r in enumerate(res['rows']):
        tm = ref['coil_rel'][i] if i < len(ref['coil_rel']) and len(res['rows']) == 4 else (float('nan'),) * 2
        print(f'{r["pair"]:<24} {r["max"]:>11.2e} {r["mean"]:>11.2e}   {tm[0]:>10.2e} {tm[1]:>12.2e}')
    p, _, r2 = res['fit']
    print(f'\nConvergence order of the coil currents: p = {p:.2f}, R^2 = {r2:.3f}   '
          f'(reference: p = {ref["coil_p"][0]}, R^2 = {ref["coil_p"][1]})')


def print_timing_table(runs, tags, mode):
    ref = REFERENCE['psi_minutes'][mode]
    print(f'{"Mesh":>10} {"through psi_check [min]":>24} {"total [min]":>13}   {"reference (Colab) [min]":>24}')
    for t in tags:
        if t not in runs:
            continue
        tm = runs[t]['timing']
        f = lambda v: 'n/d' if v is None else f'{v:.2f}'
        print(f'{t:>10} {f(tm["psi_min"]):>24} {f(tm["total_min"]):>13}   {ref[t]:>24.2f}')


def plot_timing(repo, series, fname=None, title='Run time of the semi-free solver'):
    """``series``: {label: {tag: minutes}} -- plotted against N = npr*npz."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, d) in enumerate(series.items()):
        pts = sorted((npoints(repo, t), v) for t, v in d.items() if v is not None)
        if pts:
            x, y = zip(*pts)
            style = '--' if 'reference' in label else '-'
            ax.plot(x, y, 'o' + style, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N = n_{pr}\,n_{pz}$')
    ax.set_ylabel('Time through psi_check.txt [min]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Nx = 1 specifics
# ---------------------------------------------------------------------------
def xpoint_table(runs, tags):
    """psi, B_R, B_Z evaluated exactly at the X-point (last row of sample_points_check.txt)."""
    ref = REFERENCE['Nx1']['xpoint']
    print(f'{"Mesh":>10} {"psi [Wb/rad]":>14} {"B_R [T]":>12} {"B_Z [T]":>12}   '
          f'{"ref. psi":>11} {"ref. B_R":>11} {"ref. B_Z":>11}')
    for t in tags:
        if t not in runs or runs[t]['samples'] is None:
            continue
        s = runs[t]['samples'][-1]
        print(f'{t:>10} {s[2]:>14.3e} {s[3]:>12.3e} {s[4]:>12.3e}   '
              f'{ref[t][0]:>11.2e} {ref[t][1]:>11.2e} {ref[t][2]:>11.2e}')


def _coil_squares(ax, cur, label=None):
    ax.scatter(cur[:, 0], cur[:, 1], marker='s', s=25, c='white', edgecolors='k', linewidths=0.6,
               label=label, zorder=5)


def plot_xpoint_flux(run, fname=None):
    """Nx=1: poloidal flux over the whole mesh domain. The cyan curve is the psi = 0 isocontour
    (it passes through the X-point; small loops around isolated coils are genuine psi = 0 curves)."""
    import matplotlib.pyplot as plt
    RX, ZX = XPOINT_RZ
    RR, ZZ = np.meshgrid(run['Rg'], run['Zg'])
    levels = np.linspace(-1.0, 0.25, 31)
    fig, ax = plt.subplots(figsize=(6, 7))
    cf = ax.contourf(RR, ZZ, run['psi'], levels=levels, cmap='inferno', extend='both')
    ax.contour(RR, ZZ, run['psi'], levels=levels, colors='k', linewidths=0.4)
    ax.contour(RR, ZZ, run['psi'], levels=[0.0], colors='c', linewidths=2)
    _coil_squares(ax, run['currents'])
    ax.plot([RX], [ZX], 'r+', ms=18, mew=2)
    fig.colorbar(cf, ax=ax, shrink=0.85).set_label(r'$\psi$ [Wb/rad]', fontsize=12)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(rf'DIII-D ($N_x=1$, X-point ({RX:.2f},{ZX:.2f})): Poloidal Flux $\psi(R,Z)$', fontsize=11)
    ax.set_xlim(R_LIM[0], R_LIM[1])
    ax.set_ylim(run['Zg'][0], run['Zg'][-1])
    ax.set_xticks(np.arange(R_LIM[0], R_LIM[1] + 0.01, 0.25))
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def plot_xpoint_zoom(run, fname=None):
    """Nx=1: zoom at the X-point. Thick cyan: psi = 0. Thin lines: psi = +-0.001, +-0.003, +-0.005,
    +-0.01 (solid positive, dashed negative). The colormap saturates outside [-0.27, 0.06]."""
    import matplotlib.pyplot as plt
    RX, ZX = XPOINT_RZ
    RR, ZZ = np.meshgrid(run['Rg'], run['Zg'])
    fig, ax = plt.subplots(figsize=(6.0, 5.9))
    cf = ax.contourf(RR, ZZ, run['psi'], levels=np.arange(-1.0, 0.7501, 0.01), cmap='inferno',
                     extend='both', vmin=-0.27, vmax=0.06)
    cs = ax.contour(RR, ZZ, run['psi'], levels=[-0.01, -0.005, -0.003, -0.001, 0.001, 0.003, 0.005, 0.01],
                    colors='deepskyblue', linewidths=1.0)
    ax.clabel(cs, levels=[-0.003, 0.003], fontsize=6, fmt='%g')
    ax.contour(RR, ZZ, run['psi'], levels=[0.0], colors='cyan', linewidths=3)
    _coil_squares(ax, run['currents'])
    ax.plot([RX], [ZX], 'r+', ms=18, mew=2.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85)
    cb.set_label(r'$\psi$ [Wb/rad]', fontsize=11)
    ticks = np.round(np.arange(-0.95, 0.58, 0.19), 2) + 0.0
    cb.set_ticks(ticks)
    cb.set_ticklabels([('0.00' if abs(t) < 1e-9 else f'{t:.2f}'.replace('-', '\u2212')) for t in ticks])
    ax.set_xlim(RX - 0.35, RX + 0.35)
    ax.set_ylim(ZX - 0.30, ZX + 0.30)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Zoom on the X-point region', fontsize=12)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def plot_xpoint_field(run, fname=None):
    """Nx=1: B_R and B_Z along Z at R = R_X, from the derivative of psi (cubic spline)."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline
    RX, ZX = XPOINT_RZ
    sp = RectBivariateSpline(run['Zg'], run['Rg'], run['psi'], kx=3, ky=3)
    zs = np.linspace(max(ZX - 0.5, run['Zg'][0]), ZX + 0.5, 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(zs, -sp(zs, RX, dx=1, grid=False) / RX, label=r'$B_R$')
    ax.plot(zs, sp(zs, RX, dy=1, grid=False) / RX, label=r'$B_Z$')
    ax.axvline(ZX, color='r', ls=':', lw=1)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('B [T]')
    ax.set_title(f'$B_R$ and $B_Z$ at $R={RX}$ m (from $\\partial\\psi$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Physical validation (Nx = 0, single mesh)
# ---------------------------------------------------------------------------
def field_arrays(run):
    """Return R, Z grids and B from Green's functions and from d(psi) (B_R=-1/R dpsi/dZ, B_Z=1/R dpsi/dR)."""
    Rg, Zg, psi = run['Rg'], run['Zg'], run['psi']
    RR, ZZ = np.meshgrid(Rg, Zg)
    dZ, dR = np.gradient(psi, Zg, Rg)
    return dict(Rg=Rg, Zg=Zg, RR=RR, ZZ=ZZ, psi=psi, BRg=run['BRg'], BZg=run['BZg'],
                BRf=-dZ / RR, BZf=dR / RR)


def divergence(F, BR, BZ):
    return np.gradient(F['RR'] * BR, F['Rg'], axis=1) / F['RR'] + np.gradient(BZ, F['Zg'], axis=0)


def physical_summary(run):
    """Boundary residual and coil-current statistics, printed next to the reference values."""
    ref = REFERENCE['physical']
    d = run['boundary_check'][:, 4]
    I = run['currents'][:, 2] / 1e3
    print(f'Boundary points evaluated       : {len(d)}')
    print(f'Residual |psi_b - psi_ref|  max  : {d.max():.3e} Wb/rad   (reference: {ref["boundary_max"]:.2e})')
    print(f'Residual |psi_b - psi_ref|  mean : {d.mean():.3e} Wb/rad   (reference: {ref["boundary_mean"]:.2e})')
    print(f'Coil currents, range            : {I.min():.1f} to {I.max():+.1f} kA   '
          f'(reference: {ref["currents_kA"][0]} to {ref["currents_kA"][1]:+} kA)')
    print(f'Coil currents, mean |I|         : {np.abs(I).mean():.1f} kA   (reference: {ref["mean_abs_current_kA"]} kA)')


def _decorate(ax, F, run, R_LIM, Z_LIM, coils=True):
    bd, cur = run['boundary'], run['currents']
    ax.plot(bd[:, 0], bd[:, 1], '-', color='#111111', lw=1.6)
    if coils:
        ax.scatter(cur[:, 0], cur[:, 1], marker='s', s=15, c='white', edgecolors='k', lw=0.5, zorder=5)
    ax.set_xlim(*R_LIM)
    ax.set_ylim(*Z_LIM)
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')


R_LIM = (0.75, 3.0)
Z_LIM = (-1.75, 1.75)


def plot_flux(F, run, fname=None):
    """Nx=0: poloidal flux over the mesh domain, with the D-shape boundary and the PF coils."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 7.6))
    cf = ax.contourf(F['RR'], F['ZZ'], F['psi'], levels=30, cmap='inferno')
    ax.contour(F['RR'], F['ZZ'], F['psi'], levels=30, colors='k', linewidths=0.3)
    ax.plot(run['boundary'][:, 0], run['boundary'][:, 1], 'c-', lw=2)
    _coil_squares(ax, run['currents'])
    cb = fig.colorbar(cf, ax=ax, aspect=30, pad=0.05, fraction=0.15)
    cb.set_label(r'$\psi$ [Wb/rad]', fontsize=12)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(r'DIII-D: Poloidal Flux $\psi(R,Z)$')
    ax.set_xlim(R_LIM[0], R_LIM[1])
    ax.set_ylim(F['Zg'][0], F['Zg'][-1])
    ax.set_xticks(np.arange(R_LIM[0], R_LIM[1] + 0.01, 0.25))
    fig.tight_layout()
    ax.set_aspect('equal')
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


def plot_field(F, run, BR, BZ, title, fname=None, stride=6):
    import matplotlib.pyplot as plt
    Bmag = np.hypot(BR, BZ)
    ir0, ir1 = np.searchsorted(F['Rg'], R_LIM)
    iz0, iz1 = np.searchsorted(F['Zg'], Z_LIM)
    sl = (slice(iz0, iz1), slice(ir0, ir1))
    safe = np.where(Bmag[sl] > 0, Bmag[sl], 1.0)
    vmax = np.nanpercentile(Bmag[sl], 98)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.contour(F['RR'], F['ZZ'], F['psi'], levels=20, colors='0.15', linewidths=0.5, alpha=0.85)
    q = ax.quiver(F['RR'][sl][::stride, ::stride], F['ZZ'][sl][::stride, ::stride],
                  (BR[sl] / safe)[::stride, ::stride], (BZ[sl] / safe)[::stride, ::stride],
                  np.clip(Bmag[sl][::stride, ::stride], 0, vmax), cmap='plasma', pivot='mid')
    fig.colorbar(q, ax=ax, shrink=0.8).set_label('|B| [T]')
    _decorate(ax, F, run, R_LIM, Z_LIM)
    ax.set_ylabel('Z [m]')
    ax.set_title(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def plot_field_difference(F, run, fname=None):
    import matplotlib.pyplot as plt
    diff = np.hypot(F['BRg'] - F['BRf'], F['BZg'] - F['BZf'])
    fig, ax = plt.subplots(figsize=(6, 8))
    vmax = np.nanpercentile(diff, 98)
    cf = ax.contourf(F['RR'], F['ZZ'], diff, levels=np.linspace(0, vmax, 31), cmap='viridis', extend='max')
    _decorate(ax, F, run, R_LIM, Z_LIM, coils=False)
    ax.set_ylabel('Z [m]')
    ax.set_title('Difference between the two evaluations of B')
    fig.colorbar(cf, ax=ax, shrink=0.8).set_label(r'$|\mathbf{B}_{Green}-\mathbf{B}_{deriv}|$ [T]')
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return diff, fig


def field_statistics(F, run, diff):
    """Mean |B_Green - B_derivative| far from plasma/coils and inside the plasma; div B extremes."""
    from matplotlib.path import Path
    RR, ZZ = F['RR'], F['ZZ']
    inside = Path(run['boundary']).contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
    h = max(F['Rg'][1] - F['Rg'][0], F['Zg'][1] - F['Zg'][0])
    dcoil = np.min([np.hypot(RR - c[0], ZZ - c[1]) for c in run['currents'][:, :2]], axis=0)
    ref = REFERENCE['physical']
    print(f'Far from plasma and coils (> 3h): mean |dB| = {diff[(~inside) & (dcoil > 3*h)].mean():.2e} T'
          f'   (reference: ~{ref["far_field_dB"]:.0e} T)')
    print(f'Plasma interior                  : mean |dB| = {diff[inside].mean():.3f} T'
          f'   (reference: ~{ref["interior_dB"]} T)')
    return inside, dcoil, h


def plot_divergence(F, run, div_g, div_f, far_mask, fname_raw=None, fname_excl=None):
    import matplotlib.pyplot as plt
    figs = []
    for masked, fname in [(False, fname_raw), (True, fname_excl)]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
        dg = np.where(far_mask, div_g, np.nan) if masked else div_g
        for ax, div, title in [(axes[0], dg, r'$\nabla\cdot\mathbf{B}$ (B from Green functions)'),
                               (axes[1], div_f, r'$\nabla\cdot\mathbf{B}$ (B from $\partial\psi$)')]:
            ref = div[far_mask] if masked else div
            vmax = np.nanpercentile(np.abs(ref), 98)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            cf = ax.contourf(F['RR'], F['ZZ'], div, levels=np.linspace(-vmax, vmax, 31), cmap='RdBu_r', extend='both')
            _decorate(ax, F, run, R_LIM, Z_LIM)
            ax.set_title(title, fontsize=10)
            fig.colorbar(cf, ax=ax, shrink=0.7)
        axes[0].set_ylabel('Z [m]')
        fig.suptitle('Divergence of B' + (' (excluding the neighbourhood of the coils)' if masked else ' (unfiltered)'))
        fig.tight_layout()
        if fname:
            fig.savefig(fname, dpi=150, bbox_inches='tight')
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# freegs comparison
# ---------------------------------------------------------------------------
def install_freegs():
    """Install the pinned freegs commit (NumPy-2.x fix included) and check it."""
    import importlib
    import inspect
    sh([sys.executable, '-m', 'pip', 'install', '-q',
        f'git+https://github.com/freegs-plasma/freegs.git@{FREEGS_COMMIT}'])
    importlib.invalidate_caches()
    import freegs.critical
    assert 'f_scalar' in inspect.getsource(freegs.critical.find_critical), \
        'this freegs build lacks the NumPy 2.x fix'
    print('freegs installed and verified:', FREEGS_COMMIT[:8])


def freegs_profile_selftest():
    """dp/dpsiN and dg/dpsiN of the profile class against finite differences."""
    from gs_solver_profile import GSSolverProfile
    xs = np.linspace(0.001, 0.999, 401)
    h = 1e-6
    p = lambda s: 50000.0 * ((1.0 - (1.0 - s) ** 2) ** 2 + 0.2)
    dp_fd = (p(xs + h) - p(xs - h)) / (2 * h)
    err_p = np.max(np.abs(dp_fd - 50000.0 * GSSolverProfile._dp_dpsiN(xs)) / (np.abs(dp_fd) + 1e-12))
    dg_fd = ((xs + h) ** 2 - (xs - h) ** 2) / (2 * h)
    err_g = np.max(np.abs(dg_fd - GSSolverProfile._dg_dpsiN(xs)) / (np.abs(dg_fd) + 1e-12))
    print(f'dp/dpsiN: max relative error = {err_p:.2e};  dg/dpsiN: {err_g:.2e}')
    assert err_p < 1e-4 and err_g < 1e-4
    print('Profile self-test: OK')


def run_freegs_mesh(repo, work, tag, symmetrize, out_subdir, maxits=600, rtol=1e-6,
                    nx0_subdir='Nx0', force=False):
    """freegs free-boundary Picard iteration for one mesh, with the coil currents and
    profiles of the semi-free/fixed-boundary pipeline (mesh ``tag``, Nx=0 run in ``work``).

    ``symmetrize=True`` enforces psi(R,Z) = psi(R,-Z) after every Picard step.
    Results are written to ``<work>/<out_subdir>`` and reused if already there.
    """
    out = f'{work}/{out_subdir}'
    os.makedirs(out, exist_ok=True)
    if not force and all(os.path.exists(f'{out}/{n}') for n in
                         (f'psi_freegs_{tag}.txt', f'{tag}_timing.txt', f'{tag}_history.csv')):
        print(f'[{tag}] existing freegs result reused ({out_subdir})')
        return
    sys.path.insert(0, f'{repo}/src/freegs_comparison')
    import freegs
    from freegs import machine
    from gs_solver_profile import GSSolverProfile

    cfg = load_cfg(repo, tag)
    geo, con, mesh = cfg['geometry'], cfg['constraints'], cfg['mesh']
    case = f'{work}/{nx0_subdir}/{tag}'
    corr = np.loadtxt(f'{case}/corrientes.txt')
    Dshape = np.loadtxt(f'{case}/Dshape.txt')
    npr, npz = FREEGS_GRID[tag]
    timeout_s = FREEGS_TIMEOUT_H[tag] * 3600
    coils = [(f'PF{i}', machine.Coil(R=corr[i, 0], Z=corr[i, 1], current=corr[i, 2], control=False))
             for i in range(corr.shape[0])]
    tokamak = machine.Machine(coils, wall=machine.Wall(R=list(Dshape[:, 0]), Z=list(Dshape[:, 1])))
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=mesh['r_min'], Rmax=mesh['r_max'],
                            Zmin=mesh['z_min'], Zmax=mesh['z_max'], nx=npr, ny=npz, check_limited=False)
    profile = GSSolverProfile(geo['Ro'], geo['a'], geo['kappa'], geo['delta'], con['I_plasma'],
                              con['P_axis'], con['B_axis'], Dshape, eq=eq, dynamic_psi_bndry=False,
                              fixed_psi_bndry=con['Psi_b'])
    print(f'>>> [{tag}] freegs ({npr}x{npz} grid, symmetrized={symmetrize}, limit {timeout_s/3600:.1f} h)')
    psi = eq.psi()
    history, timed_out, it, relchange = [], False, 0, float('nan')
    t0 = time.time()
    for it in range(maxits):
        psi_last = psi.copy()
        eq.solve(profile, psi=psi, psi_bndry=con['Psi_b'])
        psi = eq.psi()
        if symmetrize:
            psi = 0.5 * (psi + psi[:, ::-1])
        relchange = np.max(np.abs(psi - psi_last)) / (np.max(psi) - np.min(psi))
        history.append(dict(it=it, relchange=relchange, psi_axis=profile.psi_axis))
        if it % 10 == 0:
            print(f'  [{tag}] it={it:3d} relchange={relchange:.3e} elapsed={time.time()-t0:.0f}s')
        if relchange < rtol and it > 3:
            print(f'[{tag}] converged at it={it}, relchange={relchange:.3e}')
            break
        if time.time() - t0 > timeout_s:
            timed_out = True
            print(f'*** [{tag}] time limit reached at it={it}')
            break
    elapsed = time.time() - t0
    print(f'[{tag}] {elapsed:.1f} s ({elapsed/60:.2f} min), {it + 1} iterations')
    np.savetxt(f'{out}/psi_freegs_{tag}.txt', psi)
    with open(f'{out}/{tag}_timing.txt', 'w') as f:
        f.write(f'elapsed_seconds={elapsed:.6f}\nfinal_iteration={it}\nfinal_relchange={relchange:.6e}\n'
                f'timed_out={timed_out}\nfreegs_grid={npr}x{npz}\n')
    with open(f'{out}/{tag}_history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['it', 'relchange', 'psi_axis'])
        w.writeheader()
        w.writerows(history)


def load_freegs(repo, work, tag, out_subdir, nx0_subdir='Nx0'):
    """Load one freegs result plus the matching semi-free psi (on freegs' own grid) and statistics."""
    from matplotlib.path import Path
    from scipy.interpolate import RegularGridInterpolator
    out = f'{work}/{out_subdir}'
    if not os.path.exists(f'{out}/psi_freegs_{tag}.txt'):
        return None
    run = load_run(repo, work, 'Nx0', tag, nx0_subdir)
    if run is None:
        return None
    m = load_cfg(repo, tag)['mesh']
    npr, npz = FREEGS_GRID[tag]
    Rf = np.linspace(m['r_min'], m['r_max'], npr)
    Zf = np.linspace(m['z_min'], m['z_max'], npz)
    RF, ZF = np.meshgrid(Rf, Zf, indexing='ij')
    psi_fg = np.loadtxt(f'{out}/psi_freegs_{tag}.txt')
    interp = RegularGridInterpolator((run['Zg'], run['Rg']), run['psi'], bounds_error=False, fill_value=None)
    psi_sf = interp(np.stack([ZF.ravel(), RF.ravel()], axis=-1)).reshape(RF.shape)
    diff = psi_fg - psi_sf
    inside = Path(run['boundary']).contains_points(np.column_stack([RF.ravel(), ZF.ravel()])).reshape(RF.shape)
    info = {}
    for line in open(f'{out}/{tag}_timing.txt'):
        k, v = line.strip().split('=')
        info[k] = v
    return dict(tag=tag, R=RF, Z=ZF, psi_fg=psi_fg, psi_sf=psi_sf, diff=diff, boundary=run['boundary'],
                mean_interior=float(np.abs(diff[inside]).mean()), mean_exterior=float(np.abs(diff[~inside]).mean()),
                elapsed_s=float(info['elapsed_seconds']), iterations=int(info['final_iteration']) + 1,
                relchange=float(info['final_relchange']), timed_out=info['timed_out'] == 'True',
                semifree_psi_min=run['timing']['psi_min'])


def freegs_asymmetry(r):
    """Up-down asymmetry of psi_freegs: mean |psi(R,Z) - psi(R,-Z)| and Z of the flux maximum (axis)."""
    asym = float(np.mean(np.abs(r['psi_fg'] - r['psi_fg'][:, ::-1])))
    i, j = np.unravel_index(np.argmax(r['psi_fg']), r['psi_fg'].shape)
    return asym, float(r['Z'][i, j])


def freegs_table(results, tags, mode):
    ref = REFERENCE['freegs_sym_diff']
    print(f'{"Mesh":>10} {"final rel. change":>18} {"mean|dpsi| inside":>18} {"mean|dpsi| outside":>19} '
          f'{"asymmetry":>10} {"axis Z [m]":>11}' + (f'   {"ref. inside":>12} {"ref. outside":>13}' if mode == 'sym' else ''))
    for t in tags:
        r = results.get(t)
        if r is None:
            continue
        asym, zax = freegs_asymmetry(r)
        line = (f'{t:>10} {r["relchange"]:>14.2e} {r["mean_interior"]:>18.2e} {r["mean_exterior"]:>18.2e} '
                f'{asym:>10.2e} {zax:>10.3f}')
        if mode == 'sym':
            line += f'   {ref[t][0]:>11.2e} {ref[t][1]:>11.2e}'
        print(line)


def plot_freegs_crosscheck(r, fname=None, label=''):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = np.max(np.abs(r['diff']))
    panels = [(r['psi_sf'], r'$\psi_{semi-free}$', 'inferno', {}),
              (r['psi_fg'], r'$\psi_{freegs}$' + label, 'inferno', {}),
              (r['diff'], 'Difference', 'RdBu_r', dict(vmin=-vmax, vmax=vmax))]
    for ax, (field, title, cmap, kw) in zip(axes, panels):
        cf = ax.contourf(r['R'], r['Z'], field, levels=30, cmap=cmap, **kw)
        ax.contour(r['R'], r['Z'], field, levels=30, colors='k', linewidths=0.4)
        ax.plot(r['boundary'][:, 0], r['boundary'][:, 1], 'lime', lw=2)
        ax.set_title(f'{title} ({r["tag"]} mesh)')
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        fig.colorbar(cf, ax=ax, shrink=0.8)
    axes[0].set_ylabel('Z [m]')
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def plot_freegs_error(repo, results, tags, fname=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, lab, mk in [('mean_interior', 'Inside plasma', 'o'), ('mean_exterior', 'Outside plasma', 's')]:
        pts = [(npoints(repo, t), results[t][key]) for t in tags if t in results]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, mk + '-', label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N = n_{pr}\,n_{pz}$')
    ax.set_ylabel(r'$\overline{|\psi_{semi-free}-\psi_{freegs}|}$ [Wb/rad]')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def plot_freegs_timing(repo, curves, fname=None):
    """``curves``: {label: {tag: seconds}}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, d in curves.items():
        pts = sorted((npoints(repo, t), v / 60.0) for t, v in d.items() if v is not None)
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, 'o--' if 'reference' in label else 'o-', label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N = n_{pr}\,n_{pz}$')
    ax.set_ylabel('Time to obtain $\\psi$ [min]')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
    return fig


def zip_results(work, subdirs, zip_path, extra_files=()):
    """Bundle result folders and figures into one zip (offered for download on Colab)."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for sd in subdirs:
            base = f'{work}/{sd}'
            for root, _, files in os.walk(base):
                for fn in files:
                    p = os.path.join(root, fn)
                    z.write(p, arcname=os.path.relpath(p, work))
        for p in extra_files:
            if os.path.exists(p):
                z.write(p, arcname=os.path.relpath(p, work))
    print('Bundle:', zip_path, f'({os.path.getsize(zip_path)/1e6:.1f} MB)')
    return zip_path


def offer_download(path):
    """Trigger a browser download when running on Google Colab (no-op elsewhere)."""
    try:
        from google.colab import files
        files.download(path)
    except Exception:
        print('(not on Colab: the file is at', path, ')')
