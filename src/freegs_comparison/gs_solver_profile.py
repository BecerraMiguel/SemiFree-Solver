"""Direct Python port of GS_solver.h's CURRENT_DENSITY()/ALPHA()/p()/g()/
dp_dpsiN()/dg_dpsiN(), run inside freegs's own Picard loop (freegs.picard.solve).

Works entirely in GS_solver.h's own Lo/Io/Jo/Psio/Po/Bo/Go normalization
internally (converts freegs's physical R,Z,psi on entry, converts Jtor back to
physical A/m^2 on exit) -- this is a deliberate design choice: naively
substituting physical SI quantities directly into ALPHA()'s formula gives an
alpha off by a factor of mu0 (verified by hand, propagating the
Lo/Io/Jo/Psio/Po/Bo/Go normalization chain through ALPHA()'s ratio). Working
in the exact same normalized units as the C++ code sidesteps that pitfall
entirely, since the formulas below are copied verbatim from GS_solver.h
rather than re-derived directly in SI units.

Also handles a sign/convention flip: GS_solver.h defines
PsiN = (Psib-Psi)/(Psib-Psia) (1 at the magnetic axis, 0 at the boundary) --
the OPPOSITE of freegs's own convention psi_norm = (psi-psi_axis)/(psi_bndry-
psi_axis) (0 at axis, 1 at boundary).

Supports two ways of determining psi_bndry (see the notebook for which is used):
  - fixed_psi_bndry (dynamic_psi_bndry=False): psi_bndry is a constant, matching
    GS_solver.h's own Psi_b convention exactly. Verified this session: converges
    cleanly and monotonically with plain (unrelaxed) Picard iteration.
  - dynamic_psi_bndry=True: reads eq.psi_bndry after calling eq._updateBoundaryPsi(),
    i.e. freegs's own wall/limiter-sampling mechanism finds the boundary flux
    itself. Found THIS SESSION to have a genuine Picard instability (psi_axis and
    psi_bndry climb together each iteration -- a real feedback through the
    alpha-for-Ip renormalization, not a formula bug: confirmed by the fixed-psi_bndry
    variant converging cleanly with an identical Jtor() otherwise, and by unit tests
    on dp_dpsiN/dg_dpsiN passing to ~1e-8). Relaxing psi_bndry alone (psib_relax)
    slows but does not eliminate the drift; relaxing alpha alone (alpha_relax) does
    better but still drifts slowly; combining both makes it worse (different failure
    mode). Left in place, off by default, as a documented starting point for future
    work -- not used by the current notebook, which uses fixed_psi_bndry only.
"""

import numpy as np
import matplotlib.path as mpath
from freegs.jtor import Profile
from freegs import critical

MU0 = 4.0e-7 * np.pi


class GSSolverProfile(Profile):
    def __init__(self, Ro, a, kappa, delta, Ip, Pa, Ba, Dshape_RZ,
                 eq=None, dynamic_psi_bndry=False, fixed_psi_bndry=0.0,
                 psib_relax=0.0):
        self.Ro, self.a, self.kappa, self.delta = Ro, a, kappa, delta
        self.Ip, self.Pa_phys, self.Ba = Ip, Pa, Ba

        # Normalization constants, matching GS_solver.h::NORMALIZATION() exactly
        # (GS_solver.h:574-602).
        self.Lo = Ro
        self.Io = Ip
        self.Jo = self.Io / Ro**2
        self.Psio = MU0 * self.Jo * Ro**3
        self.Po = self.Psio * self.Jo / self.Lo
        self.Bo = self.Psio / self.Lo**2
        self.Go = self.Bo * self.Lo

        self.go_phys = Ba * Ro                   # GS_solver.h: go = Ba*Ro
        self.go_hat = self.go_phys / self.Go
        self.Pa_hat = Pa / self.Po
        self.Ip_hat = 1.0                         # always exactly 1 (Io=Ip) by construction

        self.eq = eq
        self.dynamic_psi_bndry = dynamic_psi_bndry
        self.fixed_psi_bndry = fixed_psi_bndry
        # Relaxation applied specifically to psi_bndry (not the whole psi field --
        # blending the whole field was tried and made the instability worse, see
        # notebook notes). Found necessary this session: with dynamic_psi_bndry=True,
        # psi_axis and psi_bndry were found to climb together each Picard iteration
        # (a genuine runaway feedback through the alpha-for-Ip normalization, not an
        # initial-guess artifact -- persisted even when warm-started from psi_semifree).
        # 0.0 = no relaxation (Variant B doesn't need it, psi_bndry is already fixed).
        self.psib_relax = psib_relax
        self._prev_psib_phys = None
        self.alpha_relax = 0.0
        self._prev_alpha_hat = None

        # Fixed spatial mask: "inside the D-shape polygon", matching
        # CURRENT_DENSITY()'s FLAG!=9 restriction, which never changes during
        # the SOR/Picard loop in GS_solver.h.
        self._Dshape_path = mpath.Path(Dshape_RZ)

        # Set on each Jtor() call; exposed for diagnostics and reused by
        # pprime()/ffprime() (see notes on those methods below).
        self.alpha_hat = None
        self.psi_axis = None
        self.psi_bndry = None
        self._last_dpsiN_dpsi_hat = None

    # --- profile shape functions, verbatim from GS_solver.h:1776-1809 -------
    @staticmethod
    def _dp_dpsiN(psiN_gs):
        return 4.0 * (1.0 - psiN_gs) * (1.0 - (1.0 - psiN_gs) ** 2)

    @staticmethod
    def _dg_dpsiN(psiN_gs):
        return 2.0 * psiN_gs

    def _mask(self, R, Z):
        pts = np.column_stack([R.ravel(), Z.ravel()])
        return self._Dshape_path.contains_points(pts).reshape(R.shape)

    def Jtor(self, R, Z, psi, psi_bndry=None):
        mask = self._mask(R, Z)

        # --- boundary flux (physical) ---
        if self.dynamic_psi_bndry:
            assert self.eq is not None, "dynamic_psi_bndry=True requires eq to be set"
            self.eq._updateBoundaryPsi(psi)
            psib_phys = self.eq.psi_bndry
            if psib_phys is None:
                raise RuntimeError(
                    "freegs could not determine psi_bndry dynamically -- check "
                    "Machine(wall=...) and Equilibrium(check_limited=True)."
                )
            if self._prev_psib_phys is not None and self.psib_relax > 0.0:
                psib_phys = ((1.0 - self.psib_relax) * psib_phys
                             + self.psib_relax * self._prev_psib_phys)
                self.eq.psi_bndry = psib_phys  # keep eq's own bookkeeping consistent
            self._prev_psib_phys = psib_phys
        else:
            psib_phys = self.fixed_psi_bndry

        # --- magnetic axis (physical), restricted to inside the D-shape to
        # avoid picking up a spurious local extremum near a coil ---
        opt, _xpt = critical.find_critical(R, Z, psi)
        opt_inside = [o for o in opt if self._Dshape_path.contains_point((o[0], o[1]))]
        if not opt_inside:
            raise RuntimeError("No O-point found inside the D-shape polygon.")
        psia_phys = opt_inside[0][2]

        self.psi_axis, self.psi_bndry = psia_phys, psib_phys

        # --- everything below in GS_solver.h's own normalized units ---
        r_hat = R / self.Lo
        psi_hat = psi / self.Psio
        psia_hat = psia_phys / self.Psio
        psib_hat = psib_phys / self.Psio

        dpsiN_dpsi_hat = -1.0 / (psib_hat - psia_hat)
        self._last_dpsiN_dpsi_hat = dpsiN_dpsi_hat

        # PsiN = (Psib-Psi)/(Psib-Psia): GS_solver.h convention (1 at axis, 0 at boundary)
        psiN = (psib_hat - psi_hat) / (psib_hat - psia_hat)
        psiN = np.clip(psiN, 0.0, 1.0)

        dp_dpsiN = self._dp_dpsiN(psiN)
        dg_dpsiN = self._dg_dpsiN(psiN)

        # ALPHA(): solve for alpha so total Ip matches exactly, integrating
        # ONLY over the fixed D-shape mask (GS_solver.h:1596-1617, sum over FLAG!=9).
        hr_hat = (R[1, 0] - R[0, 0]) / self.Lo
        hz_hat = (Z[0, 1] - Z[0, 0]) / self.Lo

        sum1_hat = np.sum(np.where(mask, r_hat * dp_dpsiN, 0.0))
        sum2_hat = np.sum(np.where(mask, dg_dpsiN / r_hat, 0.0))

        numerator = self.Ip_hat - self.Pa_hat * dpsiN_dpsi_hat * sum1_hat * hr_hat * hz_hat
        denominator = 0.5 * self.go_hat ** 2 * dpsiN_dpsi_hat * sum2_hat * hr_hat * hz_hat
        alpha_hat = numerator / denominator
        if self._prev_alpha_hat is not None and self.alpha_relax > 0.0:
            alpha_hat = ((1.0 - self.alpha_relax) * alpha_hat
                         + self.alpha_relax * self._prev_alpha_hat)
        self._prev_alpha_hat = alpha_hat
        self.alpha_hat = alpha_hat

        Jt_hat = (r_hat * self.Pa_hat * dp_dpsiN * dpsiN_dpsi_hat
                  + 0.5 * self.go_hat ** 2 * alpha_hat * dg_dpsiN * dpsiN_dpsi_hat / r_hat)
        Jt_hat = np.where(mask, Jt_hat, 0.0)

        return self.Jo * Jt_hat

    # --- Required by the abstract Profile base class. DIAGNOSTIC ONLY --
    # (NOT used to construct Jtor above -- that computation is independent and
    # authoritative). Derived by matching the r_hat / (1/r_hat) structure of
    # Jt_hat against freegs's own convention Jtor = R*p' + ff'/(R*mu0).
    def pprime(self, psinorm_freegs):
        psiN_gs = 1.0 - np.clip(psinorm_freegs, 0.0, 1.0)
        dpsiN_dpsi_hat = self._last_dpsiN_dpsi_hat or 0.0
        pprime_hat = self.Pa_hat * self._dp_dpsiN(psiN_gs) * dpsiN_dpsi_hat
        return (self.Jo / self.Lo) * pprime_hat

    def ffprime(self, psinorm_freegs):
        psiN_gs = 1.0 - np.clip(psinorm_freegs, 0.0, 1.0)
        dpsiN_dpsi_hat = self._last_dpsiN_dpsi_hat or 0.0
        alpha_hat = self.alpha_hat or 0.0
        ffprime_over_mu0_hat = (0.5 * self.go_hat ** 2 * alpha_hat
                                 * self._dg_dpsiN(psiN_gs) * dpsiN_dpsi_hat)
        return self.Jo * self.Lo * MU0 * ffprime_over_mu0_hat

    def fvac(self):
        return self.go_phys
