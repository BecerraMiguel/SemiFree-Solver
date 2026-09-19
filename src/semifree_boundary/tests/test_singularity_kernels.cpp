/*=============================================================================
  UNIT TESTS: Paso 0 (GBr/GBz numerical floor fix) + singularity_kernels.h
  (Modulo 2, sub-punto)
  =============================================================================
  Reuses SemiFree_Solver.cpp's own GPsi/GBr/GBz via the same
  #define UNIT_TESTING + #include "SemiFree_Solver.cpp" trick already
  established by test_regularization.cpp (keeps main() out of this binary).

  Compile:
    g++ -O3 -Isrc/third_party -Isrc/semifree_boundary -o build/test_singularity_kernels \
        src/semifree_boundary/tests/test_singularity_kernels.cpp -lm -std=c++11
  Run (from src/semifree_boundary/tests/fixtures): ../../../../build/test_singularity_kernels
  =============================================================================*/
#define UNIT_TESTING
#include "SemiFree_Solver.cpp"
#include "singularity_kernels.h"
#include <cstdio>
#include <cmath>
#include <string>

static int g_failures = 0;

static void check(bool cond, const string& label){
    if(cond) printf("[PASS] %s\n", label.c_str());
    else { printf("[FAIL] %s\n", label.c_str()); g_failures++; }
}

static void checkClose(double actual, double expected, double relTol, const string& label){
    double denom = fabs(expected) > 0 ? fabs(expected) : 1.0;
    double relErr = fabs(actual-expected)/denom;
    bool ok = relErr <= relTol;
    printf("%s %s: actual=%.15e expected=%.15e relErr=%.3e tol=%.3e\n",
           ok ? "[PASS]" : "[FAIL]", label.c_str(), actual, expected, relErr, relTol);
    if(!ok) g_failures++;
}

int main(){

    // ---- Paso 0: GBr/GBz numerical floor fix -------------------------------

    // test_p_identity_vs_naive_cancellation: demonstrate the cancellation the
    // fix avoids. p_naive = 1-4*R*Rs/gamma2 loses essentially all precision
    // vs the exact identity p = d2/gamma2 once d is tiny.
    {
        double R = 1.15, Z = 0.30;
        double Rs = R + 1e-10, Zs = Z; // d ~ 1e-10 along R
        double gamma2 = (R+Rs)*(R+Rs) + (Z-Zs)*(Z-Zs);
        double d2 = (R-Rs)*(R-Rs) + (Z-Zs)*(Z-Zs);
        double p_exact = d2/gamma2;
        double p_naive = 1.0 - 4.0*R*Rs/gamma2;
        double relErr = fabs(p_naive - p_exact) / p_exact;
        printf("%s test_p_identity_vs_naive_cancellation: p_exact=%.6e p_naive=%.6e relErr=%.3e\n",
               (relErr > 1e-6) ? "[PASS]" : "[FAIL]", p_exact, p_naive, relErr);
        // relErr here is bounded above by ~1.0 (p_naive can only go so far
        // wrong relative to the tiny p_exact before hitting its own floor of
        // 0); a relErr orders of magnitude above double precision (~1e-16)
        // already demonstrates catastrophic loss of significant digits --
        // in this engineered case p_naive underflows to exactly 0 (relErr=1,
        // i.e. every digit lost).
        if(!(relErr > 1e-6)) g_failures++;
    }

    // test_GBr_GBz_no_cancellation_near_d0: evaluate at d~1e-8 (should be
    // finite and close to the angle-dependent bounded limit once the
    // dipole/log singular part is subtracted) and at a point engineered so
    // R+dR rounds EXACTLY back to R in double precision (d2 underflows to
    // exactly 0.0) -- confirm no NaN/Inf either way.
    {
        double R = 1.15, Z = 0.30;
        double dR = 1e-8;
        double Rs1 = R + dR, Zs1 = Z;
        double gbr1 = GBr(Rs1, Zs1, R, Z);
        double gbz1 = GBz(Rs1, Zs1, R, Z);
        check(std::isfinite(gbr1) && std::isfinite(gbz1),
              "test_GBr_GBz_no_cancellation_near_d0: finite at d~1e-8");

        // Engineer exact-underflow: an offset far below R's ULP.
        double tinyOffset = R * 1e-18;
        double Rs2 = R + tinyOffset; // rounds back to exactly R
        check(Rs2 == R, "test_GBr_GBz_no_cancellation_near_d0: offset genuinely underflows (Rs2==R)");
        double gbr2 = GBr(Rs2, Z, R, Z);
        double gbz2 = GBz(Rs2, Z, R, Z);
        check(std::isfinite(gbr2) && std::isfinite(gbz2),
              "test_GBr_GBz_no_cancellation_near_d0: finite at exact d2=0 underflow (no NaN/Inf)");
    }

    // ---- Modulo 2: singularity_kernels.h -----------------------------------

    // test_greg_limit_matches_closed_form: for several Rp, greg(s;P) should
    // converge to gregLimit(Rp) as d->0, independent of the approach angle.
    {
        double Rps[] = {0.5, 1.15, 2.0};
        for(double Rp : Rps){
            double Zp = 0.2;
            double limitVal = Singularity::gregLimit(Rp);
            double phis[] = {0.0, M_PI/4.0, 1.3, 2.7};
            double vals[4];
            int n = 4;
            for(int i = 0; i < n; ++i){
                double d = 1e-7;
                double Rs = Rp + d*cos(phis[i]);
                double Zs = Zp + d*sin(phis[i]);
                vals[i] = Singularity::greg(Rs, Zs, Rp, Zp);
            }
            char label[128];
            for(int i = 0; i < n; ++i){
                snprintf(label, sizeof(label), "test_greg_limit_matches_closed_form Rp=%.2f phi=%.2f", Rp, phis[i]);
                checkClose(vals[i], limitVal, 1e-6, label);
            }
            double maxSpread = 0.0;
            for(int i = 1; i < n; ++i) maxSpread = max(maxSpread, fabs(vals[i]-vals[0]));
            snprintf(label, sizeof(label), "test_greg_limit_matches_closed_form Rp=%.2f angle-independent", Rp);
            check(maxSpread < 1e-6, label);
        }
    }

    // test_gbr_bnd_limit_is_now_angle_independent
    //
    // BEHAVIOR CHANGE (docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf): gbrBnd()
    // used to be GBr()-gbrSing() with gbrSing FROZEN at s=P's own
    // coefficient (1/(2*pi*d^2), no gamma factor) -- that frozen convention
    // is what made gbrBndLimit(Rp,phi) angle-DEPENDENT. gbrSing is now the
    // UNFROZEN dipole (gamma(s,P)/(4*pi*Rp*d^2)), and gbrBnd is now the
    // independently-derived closed form G^BR_reg -- a genuinely different
    // decomposition of the same GBr(), not merely a faster way to compute
    // the old one. Its proven limit is angle-INDEPENDENT (0 for every
    // approach direction, docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf
    // sec 10.3 eq. 4) -- gbrBndLimit() (still defined in singularity_kernels.h)
    // now describes a decomposition that is no longer in force; it is kept
    // only as a labeled historical artifact, not as a correctness oracle.
    // See test_closedform_greg.cpp (test 3) for the full angular table.
    {
        double Rp = 1.15, Zp = 0.30;
        double phiA = 0.4, phiB = M_PI/4.0 + 0.6;
        double d = 1e-7;
        double valA = Singularity::gbrBnd(Rp+d*cos(phiA), Zp+d*sin(phiA), Rp, Zp);
        double valB = Singularity::gbrBnd(Rp+d*cos(phiB), Zp+d*sin(phiB), Rp, Zp);
        checkClose(valA, 0.0, 1e-5, "test_gbr_bnd_limit phiA -> 0 (new angle-independent limit)");
        checkClose(valB, 0.0, 1e-5, "test_gbr_bnd_limit phiB -> 0 (new angle-independent limit)");
        check(fabs(valA - valB) < 1e-5, "test_gbr_bnd_limit is now angle-INDEPENDENT (phiA ~= phiB, unlike the old frozen decomposition)");
    }

    // test_gbz_bnd_limit_is_now_angle_independent (same behavior change as
    // above). gbzBnd's own exact limit is (ln(8*Rp)-1)/(4*pi*Rp), NOT the
    // naive -1/(4*pi*Rp) the briefing's prose suggests (that value is the
    // limit of G^BZ_reg alone -- see the comment above gbzBnd() in
    // singularity_kernels.h -- gbzBnd additionally carries the
    // ln(4*gamma)/(2*pi*gamma) term, whose own limit ln(8*Rp)/(4*pi*Rp) is
    // not zero and must be added).
    {
        double Rp = 1.15, Zp = 0.30;
        double phi = 0.9;
        double d = 1e-7;
        double val = Singularity::gbzBnd(Rp+d*cos(phi), Zp+d*sin(phi), Rp, Zp);
        double corrected = (log(8.0*Rp)-1.0)/(4.0*M_PI*Rp);
        checkClose(val, corrected, 1e-5, "test_gbz_bnd_limit -> (ln(8Rp)-1)/(4*pi*Rp) (new angle-independent limit)");
    }

    // test_gbz_sing_full_vs_dipole_only: the FULL split (gbzSing, now the
    // UNFROZEN dipole+log of eq. 3) converges to G^BZ_reg's own proven
    // limit -1/(4*pi*Rp) (angle-independent); the dipole-ONLY split still
    // diverges logarithmically as d->0.
    {
        double Rp = 1.15, Zp = 0.30, phi = 0.5;
        double ds[] = {1e-3, 1e-5, 1e-7};
        double fullVals[3], dipoleOnlyVals[3];
        for(int i = 0; i < 3; ++i){
            double Rs = Rp + ds[i]*cos(phi), Zs = Zp + ds[i]*sin(phi);
            fullVals[i] = GBz(Rs,Zs,Rp,Zp) - Singularity::gbzSing(Rs,Zs,Rp,Zp);
            dipoleOnlyVals[i] = GBz(Rs,Zs,Rp,Zp) - Singularity::gbzSingDipole(Rs,Zs,Rp,Zp);
        }
        checkClose(fullVals[2], -1.0/(4.0*M_PI*Rp), 1e-4,
                   "test_gbz_sing_full_vs_dipole_only: full split converges to G^BZ_reg's angle-independent limit");
        // The missing term is now the full L/(2*pi*gamma) log (not just its
        // frozen -sqrt(Rs*R)/(4*pi*R^2)*ln(d) piece), but since
        // gamma(P,P)=2*Rp, its growth rate as s->P is still, to leading
        // order, -ln(d)/(4*pi*Rp) -- numerically the same predicted rate as
        // before the change (a coincidence of gamma(P,P)=2*Rp, not a sign
        // that nothing changed: gbzSing's own VALUE at finite d did change,
        // only this leading-order growth RATE happens to match).
        double predictedGrowth = log(ds[0]/ds[2]) / (4.0*M_PI*Rp);
        double observedGrowth = fabs(dipoleOnlyVals[2] - dipoleOnlyVals[0]);
        checkClose(observedGrowth, predictedGrowth, 1e-2,
                   "test_gbz_sing_full_vs_dipole_only: dipole-only split diverges at the predicted -ln(d)/(4*pi*Rp) rate");
    }

    // ---- Cross-check against tools/singularity/kernels.py -----------------
    // Reference values frozen from a one-time run of:
    //   python3 -c "import sys; sys.path.insert(0,'tools/singularity');
    //                import kernels as K; print(K.gpsi(Rs,Zs,R,Z), ...)"
    // at the three (Rs,Zs,R,Z) points below (the third is near the single
    // X-point scenario, R_X=0.56, Z_X=-1.24).
    //
    // gbrBnd/gbzBnd removed from this cross-check (docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf):
    // gbrSing/gbzSing are now the UNFROZEN dipole(+log), so gbrBnd/gbzBnd
    // are a genuinely different decomposition of GBr/GBz at every finite d,
    // not merely a faster way to compute the frozen values these were
    // originally recorded from -- those old numbers are stale by
    // construction, not a regression (confirmed: they fail by 10-140%
    // relative after this change, while gpsi/gbr/gbz/greg below, whose
    // decomposition did NOT change, still match to 1e-9 or better). gbrBnd/
    // gbzBnd's correctness is instead validated directly in
    // test_closedform_greg.cpp (test 2, reconstruction against GPsi/GBr/GBz
    // to ~1e-9, a stronger check than a frozen external snapshot).
    {
        struct Case { double Rs,Zs,R,Z, gpsi,gbr,gbz,greg; };
        Case cases[] = {
            {1.2, 0.3, 0.8, -0.5,
             0.06973444001587126, -0.11949719763989586, 0.18891437860728214,
             0.05233600869831005},
            {0.9, -0.2, 1.1, 0.4,
             0.11017919438927708, 0.1627271310421309, 0.04059746753974696,
             0.03762859128114912},
            {0.56, -1.24, 0.60, -1.20,
             0.22267495427130657, 1.8974839448841978, -1.4617142454039875,
             -0.04230920121923992},
        };
        for(const Case& c : cases){
            char label[160];
            snprintf(label, sizeof(label), "cross-check GPsi (Rs=%.2f,Zs=%.2f,R=%.2f,Z=%.2f)", c.Rs,c.Zs,c.R,c.Z);
            checkClose(GPsi(c.Rs,c.Zs,c.R,c.Z), c.gpsi, 1e-9, label);
            snprintf(label, sizeof(label), "cross-check GBr  (Rs=%.2f,Zs=%.2f,R=%.2f,Z=%.2f)", c.Rs,c.Zs,c.R,c.Z);
            checkClose(GBr(c.Rs,c.Zs,c.R,c.Z), c.gbr, 1e-9, label);
            snprintf(label, sizeof(label), "cross-check GBz  (Rs=%.2f,Zs=%.2f,R=%.2f,Z=%.2f)", c.Rs,c.Zs,c.R,c.Z);
            checkClose(GBz(c.Rs,c.Zs,c.R,c.Z), c.gbz, 1e-9, label);
            snprintf(label, sizeof(label), "cross-check greg (Rs=%.2f,Zs=%.2f,R=%.2f,Z=%.2f)", c.Rs,c.Zs,c.R,c.Z);
            checkClose(Singularity::greg(c.Rs,c.Zs,c.R,c.Z), c.greg, 1e-9, label);
        }
    }

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
