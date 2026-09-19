/*=============================================================================
  UNIT TESTS: adaptive_quadrature.h (Modulo 1, sub-punto)
  =============================================================================
  Compile:
    g++ -O3 -Isrc/third_party -Isrc/semifree_boundary -o build/test_adaptive_quadrature \
        src/semifree_boundary/tests/test_adaptive_quadrature.cpp -lm -std=c++11
  Run (from src/semifree_boundary/tests/fixtures): ../../../../build/test_adaptive_quadrature
  =============================================================================*/
#include "adaptive_quadrature.h"
#include <cstdio>
#include <cmath>
#include <string>

using namespace std;

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

    // test_polynomial_exact: Gauss-7 integrates polynomials up to degree 13
    // exactly, so x^3 on [0,2] should match the closed form (8) to machine
    // precision even with a very loose tolerance.
    {
        auto f = [](double x){ return x*x*x; };
        auto r = Quadrature::integrate(f, 0.0, 2.0, {}, 1e-11, 1e-6);
        checkClose(r.value, 4.0, 1e-13, "test_polynomial_exact (x^3 on [0,2] = 4)");
    }

    // test_smooth_transcendental
    {
        auto f = [](double x){ return sin(x); };
        auto r = Quadrature::integrate(f, 0.0, M_PI, {}, 1e-12, 1e-10);
        checkClose(r.value, 2.0, 1e-10, "test_smooth_transcendental (sin(x) on [0,pi] = 2)");
    }

    // test_log_singularity_known_closed_form: integral(-ln(x),0,1) = 1.
    // No breakpoint at x=0 (it IS the lower bound) -- the adaptive search
    // must refine on its own toward the singular endpoint.
    {
        auto f = [](double x){ return -log(x); };
        auto r = Quadrature::integrate(f, 0.0, 1.0, {}, 1e-9, 1e-7, 60, 20000);
        checkClose(r.value, 1.0, 1e-5, "test_log_singularity_known_closed_form (-ln x on [0,1] = 1)");
        check(r.nPanels > 1, "test_log_singularity_known_closed_form: subdivided (nPanels>1)");
    }

    // test_breakpoint_resolves_jump_discontinuity_with_fewer_evals: a known
    // breakpoint's real payoff (per the briefing, sec 6.7) is isolating a
    // KINK/JUMP so each side becomes smooth again for the fixed-order
    // rule -- not an unbounded point singularity, where both the "with" and
    // "without breakpoint" paths must drill down to comparable depth near
    // the point regardless (verified above: for -ln|x-c|, passing the exact
    // breakpoint costs MORE evals, since it pins a panel edge exactly on the
    // singularity and both flanking panels get driven all the way to the
    // relative-width floor from scratch). A bounded jump discontinuity is
    // the right shape to demonstrate the win: without knowing where it is,
    // adaptive refinement needs many levels to isolate it (bisecting until
    // the panel width times the jump height clears the tolerance); knowing
    // it exactly makes both sides plain-smooth immediately.
    {
        const double c = 0.37;
        auto f = [c](double x){ return sin(3.0*x) + (x < c ? 0.0 : 5.0); };
        // integral(sin(3x),-1,1)=0 exactly by odd symmetry; the jump adds
        // 5*(length of [c,1]).
        double expected = 5.0*(1.0-c);

        auto withBreak = Quadrature::integrate(f, -1.0, 1.0, {c}, 1e-10, 1e-9, 60, 20000);
        auto noBreak    = Quadrature::integrate(f, -1.0, 1.0, {},  1e-10, 1e-9, 60, 20000);
        checkClose(withBreak.value, expected, 1e-7, "test_breakpoint_resolves_jump: value (with break)");
        checkClose(noBreak.value, expected, 1e-7, "test_breakpoint_resolves_jump: value (no break, sanity)");
        printf("%s test_breakpoint_resolves_jump_discontinuity_with_fewer_evals: nEvals with break=%d, without=%d\n",
               (withBreak.nEvals < noBreak.nEvals) ? "[PASS]" : "[FAIL]", withBreak.nEvals, noBreak.nEvals);
        if(withBreak.nEvals >= noBreak.nEvals) g_failures++;
    }

    // test_dipole_like_singularity: integral(1/sqrt(|x|), -1, 1) = 4.
    {
        auto f = [](double x){ return 1.0/sqrt(fabs(x)); };
        auto r = Quadrature::integrate(f, -1.0, 1.0, {0.0}, 1e-9, 1e-7, 60, 20000);
        checkClose(r.value, 4.0, 1e-4, "test_dipole_like_singularity (1/sqrt|x| on [-1,1] = 4)");
    }

    // test_tight_tolerance_forces_subdivision: same oscillatory function,
    // tight vs loose tolerance should use more panels when tight.
    {
        auto f = [](double x){ return sin(50.0*x); };
        auto loose = Quadrature::integrate(f, 0.0, 3.0, {}, 1e-11, 1e-3);
        auto tight = Quadrature::integrate(f, 0.0, 3.0, {}, 1e-13, 1e-10);
        printf("%s test_tight_tolerance_forces_subdivision: nPanels loose=%d tight=%d\n",
               (tight.nPanels > loose.nPanels) ? "[PASS]" : "[FAIL]", loose.nPanels, tight.nPanels);
        if(tight.nPanels <= loose.nPanels) g_failures++;
    }

    // test_max_panels_respected: sin(1/x) near x=0 without a breakpoint is
    // pathological (infinitely many oscillations); confirm the routine
    // terminates and honors maxPanels instead of recursing forever.
    {
        auto f = [](double x){ return sin(1.0/x); };
        int maxPanels = 200;
        auto r = Quadrature::integrate(f, 1e-6, 1.0, {}, 1e-14, 1e-14, 60, maxPanels);
        check(r.nPanels <= maxPanels, "test_max_panels_respected (terminates, nPanels<=maxPanels)");
    }

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
