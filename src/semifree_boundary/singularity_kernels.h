/*=============================================================================
  SINGULARITY KERNELS: singular/bounded decomposition of GPsi/GBr/GBz
  =============================================================================
  Port of tools/singularity/kernels.py (derivation in
  docs/reports/GREEN_FUNCTION_SINGULARITY.tex). GPsi/GBr/GBz themselves
  (defined earlier in SemiFree_Solver.cpp) are NOT modified by this header --
  it only subtracts/adds closed-form pieces around them. Requires GPsi/GBr/
  GBz to already be declared/defined before this file is included.

  d = |s-P| = hypot(Rs-R, Zs-Z), s=(Rs,Zs) the source, P=(R,Z) the field
  point (always on the boundary Gamma in this project, see
  docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf section 1).

  greg/gbrBnd/gbzBnd (the "regular" remainders) are computed from the
  convergent power-series closed forms derived in
  docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf and specified for this
  file in docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf, NOT by evaluating
  GPsi/GBr/GBz (which would require the AGM iteration in ellipticKE() and
  amounts to the circular f = g + (f-g) this replaces). Note the derivation
  document writes the SOURCE point as unsubscripted (R,Z) and the FIELD
  point as (R_P,Z_P) -- the opposite of this file's (Rs,Zs)=source,
  (R,Z)=field convention -- every formula below has already been translated
  accordingly (doc-R/Z -> Rs/Zs, doc-R_P/Z_P -> R/Z).
  =============================================================================*/
#ifndef SINGULARITY_KERNELS_H
#define SINGULARITY_KERNELS_H

#include <cmath>
#include <algorithm>

double GPsi(double Rs, double Zs, double R, double Z);
double GBr (double Rs, double Zs, double R, double Z);
double GBz (double Rs, double Zs, double R, double Z);

namespace Singularity {

// d^2 with the same 1e-300 floor used throughout this project's singular
// kernels, so log(d) / 1/d^2 below never see an exact zero.
inline double dist2Floored(double Rs, double Zs, double R, double Z){
    double d2 = (R-Rs)*(R-Rs) + (Z-Zs)*(Z-Zs);
    return std::max(d2, 1e-300);
}

// ---------------------------------------------------------------------------
// analytic limits at s -> P (needed before greg/gbrBnd/gbzBnd below, which
// use gregLimit at the exact-coincidence point d=0)
// ---------------------------------------------------------------------------

// lim_{s->P} greg(s;P): angle-INDEPENDENT (continuous at P).
inline double gregLimit(double Rp){
    return (Rp/(2.0*M_PI)) * (std::log(8.0*Rp) - 2.0);
}

// lim_{s->P} gbrBnd along direction phi (s=P+d*(cos phi,sin phi)):
// angle-DEPENDENT (bounded, not continuous at P).
inline double gbrBndLimit(double Rp, double phi){
    return -std::sin(phi)*std::cos(phi) / (4.0*M_PI*Rp);
}

// lim_{s->P} gbzBnd (FULL split) along direction phi: angle-DEPENDENT.
inline double gbzBndLimit(double Rp, double phi){
    return (std::cos(phi)*std::cos(phi) + std::log(8.0*Rp) - 1.0) / (4.0*M_PI*Rp);
}

// ---------------------------------------------------------------------------
// gamma(s;P) = sqrt((Rs+R)^2 + (Zs-Z)^2), the "unfrozen" companion radius
// used throughout the closed-form series (docs/reports/
// GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf sec 4). NOT floored: gamma is never
// zero for physical Rs,R > 0.
// ---------------------------------------------------------------------------
inline double gammaSq(double Rs, double Zs, double R, double Z){
    return (Rs+R)*(Rs+R) + (Zs-Z)*(Zs-Z);
}

// ---------------------------------------------------------------------------
// Universal power-series coefficients for the closed-form regular terms
// (docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf secs 2-4, 10). These
// do NOT depend on s or P -- computed once (function-local static, C++11
// magic-static init is thread-safe) and reused by every evaluation.
//
// N=30, NOT the briefing's suggested N=10, and NOT a larger N either --
// found empirically while implementing this file, via both accuracy AND
// PERFORMANCE measurements (the briefing's own stated motivation for this
// whole feature is eliminating ellipticKE()'s AGM iteration from the hot
// loop, sec 2 -- an accuracy fix that made the code slower than what it
// replaced would defeat that purpose, not just miss a nice-to-have).
//
// Accuracy: this project's own Soloviev-family D-shape (tall, kappa>1)
// reaches u~0.6-0.8 between far-apart interior/boundary points (gamma is
// dominated by 2*R there, not by d, since both points sit at similar R),
// well past the u<~0.1-0.3 "machine precision" comfort zone the derivation
// document validated N=10 against. Measured relative error at the Soloviev
// worst pair as N grows (test_closedform_greg.cpp test 4, no formula change,
// N alone): 2.3e-4 at N=10, 2.1e-5 at N=15, 6.3e-8 at N=30, 9.5e-11 at N=50,
// 6.8e-15 at N=80 -- confirms the derivation document's own "more terms
// extends the valid range at negligible SERIES-EVALUATION cost" claim
// (sec 5.1), but that claim does not by itself say whether the total is
// still faster than the AGM iteration it replaces.
//
// Performance (measured directly, mixed u from ~0 to ~0.82, 2e6 evaluations
// of greg+gbrBnd+gbzBnd each, -O3, one core): N=10 is 2.6x FASTER than the
// old GPsi/GBr/GBz-based circular subtraction; N=30 is still 1.5x faster;
// N=50 is 0.6x -- i.e. 65% SLOWER, and this actually reproduced end-to-end
// on a full solver run (Google Colab, cases/soloviev_76x131 and
// cases/DIII-D_xpoint, before/after binaries): ~7% slower overall with
// N=50. The plain power series has only LINEAR convergence in u (each term
// gains roughly a factor of u, not the AGM's quadratic convergence), so
// reaching machine precision at u~0.6-0.8 costs far more terms than AGM
// ever needs -- a fixed large N "for safety margin" is not free here.
//
// DECISION (made with the user after presenting this trade-off): N=30,
// accepting ~6.3e-8 relative at the worst pair actually occurring in this
// project's configs -- close to, but not strictly under, the briefing's
// stated <1e-8 target -- in exchange for a real, net 1.5x speedup, which is
// what this feature is actually for. 6.3e-8 remains four orders of
// magnitude tighter than the 5e-3 boundary-fit tolerance this pipeline
// already accepts elsewhere (regularization.h, test_singularity_subtraction.cpp),
// so this is not a meaningful accuracy compromise for the physics -- see
// docs/reports/CLOSEDFORM_GREG_IMPLEMENTATION_REPORT.pdf for the full
// measurement table (all configs, not just Soloviev) and the discarded
// alternative (a hybrid series/AGM dispatch by u, which would satisfy both
// the strict accuracy bar and the speed target but adds a branch not in the
// original spec -- not pursued, since the user opted for the simpler fixed-N
// trade-off instead).
// ---------------------------------------------------------------------------
namespace detail {

struct SeriesCoeffs {
    static const int N = 30;
    double A[N+1], d[N+1], B[N+1], e[N+1], C[N+1], D[N+1];
    SeriesCoeffs(){
        A[0] = 1.0; d[0] = 0.0;
        for(int n = 1; n <= N; ++n){
            double r = (2.0*n-1.0)/(2.0*n);
            A[n] = A[n-1]*r*r;
            d[n] = d[n-1] + 1.0/(n*(2.0*n-1.0));
        }
        B[0] = 0.0; e[0] = 0.0; // unused, kept only so indices align
        B[1] = 0.5; e[1] = 0.5;
        for(int n = 2; n <= N; ++n){
            double s1 = 0.0, s2 = 0.0;
            for(int m = 1; m <= n-1; ++m){
                s1 += A[m]-B[m];
                s2 += A[m]*d[m] - B[m]*e[m];
            }
            B[n] = (1.0+s1)/(2.0*n);
            e[n] = (1.0/n) * ((1.0+s2)/(2.0*B[n]) - 0.5);
        }
        for(int n = 1; n <= N; ++n){
            C[n] = A[n] + A[n-1] - 2.0*B[n];
            D[n] = A[n]*d[n] + A[n-1]*d[n-1] - 2.0*B[n]*e[n];
        }
    }
};

inline const SeriesCoeffs& coeffs(){
    static const SeriesCoeffs c;
    return c;
}

// P(u) = 1 + sum_{n=1}^{N} C_n u^n
inline double seriesP(double u){
    const SeriesCoeffs& c = coeffs();
    double s = 1.0, un = 1.0;
    for(int n = 1; n <= SeriesCoeffs::N; ++n){ un *= u; s += c.C[n]*un; }
    return s;
}

// Q(u) = 2 + sum_{n=1}^{N} D_n u^n
inline double seriesQ(double u){
    const SeriesCoeffs& c = coeffs();
    double s = 2.0, un = 1.0;
    for(int n = 1; n <= SeriesCoeffs::N; ++n){ un *= u; s += c.D[n]*un; }
    return s;
}

// E1(u,L) = sum_{n=1}^{N} B_n u^{n-1} (L-e_n)
inline double seriesE1(double u, double L){
    const SeriesCoeffs& c = coeffs();
    double s = 0.0, un = 1.0;
    for(int n = 1; n <= SeriesCoeffs::N; ++n){ s += c.B[n]*un*(L-c.e[n]); un *= u; }
    return s;
}

// K1(u,L) = sum_{n=1}^{N} A_n u^{n-1} (L-d_n)
inline double seriesK1(double u, double L){
    const SeriesCoeffs& c = coeffs();
    double s = 0.0, un = 1.0;
    for(int n = 1; n <= SeriesCoeffs::N; ++n){ s += c.A[n]*un*(L-c.d[n]); un *= u; }
    return s;
}

} // namespace detail

// ---------------------------------------------------------------------------
// closed-form singular parts (unfrozen: use gamma(s;P), not a value frozen
// at s=P -- required so that subtracting these from GBr/GBz leaves EXACTLY
// the gbrBnd/gbzBnd closed forms below, see GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf
// sec 10 eqs 26-27). Not used by gbrBnd/gbzBnd themselves any more (those
// now evaluate the closed form directly) -- kept for the reconstruction
// tests and as documentation of the convention in force.
// ---------------------------------------------------------------------------

// G^psi_sing(s;P) = -sqrt(Rs*R)/(2*pi) * ln(d)   [log singularity]
inline double gpsiSing(double Rs, double Zs, double R, double Z){
    double d = std::sqrt(dist2Floored(Rs,Zs,R,Z));
    return -std::sqrt(Rs*R)/(2.0*M_PI) * std::log(d);
}

// G^BR_sing(s;P) = (Z-Zs)*gamma/(4*pi*R*d^2)   [unfrozen 1/d dipole, NO log term]
inline double gbrSing(double Rs, double Zs, double R, double Z){
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double gamma = std::sqrt(gammaSq(Rs,Zs,R,Z));
    return (Z-Zs)*gamma / (4.0*M_PI*R*d2);
}

// Dipole-only piece of G^BZ_sing. Exposed SOLELY so a test can demonstrate
// why it must never be used alone in production: subtracting only this
// leaves a residual that diverges logarithmically (see gbzSing below).
inline double gbzSingDipole(double Rs, double Zs, double R, double Z){
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double gamma = std::sqrt(gammaSq(Rs,Zs,R,Z));
    return (Rs-R)*gamma / (4.0*M_PI*R*d2);
}

// FULL singular part of G^BZ: unfrozen dipole + a genuine extra log term.
// This is the form to use in production.
inline double gbzSing(double Rs, double Zs, double R, double Z){
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double d = std::sqrt(d2);
    double gamma = std::sqrt(gammaSq(Rs,Zs,R,Z));
    double L = std::log(4.0*gamma) - std::log(d);
    return (Rs-R)*gamma/(4.0*M_PI*R*d2) + L/(2.0*M_PI*gamma);
}

// ---------------------------------------------------------------------------
// bounded remainders: closed-form convergent power series (NOT a subtraction
// of two large closed-form elliptic terms -- GPsi/GBr/GBz, with their AGM
// iteration, are never called on this path). docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf
// sections 5-6, docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf eqs 15,26,27.
// ---------------------------------------------------------------------------

// greg(s;P) = G^psi_reg(s;P), eq. (15). Bounded everywhere, including at
// d=0 exactly (angle-independent limit gregLimit(R), which is this same
// formula's own u=0 special case -- kept as an explicit branch because
// evaluating -ln(d)*(P(u)/kappa-1) literally at d=0 is (+inf)*0 = NaN in
// IEEE754, even though the algebraic limit is finite).
inline double greg(double Rs, double Zs, double R, double Z){
    double d2raw = (R-Rs)*(R-Rs) + (Z-Zs)*(Z-Zs);
    if(d2raw == 0.0) return gregLimit(R);
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double d = std::sqrt(d2);
    double gamma2 = gammaSq(Rs,Zs,R,Z);
    double gamma = std::sqrt(gamma2);
    double u = d2/gamma2;
    double kappa = std::sqrt(std::max(1.0-u, 1e-300));
    double Pu = detail::seriesP(u);
    double Qu = detail::seriesQ(u);
    double coeff = std::sqrt(Rs*R)/(2.0*M_PI);
    double term1 = -std::log(d) * coeff * (Pu/kappa - 1.0);
    double term2 = (coeff/kappa) * (std::log(4.0*gamma)*Pu - Qu);
    return term1 + term2;
}

// gbrBnd(s;P) = G^BR_reg(s;P), eq. (26). Continuous at s=P (limit 0,
// independent of the approach angle -- see docs/reports/
// GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf sec 10.3), unlike the old
// angle-dependent gbrBndLimit() this replaces at d2raw==0.
inline double gbrBnd(double Rs, double Zs, double R, double Z){
    double d2raw = (R-Rs)*(R-Rs) + (Z-Zs)*(Z-Zs);
    if(d2raw == 0.0) return 0.0;
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double d = std::sqrt(d2);
    double gamma2 = gammaSq(Rs,Zs,R,Z);
    double gamma = std::sqrt(gamma2);
    double u = d2/gamma2;
    double L = std::log(4.0*gamma) - std::log(d);
    double E1 = detail::seriesE1(u,L);
    double K1 = detail::seriesK1(u,L);
    return (Z-Zs)/(2.0*M_PI*R*gamma) * ( -L + 0.5 + E1/2.0 + (u/2.0)*(E1-2.0*K1) );
}

// gbzBnd(s;P) = ln(4*gamma)/(2*pi*gamma) + G^BZ_reg(s;P), eq. (27) (the
// smooth part of the log term is folded in here, not left for a caller's
// patch -- see docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf sec 5.3). Continuous
// at s=P: docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.pdf sec 10.3 (eq. 4)
// proves lim G^BZ_reg = -1/(4*pi*R) -- but that limit is for G^BZ_reg ALONE,
// the boxed bracket term, NOT for the full gbzBnd this function returns.
// gbzBnd also carries the separate ln(4*gamma)/(2*pi*gamma) piece, and
// gamma(P,P) = 2*R is finite and angle-independent as s->P, contributing
// ln(8*R)/(4*pi*R) on top. NUMERIC CHANGE from the old placeholder 0.0 at
// d2raw==0, but to (log(8R)-1)/(4*pi*R), not to the briefing's literal
// "-1/(4*pi*R)" text (sec 6.1 point 4), which omits this second piece --
// verified both algebraically (the L-divergent terms inside G^BZ_reg's own
// bracket vanish because they carry a u or (Rs-R) prefactor -> 0, while
// ln(4*gamma)/(2*pi*gamma) has no such prefactor and survives) and
// numerically (test_closedform_greg.cpp: gbzBnd evaluated at several small
// finite d from several angles converges to this corrected value, not to
// the briefing's -1/(4*pi*R)). Flagged as an unexpected finding, see
// docs/reports/CLOSEDFORM_GREG_IMPLEMENTATION_REPORT.pdf.
inline double gbzBnd(double Rs, double Zs, double R, double Z){
    double d2raw = (R-Rs)*(R-Rs) + (Z-Zs)*(Z-Zs);
    if(d2raw == 0.0) return (std::log(8.0*R) - 1.0) / (4.0*M_PI*R);
    double d2 = dist2Floored(Rs,Zs,R,Z);
    double d = std::sqrt(d2);
    double gamma2 = gammaSq(Rs,Zs,R,Z);
    double gamma = std::sqrt(gamma2);
    double u = d2/gamma2;
    double L = std::log(4.0*gamma) - std::log(d);
    double E1 = detail::seriesE1(u,L);
    double K1 = detail::seriesK1(u,L);
    double reg = ( R*u*K1 + (Rs-R)*E1/2.0 - (Rs+R)/2.0 - (Rs+R)*u*E1/2.0 ) / (2.0*M_PI*R*gamma);
    return std::log(4.0*gamma)/(2.0*M_PI*gamma) + reg;
}

} // namespace Singularity
#endif // SINGULARITY_KERNELS_H
