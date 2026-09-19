/*=============================================================================
  SINGULARITY SUBTRACTION WITH ANALYTIC PATCH
  =============================================================================
  Port of tools/singularity/{polygon_geom_constants,xpoint_subtraction}.py.
  Combines cutcell_geometry.h (Module A), grid_jphi_reconstruction.h
  (Module B), singularity_kernels.h and adaptive_quadrature.h (Module C
  infrastructure) into the final psi_p/B_R/B_Z evaluation at a point P
  ALWAYS on the boundary Gamma (see docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf
  section 1).

  Requires `struct Point{double R,Z;};` to already be visible, and must be
  included after cutcell_geometry.h, grid_jphi_reconstruction.h,
  singularity_kernels.h, adaptive_quadrature.h.
  =============================================================================*/
#ifndef SINGULARITY_SUBTRACTION_H
#define SINGULARITY_SUBTRACTION_H

#include "cutcell_geometry.h"
#include "grid_jphi_reconstruction.h"
#include "singularity_kernels.h"
#include "adaptive_quadrature.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

namespace Singularity {

struct GeomConstants {
    double lnD=0, dipZ=0, dipR=0, mlnR=0, mlnZ=0, dZ_R=0, dZ_Z=0, dR_R=0, dR_Z=0;
};

// ---------------------------------------------------------------------------
// Star-shaped audit: DIAGNOSTIC ONLY. Never used to select the integration
// method -- the polar/ray reduction this would enable was implemented,
// tested, and discarded (see docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf
// section 6.6): it fails near an X-point cusp (the two separatrix branches
// run nearly parallel, so a ray can cross that double branch twice), and even
// where it DOES pass, a fixed-order angular quadrature over L(alpha) loses
// its convergence at each polygon vertex's genuine curvature kink. Row-scan
// (below) is the only production integration route, unconditionally.
// ---------------------------------------------------------------------------
struct StarShapedReport { int nProbe=0, nViolations=0; double violationFraction=0.0; };

inline StarShapedReport checkStarShaped(const std::vector<Point>& boundary,
                                         double RP, double ZP,
                                         int nProbe = 2000, double hi0 = 5.0){
    const int nSteps = 2000;
    int nBad = 0;
    for(int a = 0; a < nProbe; ++a){
        double alpha = 2.0*M_PI*a/(double)nProbe;
        double cs = std::cos(alpha), sn = std::sin(alpha);
        double dr = hi0/(double)nSteps;
        bool prevInside = CutCell::insidePolygon(RP+dr*cs, ZP+dr*sn, boundary);
        int transitions = 0;
        for(int s = 2; s <= nSteps; ++s){
            double r = dr*s;
            bool inside = CutCell::insidePolygon(RP+r*cs, ZP+r*sn, boundary);
            if(inside != prevInside) transitions++;
            prevInside = inside;
        }
        if(transitions != 1) nBad++;
    }
    StarShapedReport rep;
    rep.nProbe = nProbe;
    rep.nViolations = nBad;
    rep.violationFraction = (double)nBad/(double)nProbe;
    return rep;
}

// ---------------------------------------------------------------------------
// Row crossings / intervals: handles an arbitrary EVEN number of crossings
// per row (never assumes exactly 2). An odd count (degenerate tangency)
// silently drops the last crossing, mirroring polygon_geom_constants.py's
// _row_intervals.
// ---------------------------------------------------------------------------
inline std::vector<std::pair<double,double>> rowIntervals(const std::vector<Point>& boundary, double z){
    std::vector<double> crossings;
    int n = (int)boundary.size();
    for(int e = 0; e < n; ++e){
        const Point& a = boundary[e];
        const Point& b = boundary[(e+1)%n];
        bool straddle = (a.Z <= z && b.Z > z) || (b.Z <= z && a.Z > z);
        if(straddle){
            double t = (z - a.Z) / (b.Z - a.Z);
            crossings.push_back(a.R + t*(b.R-a.R));
        }
    }
    std::sort(crossings.begin(), crossings.end());
    std::vector<std::pair<double,double>> intervals;
    int nPairs = (int)crossings.size()/2;
    for(int i = 0; i < nPairs; ++i) intervals.push_back({crossings[2*i], crossings[2*i+1]});
    return intervals;
}

// Whether EVERY row (scanned at nScan heights strictly inside [zmin,zmax])
// crosses the boundary exactly twice -- checked explicitly, never assumed
// from the polygon's shape.
inline bool verifyVerticallySimple(const std::vector<Point>& boundary, int nScan = 4000){
    double zmin = 1e300, zmax = -1e300;
    for(const Point& p : boundary){ zmin = std::min(zmin,p.Z); zmax = std::max(zmax,p.Z); }
    double margin = 1e-9*(zmax-zmin);
    int n = (int)boundary.size();
    for(int s = 0; s < nScan; ++s){
        double z = zmin + margin + (zmax-zmin-2*margin)*s/(double)(nScan-1);
        int count = 0;
        for(int e = 0; e < n; ++e){
            const Point& a = boundary[e];
            const Point& b = boundary[(e+1)%n];
            bool straddle = (a.Z <= z && b.Z > z) || (b.Z <= z && a.Z > z);
            if(straddle) count++;
        }
        if(count != 2) return false;
    }
    return true;
}

namespace detail {
inline double interp1(const std::vector<double>& xs, const std::vector<double>& ys, double x){
    int n = (int)xs.size();
    if(n == 1) return ys[0];
    if(x <= xs[0]) return ys[0];
    if(x >= xs[n-1]) return ys[n-1];
    int lo = 0, hi = n-1;
    while(hi-lo > 1){
        int mid = (lo+hi)/2;
        if(xs[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x-xs[lo]) / (xs[hi]-xs[lo]);
    return ys[lo] + t*(ys[hi]-ys[lo]);
}
} // namespace detail

// Fast path for a VERTICALLY SIMPLE polygon: split the vertex ring into two
// Z-monotonic branches (from the min-Z vertex to the max-Z vertex, and back)
// and use exact linear interpolation (the edges already ARE straight
// segments) instead of re-scanning every edge on each row query. Measured
// 40-70x faster in the Python prototype with no precision loss, for the
// polygons this project actually uses. NEVER built without first confirming
// verifyVerticallySimple().
struct VSimpleExtent {
    std::vector<double> zA, rA, zB, rB, zBreaks;
    double zmin = 0.0, zmax = 0.0;
    void extent(double z, double& rLeft, double& rRight) const {
        double ra = detail::interp1(zA, rA, z);
        double rb = detail::interp1(zB, rB, z);
        rLeft = std::min(ra, rb);
        rRight = std::max(ra, rb);
    }
};

inline VSimpleExtent buildVSimpleExtent(const std::vector<Point>& boundary){
    int n = (int)boundary.size();
    int iMin = 0, iMax = 0;
    for(int i = 1; i < n; ++i){
        if(boundary[i].Z < boundary[iMin].Z) iMin = i;
        if(boundary[i].Z > boundary[iMax].Z) iMax = i;
    }
    VSimpleExtent ve;
    int i = iMin;
    while(true){
        ve.zA.push_back(boundary[i].Z);
        ve.rA.push_back(boundary[i].R);
        if(i == iMax) break;
        i = (i+1)%n;
    }
    std::vector<double> zBtmp, rBtmp;
    i = iMax;
    while(true){
        zBtmp.push_back(boundary[i].Z);
        rBtmp.push_back(boundary[i].R);
        if(i == iMin) break;
        i = (i+1)%n;
    }
    ve.zB.assign(zBtmp.rbegin(), zBtmp.rend());
    ve.rB.assign(rBtmp.rbegin(), rBtmp.rend());
    ve.zmin = boundary[iMin].Z;
    ve.zmax = boundary[iMax].Z;

    std::vector<double> allZ;
    for(const Point& p : boundary) allZ.push_back(p.Z);
    std::sort(allZ.begin(), allZ.end());
    allZ.erase(std::unique(allZ.begin(), allZ.end()), allZ.end());
    ve.zBreaks = allZ;
    return ve;
}

// ---------------------------------------------------------------------------
// Row-scan nested quadrature: the ONLY production integration route (see the
// star-shaped audit note above). Sweeps Z (adaptive, with panel breaks at
// every polygon-vertex height and at Z=ZP), and for each row integrates in R
// over every "inside" interval (adaptive, with a panel break at R=RP when
// the row passes within rBreakWindow of ZP).
// ---------------------------------------------------------------------------
inline double nestedQuadRowScan(const std::function<double(double,double)>& integrandRZ,
                                 double RP, double ZP, const std::vector<Point>& boundary,
                                 const VSimpleExtent* vsimple,
                                 double epsrel = 1e-8, double rBreakWindow = 0.15,
                                 const std::vector<double>& extraZBreaks = {}){
    double zmin = 1e300, zmax = -1e300;
    for(const Point& p : boundary){ zmin = std::min(zmin,p.Z); zmax = std::max(zmax,p.Z); }

    auto inner = [&](double z) -> double {
        std::vector<std::pair<double,double>> intervals;
        if(vsimple){
            double rl, rr;
            vsimple->extent(z, rl, rr);
            intervals.push_back({rl,rr});
        } else {
            intervals = rowIntervals(boundary, z);
        }
        double val = 0.0;
        for(const auto& iv : intervals){
            double a = iv.first, b = iv.second;
            if(b-a < 1e-13) continue;
            std::vector<double> rBreaks;
            if(a < RP && RP < b && std::fabs(z-ZP) < rBreakWindow) rBreaks.push_back(RP);
            auto f1 = [&](double r){ return integrandRZ(r,z); };
            auto res = Quadrature::integrate(f1, a, b, rBreaks, 1e-13, epsrel);
            val += res.value;
        }
        return val;
    };

    std::vector<double> zBreaks;
    if(vsimple) zBreaks = vsimple->zBreaks;
    if(ZP > zmin && ZP < zmax) zBreaks.push_back(ZP);
    for(double zb : extraZBreaks) if(zb > zmin && zb < zmax) zBreaks.push_back(zb);

    auto res = Quadrature::integrate(inner, zmin, zmax, zBreaks, 1e-12, epsrel);
    return res.value;
}

// The 9 exact geometric constants (integrals of the pure singular kernel,
// no Jphi, over the whole domain D) needed for the analytic patch below.
inline GeomConstants geomConstantsRowScan(double RP, double ZP,
                                           const std::vector<Point>& boundary,
                                           const VSimpleExtent* vsimple,
                                           double epsrel = 1e-8){
    auto d2f = [&](double R, double Z){
        double d2 = (R-RP)*(R-RP) + (Z-ZP)*(Z-ZP);
        return std::max(d2, 1e-300);
    };
    auto mln = [&](double R, double Z){ return -0.5*std::log(d2f(R,Z)); };

    GeomConstants c;
    c.lnD  = nestedQuadRowScan([&](double R,double Z){ return mln(R,Z); },                    RP,ZP,boundary,vsimple,epsrel);
    c.dipZ = nestedQuadRowScan([&](double R,double Z){ return (ZP-Z)/d2f(R,Z); },              RP,ZP,boundary,vsimple,epsrel);
    c.dipR = nestedQuadRowScan([&](double R,double Z){ return (R-RP)/d2f(R,Z); },              RP,ZP,boundary,vsimple,epsrel);
    c.mlnR = nestedQuadRowScan([&](double R,double Z){ return (R-RP)*mln(R,Z); },              RP,ZP,boundary,vsimple,epsrel);
    c.mlnZ = nestedQuadRowScan([&](double R,double Z){ return (Z-ZP)*mln(R,Z); },              RP,ZP,boundary,vsimple,epsrel);
    c.dZ_R = nestedQuadRowScan([&](double R,double Z){ return (R-RP)*(ZP-Z)/d2f(R,Z); },       RP,ZP,boundary,vsimple,epsrel);
    c.dZ_Z = nestedQuadRowScan([&](double R,double Z){ return (Z-ZP)*(ZP-Z)/d2f(R,Z); },       RP,ZP,boundary,vsimple,epsrel);
    c.dR_R = nestedQuadRowScan([&](double R,double Z){ return (R-RP)*(R-RP)/d2f(R,Z); },       RP,ZP,boundary,vsimple,epsrel);
    c.dR_Z = nestedQuadRowScan([&](double R,double Z){ return (ZP-Z)*(R-RP)/d2f(R,Z); },       RP,ZP,boundary,vsimple,epsrel);
    return c;
}

// ---------------------------------------------------------------------------
// Cached per-mesh context: built ONCE per SemiFree_Solver.cpp run (see the
// design decisions for this feature -- a local object in main(), right
// after `boundary` is finalized), reused by every subsequent evaluation.
// ---------------------------------------------------------------------------
struct PlasmaContext {
    CutCell::Geometry geom;
    JphiGrid::Field jfield;
    std::vector<Point> boundary;
    VSimpleExtent vsimpleExtent;
    bool useVSimple = false;
};

inline PlasmaContext build(const std::vector<Point>& boundary,
                            const std::vector<std::vector<double>>& JphiMasked,
                            const std::vector<double>& R, const std::vector<double>& Z){
    PlasmaContext ctx;
    ctx.boundary = boundary;
    ctx.geom = CutCell::build(boundary, R, Z);
    ctx.jfield = JphiGrid::build(ctx.geom, JphiMasked, R, Z);
    ctx.useVSimple = verifyVerticallySimple(boundary);
    if(ctx.useVSimple) ctx.vsimpleExtent = buildVSimpleExtent(boundary);
    return ctx;
}

// ---------------------------------------------------------------------------
// Cut-cell quadrature of an arbitrary (r,z)->double integrand over the whole
// domain (Q_h in the briefing's notation): reuses cutcell_geometry.h's
// integrate(), sampling at nodes for Interior cells and at the ghost-filled/
// bilinear-interpolated value for Cut cells' centroids. Needs the mesh R/Z
// arrays to evaluate g at each interior node's true (R,Z), not just its
// index into ctx.geom.
inline double cutCellIntegral(const PlasmaContext& ctx,
                               const std::vector<double>& R, const std::vector<double>& Z,
                               const std::function<double(double,double)>& g){
    int npr = ctx.geom.npr, npz = ctx.geom.npz;
    std::vector<std::vector<double>> nodesVal(npr, std::vector<double>(npz, 0.0));
    for(int i = 0; i < npr; ++i)
        for(int k = 0; k < npz; ++k)
            if(ctx.geom.cell[i][k].state == CutCell::CellState::Interior)
                nodesVal[i][k] = g(R[i], Z[k]);
    return CutCell::integrate(ctx.geom, nodesVal, g);
}

// weakSingIntegral(m,w,mP,gradM,Iw,Isw) = mP*Iw + gradM.Isw
//   + Q_h[ (m(s)-mP-gradM.(s-P)) * w(s) ]
inline double weakSingIntegral(const PlasmaContext& ctx,
                                const std::vector<double>& R, const std::vector<double>& Z,
                                double RP, double ZP,
                                const std::function<double(double,double)>& m,
                                const std::function<double(double,double)>& w,
                                double mP, double gradMR, double gradMZ,
                                double Iw, double IswR, double IswZ){
    double analytic = mP*Iw + gradMR*IswR + gradMZ*IswZ;
    auto residualTimesW = [&](double r, double z){
        double rem = m(r,z) - mP - (gradMR*(r-RP) + gradMZ*(z-ZP));
        return rem * w(r,z);
    };
    double residual = cutCellIntegral(ctx, R, Z, residualTimesW);
    return analytic + residual;
}

struct Result { double psi=0.0, br=0.0, bz=0.0; };

// Low-level assembly: takes Jphi(P), grad(Jphi)(P) and the Jphi(r,z)
// evaluator as explicit parameters, rather than reading them from
// ctx.jfield. This mirrors xpoint_subtraction.subtraction_poly(mesh, geom,
// P, Jfun, dJfun, C, order)'s own signature, and is what lets a test isolate
// Module C (this file) from Module B (grid_jphi_reconstruction.h): pass an
// ANALYTIC Jfun/gradient (e.g. the Soloviev closed form) instead of the
// grid reconstruction, exactly as verify_xpoint_endtoend.py's "Track A" does.
// order=2: production (value + gradient Taylor term). order=1: intermediate
// check (value only, all gradient terms multiplied by 0 -- xpoint_subtraction
// .py's g1[0] switch).
inline Result evaluateAllWithJfun(const PlasmaContext& ctx,
                                   const std::vector<double>& R, const std::vector<double>& Z,
                                   double RP, double ZP,
                                   const std::function<double(double,double)>& Jfun,
                                   double JP, double dJR, double dJZ,
                                   int order = 2){
    double g = (order == 2) ? 1.0 : 0.0;

    const VSimpleExtent* vs = ctx.useVSimple ? &ctx.vsimpleExtent : nullptr;
    GeomConstants C = geomConstantsRowScan(RP, ZP, ctx.boundary, vs);

    // ---- psi_p --------------------------------------------------------
    auto gregK = [&](double r, double z){ return Singularity::greg(r,z,RP,ZP); };
    double psiBnd = cutCellIntegral(ctx, R, Z, [&](double r,double z){ return Jfun(r,z)*gregK(r,z); });

    auto mC = [&](double r,double z){ return Jfun(r,z)*std::sqrt(r*RP)/(2.0*M_PI); };
    auto wLn = [&](double r,double z){
        double d2 = std::max((r-RP)*(r-RP)+(z-ZP)*(z-ZP), 1e-300);
        return -0.5*std::log(d2);
    };
    double mP_c = JP*RP/(2.0*M_PI);
    double gradCR = g*(dJR*RP + JP*0.5)/(2.0*M_PI);
    double gradCZ = g*dJZ*RP/(2.0*M_PI);
    double psiWeak = weakSingIntegral(ctx,R,Z,RP,ZP, mC, wLn, mP_c, gradCR, gradCZ, C.lnD, C.mlnR, C.mlnZ);

    Result out;
    out.psi = psiBnd + psiWeak;

    // ---- B_R ------------------------------------------------------------
    auto gbrBndK = [&](double r,double z){ return Singularity::gbrBnd(r,z,RP,ZP); };
    double brBnd = cutCellIntegral(ctx, R, Z, [&](double r,double z){ return Jfun(r,z)*gbrBndK(r,z); });
    auto wZ = [&](double r,double z){
        double d2 = std::max((r-RP)*(r-RP)+(z-ZP)*(z-ZP), 1e-300);
        return (ZP-z)/d2;
    };
    // mBR(s) = Jphi(s)*gamma(s,P)/(4*pi*RP): the UNFROZEN dipole coefficient
    // shared by both B_R's own dipole (below, against wZ) and B_Z's dipole
    // (against wR, further down) -- docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf
    // sec 5.4/6.2 eq. 10. gamma(P,P)=2*RP and grad_s gamma(P,P)=(1,0)
    // (immediate from gamma's own definition, derived by hand there) give
    // mBR(P) and its gradient in closed form, replacing Jfun/JP/dJR/dJZ
    // (the OLD frozen coefficient, folded in afterwards via the outer
    // 1/(2*pi) multiply that this replaces) as the m passed to
    // weakSingIntegral. No new exact geometric constants are needed: wZ,
    // wR, and C.dipZ/dZ_R/dZ_Z, C.dipR/dR_R/dR_Z (the bare-kernel exact
    // integrals) are unchanged -- only which function plays "m" changes.
    auto mBR = [&](double r, double z){
        double gam = std::sqrt(Singularity::gammaSq(r,z,RP,ZP));
        return Jfun(r,z) * gam / (4.0*M_PI*RP);
    };
    double mBR_P     = JP / (2.0*M_PI);
    double gradMBR_R = g*( dJR/(2.0*M_PI) + JP/(4.0*M_PI*RP) );
    double gradMBR_Z = g*( dJZ/(2.0*M_PI) );
    double brWeak = weakSingIntegral(ctx,R,Z,RP,ZP, mBR, wZ, mBR_P, gradMBR_R, gradMBR_Z, C.dipZ, C.dZ_R, C.dZ_Z);
    out.br = brBnd + brWeak;

    // ---- B_Z (dipole term + extra log term) -----------------------------
    auto gbzBndK = [&](double r,double z){ return Singularity::gbzBnd(r,z,RP,ZP); };
    double bzBnd = cutCellIntegral(ctx, R, Z, [&](double r,double z){ return Jfun(r,z)*gbzBndK(r,z); });
    auto wR = [&](double r,double z){
        double d2 = std::max((r-RP)*(r-RP)+(z-ZP)*(z-ZP), 1e-300);
        return (r-RP)/d2;
    };
    double bzWeakDipole = weakSingIntegral(ctx,R,Z,RP,ZP, mBR, wR, mBR_P, gradMBR_R, gradMBR_Z, C.dipR, C.dR_R, C.dR_Z);

    // NOTE the sign: this term's weight is +ln(d), the OPPOSITE of psi_p's
    // w=-ln(d) (wLn above) -- Iw/Isw are correspondingly -lnD/-mlnR/-mlnZ
    // (since lnD/mlnR/mlnZ are defined as integrals of -ln(d)), but that
    // alone is not enough: the residual weight function passed to
    // weakSingIntegral must be the SAME +ln(d), not wLn itself. Using wLn
    // here (found 2026-09-11 via the B_Z level-2 acceptance test showing a
    // stalled ~0 order instead of the expected ~2) silently integrated the
    // residual against the wrong-signed kernel while only Iw/Isw were fixed.
    //
    // With that wLnPos/-C.lnD convention kept as-is (per this file's own
    // design), m itself must carry a NEGATIVE sign so that m(s)*wLnPos(s)
    // reproduces the true target Jphi(s)*(-ln d(s,P))/(2*pi*gamma(s,P)) --
    // the genuinely divergent piece of B_Z's log term (L/(2*pi*gamma) =
    // ln(4*gamma)/(2*pi*gamma) - ln(d)/(2*pi*gamma), see gbzBnd() in
    // singularity_kernels.h). I.e. m = -m_{BZ,log} of
    // docs/FEATURE_CLOSEDFORM_GREG_BRIEFING.pdf eq. 11, not +m_{BZ,log} as
    // literally written there (that equation does not specify which sign of
    // the log kernel it pairs with). Confirmed against the OLD frozen-
    // coefficient mG2/mP_g2 this replaces, which carried the exact same
    // leading minus sign for the same reason (mP_g2 = -JP/(4*pi*RP) equals
    // -m_{BZ,log}(P) here, since gamma(P,P)=2*RP makes the frozen and
    // unfrozen coefficients coincide exactly at P).
    auto wLnPos = [&](double r,double z){ return -wLn(r,z); };
    auto mBZlog = [&](double r, double z){
        double gam = std::sqrt(Singularity::gammaSq(r,z,RP,ZP));
        return -Jfun(r,z) / (2.0*M_PI*gam);
    };
    double mBZlog_P     = -JP / (4.0*M_PI*RP);
    double gradMBZlog_R = g*( -dJR/(4.0*M_PI*RP) + JP/(8.0*M_PI*RP*RP) );
    double gradMBZlog_Z = g*( -dJZ/(4.0*M_PI*RP) );
    double bzWeakLog = weakSingIntegral(ctx,R,Z,RP,ZP, mBZlog, wLnPos, mBZlog_P, gradMBZlog_R, gradMBZlog_Z, -C.lnD, -C.mlnR, -C.mlnZ);

    out.bz = bzBnd + bzWeakDipole + bzWeakLog;

    return out;
}

// Production entry point: Jphi(P) and its gradient come from the cached
// grid reconstruction (ctx.jfield/ctx.geom), computed once here and reused
// for all three fields (avoids recomputing the 9 geometric constants three
// times, which is why buildXPointConstraints() should call this once per
// X-point rather than psiAt/brAt/bzAt separately).
inline Result evaluateAll(const PlasmaContext& ctx,
                           const std::vector<double>& R, const std::vector<double>& Z,
                           double RP, double ZP, int order = 2){
    double JP = JphiGrid::valueAt(ctx.jfield, RP, ZP);
    JphiGrid::Gradient grad = JphiGrid::gradientAt(ctx.geom, ctx.jfield, RP, ZP);
    auto Jfun = [&](double r, double z){ return JphiGrid::valueAt(ctx.jfield, r, z); };
    return evaluateAllWithJfun(ctx, R, Z, RP, ZP, Jfun, JP, grad.dR, grad.dZ, order);
}

inline double psiAt(const PlasmaContext& ctx, const std::vector<double>& R, const std::vector<double>& Z,
                     double RP, double ZP, int order = 2){ return evaluateAll(ctx,R,Z,RP,ZP,order).psi; }
inline double brAt(const PlasmaContext& ctx, const std::vector<double>& R, const std::vector<double>& Z,
                    double RP, double ZP, int order = 2){ return evaluateAll(ctx,R,Z,RP,ZP,order).br; }
inline double bzAt(const PlasmaContext& ctx, const std::vector<double>& R, const std::vector<double>& Z,
                    double RP, double ZP, int order = 2){ return evaluateAll(ctx,R,Z,RP,ZP,order).bz; }

} // namespace Singularity
#endif // SINGULARITY_SUBTRACTION_H
