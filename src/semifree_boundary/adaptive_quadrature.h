/*=============================================================================
  ADAPTIVE GAUSS-KRONROD QUADRATURE (1D)
  =============================================================================
  Adaptive Gauss-Kronrod (G7-K15) quadrature for smooth-per-panel 1D
  integrands with a priori known "difficult" points (polygon vertices, the
  singular point of the row-scan in singularity_subtraction.h). This is the
  only piece of quadrature infrastructure in the project with no Python
  equivalent (the prototypes use scipy.integrate.quad) -- see
  docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf section 6.7 and the
  "Decisiones de diseno pendientes" resolved in
  ~/.claude memory cutcell_singularity_design_decisions: this implements the
  Gauss-Kronrod METHOD itself (error estimate from the Gauss/Kronrod
  rule-pair difference, plus recursive panel subdivision), not a
  byte-for-byte reproduction of scipy.integrate.quad/QUADPACK's internal
  extrapolation (wynn epsilon-algorithm) or default tolerances.
  =============================================================================*/
#ifndef ADAPTIVE_QUADRATURE_H
#define ADAPTIVE_QUADRATURE_H

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

namespace Quadrature {

struct Result {
    double value;
    double errorEstimate;
    int nPanels;
    int nEvals;
};

namespace detail {

// Standard Gauss-7 / Kronrod-15 nodes and weights on [-1,1] (the QUADPACK
// QK15 constants). xgk holds the 8 non-negative Kronrod nodes (index 7 is
// the center, x=0); the other 7 Kronrod nodes are -xgk[0..6] by symmetry.
// wgk are the Kronrod weights aligned with xgk. wg are the 4 Gauss-7
// weights, aligned with the ODD-indexed Kronrod nodes xgk[1],xgk[3],xgk[5]
// and the center xgk[7] -- those four nodes are exactly the 7-point
// Gauss-Legendre nodes, which is what lets one function-evaluation pass
// serve both rules at once.
static const double xgk[8] = {
    0.991455371120813, 0.949107912342759, 0.864864423359769,
    0.741531185599394, 0.586087235467691, 0.405845151377397,
    0.207784955007898, 0.000000000000000
};
static const double wgk[8] = {
    0.022935322010529, 0.063092092629979, 0.104790010322250,
    0.140653259715525, 0.169004726639267, 0.190350578064785,
    0.204432940075298, 0.209482141084728
};
static const double wg[4] = {
    0.129484966168870, 0.279705391489277, 0.381830050505119,
    0.417959183673469
};

struct PanelResult { double kronrod, gauss, errEstimate; int nEvals; };

inline PanelResult evalPanel(const std::function<double(double)>& f, double a, double b){
    double halfLength = 0.5*(b-a);
    double center = 0.5*(a+b);

    double fc = f(center);
    double resk = wgk[7]*fc;
    double resg = wg[3]*fc;
    int nEvals = 1;

    // Nodes shared between Gauss-7 and Kronrod-15: xgk[1],xgk[3],xgk[5].
    for(int j = 0; j < 3; ++j){
        int gkIdx = 2*j + 1;
        double xj = xgk[gkIdx]*halfLength;
        double f1 = f(center - xj);
        double f2 = f(center + xj);
        nEvals += 2;
        resg += wg[j]*(f1+f2);
        resk += wgk[gkIdx]*(f1+f2);
    }
    // Kronrod-only nodes: xgk[0],xgk[2],xgk[4],xgk[6].
    for(int j = 0; j < 4; ++j){
        int gkIdx = 2*j;
        double xj = xgk[gkIdx]*halfLength;
        double f1 = f(center - xj);
        double f2 = f(center + xj);
        nEvals += 2;
        resk += wgk[gkIdx]*(f1+f2);
    }

    PanelResult out;
    out.kronrod = resk*halfLength;
    out.gauss   = resg*halfLength;
    out.errEstimate = std::fabs(out.kronrod - out.gauss);
    out.nEvals = nEvals;
    return out;
}

struct Task { double a, b, epsabs; int depth; };

// Iterative (explicit-stack) refinement, deliberately NOT recursive: with
// plain recursion, the check "have we used up the panel budget yet" is only
// re-read at the moment a node DECIDES to split, but a single already-
// permitted split can then dive arbitrarily deep before any of its
// descendants finally accept a leaf (an oscillatory integrand can need many
// levels of halving before any sub-panel satisfies tolerance), so nPanels
// stays stale for the whole dive and the eventual burst of leaves at the
// bottom can blow far past maxPanels. Working off an explicit stack lets the
// budget check use nPanels + stack.size() (done + still-pending panels) at
// EVERY single split decision, which bounds the final panel count to
// maxPanels (up to a small additive slack) regardless of how adversarial the
// integrand is.
inline void refineIterative(const std::function<double(double)>& f,
                             const std::vector<Task>& initialTasks,
                             double epsrel, int maxDepth, int maxPanels,
                             double& valueAccum, double& errAccum, int& nPanels, int& nEvals){
    std::vector<Task> stack(initialTasks.rbegin(), initialTasks.rend());

    while(!stack.empty()){
        Task t = stack.back();
        stack.pop_back();

        double mid = 0.5*(t.a+t.b);
        double scale = std::max(std::max(std::fabs(t.a), std::fabs(t.b)), 1.0);
        // A relative-width floor (not just the exact mid==a/mid==b check)
        // is needed: once a panel's width is within a few orders of
        // magnitude of the endpoints' own floating-point resolution,
        // individual Gauss-Kronrod node OFFSETS from the center can already
        // round away to nothing (center +/- tiny_offset rounds back to
        // center) well before mid itself becomes bit-identical to a or b,
        // silently placing an evaluation point exactly on a singularity a
        // breakpoint deliberately put at a panel edge.
        bool degenerate = !(mid > t.a) || !(mid < t.b) || ((t.b - t.a) < 1e-12*scale);
        if(degenerate){
            // Repeated bisection has converged onto a single point (almost
            // always an integrable singularity a breakpoint placed us on
            // top of): this panel's measure is numerically zero, and for a
            // convergent integral (log or 1/sqrt-type) its true contribution
            // vanishes with its width (e.g. integral(-ln|x|,-eps,eps) -> 0 as
            // eps -> 0). Skip evaluating f() here entirely rather than risk
            // feeding it the exact singular point (which would otherwise
            // produce -inf/NaN and poison the whole sum).
            nPanels += 1;
            continue;
        }

        PanelResult pr = evalPanel(f, t.a, t.b);
        nEvals += pr.nEvals;

        double tol = std::max(t.epsabs, epsrel*std::fabs(pr.kronrod));
        // Splitting replaces this one pending task with two: allow it only
        // if doing so still keeps (done + pending) within maxPanels.
        bool budgetLeft = (nPanels + (int)stack.size() + 2 <= maxPanels);
        bool canSubdivide = (t.depth < maxDepth) && budgetLeft;

        if(pr.errEstimate <= tol || !canSubdivide){
            valueAccum += pr.kronrod;
            errAccum   += pr.errEstimate;
            nPanels    += 1;
        } else {
            stack.push_back({t.a, mid, 0.5*t.epsabs, t.depth+1});
            stack.push_back({mid, t.b, 0.5*t.epsabs, t.depth+1});
        }
    }
}

} // namespace detail

// Integrates f over [a,b] (a<=b or a>b, handled by sign flip), refining
// recursively wherever the Gauss-7/Kronrod-15 disagreement exceeds
// max(epsabs, epsrel*|Kronrod value|). `breakpoints` strictly inside (a,b)
// are inserted as initial panel boundaries -- known non-smooth points
// (polygon vertices, the row-scan's singular row/column) should always be
// passed here rather than left for the adaptive search to discover, both
// for speed and because a kink the recursion never has to "notice" cannot
// be missed by an unlucky panel split.
inline Result integrate(const std::function<double(double)>& f, double a, double b,
                         const std::vector<double>& breakpoints = {},
                         double epsabs = 1e-11, double epsrel = 1e-8,
                         int maxDepth = 50, int maxPanels = 4000){
    double lo = std::min(a,b), hi = std::max(a,b);
    double sign = (b >= a) ? 1.0 : -1.0;

    std::vector<double> nodes;
    nodes.push_back(lo);
    for(double bp : breakpoints) if(bp > lo && bp < hi) nodes.push_back(bp);
    nodes.push_back(hi);
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    double value = 0.0, err = 0.0;
    int nPanels = 0, nEvals = 0;
    int nSeg = (int)nodes.size() - 1;
    double segEpsabs = (nSeg > 0) ? epsabs/nSeg : epsabs;

    std::vector<detail::Task> initialTasks;
    initialTasks.reserve(nSeg);
    for(int i = 0; i < nSeg; ++i)
        initialTasks.push_back({nodes[i], nodes[i+1], segEpsabs, 0});

    detail::refineIterative(f, initialTasks, epsrel, maxDepth, maxPanels, value, err, nPanels, nEvals);

    Result out;
    out.value = sign*value;
    out.errorEstimate = err;
    out.nPanels = nPanels;
    out.nEvals = nEvals;
    return out;
}

} // namespace Quadrature
#endif // ADAPTIVE_QUADRATURE_H
