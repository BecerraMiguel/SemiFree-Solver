/*-----------------------------------------------------------------------------------------------------------
 REGULARIZATION.H -- automatic Tikhonov regularization for the coil-current inverse problem
 min_I ||J*I - r||^2, via the SVD of J (never forming the normal equations A=2*J^T*J).

 Direct C++/Eigen port of the algorithm already validated in Python
 (tools/regularization/common.py, tools/regularization/lcurve_analysis.py) and documented in
 docs/reports/REGULARIZATION_STUDY_REPORT.md -- see that report and
 docs/FEATURE_AUTO_REGULARIZATION_BRIEFING.md for the full derivation, validation, and reference
 numbers (kappa(J), lambda_MC, resolved currents) this header's own test suite
 (src/semifree_boundary/test_regularization.cpp) checks against.

 Scope: covers both the Nx=0 (no X-points) case, via solveWithAutoRegularization() directly, and
 the Nx>0 (X-point) case, via solveConstrainedWithAutoRegularization() -- an equality-constrained
 least-squares (LSE) problem min||JI-r||^2 s.t. CI=d, reduced to an unconstrained problem through
 the null-space method (Golub & Van Loan) and then solved by reusing solveWithAutoRegularization()
 unmodified. See docs/FEATURE_XPOINT_NULLSPACE_REGULARIZATION_BRIEFING.md for the full derivation.

 Conventions:
   - J is Nb x Nc (Nb boundary points, Nc coils), r is length Nb, both in the solver's normalized
     units (same convention as SemiFree_Solver.cpp -- R,Z by Lo, currents by Io, flux by Psi_o).
   - "lambda" here is always in the STANDARD Tikhonov convention,
     min_I ||J*I-r||^2 + lambda*||I||^2 -- NOT the buildSystem()-diagonal convention
     (A_ii += lambda, where A=2*J^T*J), which is exactly 2x this lambda. Callers that need to
     compare against buildSystem()'s historical lambda=2.976351e-05 must divide it by 2 first.

 Algorithm for the "auto" lambda: Cultrera, A., Callegaro, L., "A simple algorithm to find the
 L-curve corner in the regularisation of ill-posed inverse problems," arXiv:1608.04571 (2016) --
 Menger curvature of the (log-residual, log-solution-norm) L-curve, maximized via a golden-section
 search over log10(lambda). Unlike the Python reference (which pre-sweeps 60 lambda values and
 fits a cubic spline to interpolate between them, for cheap plotting), this implementation
 evaluates the L-curve point exactly at whatever lambda the search queries -- each evaluation is
 already O(Nc) given the cached SVD, so there is no need to sweep+interpolate here; this is
 slightly more accurate than the Python version, not an approximation of it.
-----------------------------------------------------------------------------------------------------------*/
#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace Regularization {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;

/*-------------------------------------------------------------------------------------------------
SVD
-------------------------------------------------------------------------------------------------*/
inline JacobiSVD<MatrixXd> computeSVD(const MatrixXd& J){
    return JacobiSVD<MatrixXd>(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
}

// kappa(J) = sigma_max/sigma_min. Assumes S is sorted descending (Eigen's JacobiSVD guarantees this).
inline double conditionNumber(const VectorXd& S){
    return S(0) / S(S.size() - 1);
}

/*-------------------------------------------------------------------------------------------------
TIKHONOV FILTER AND SOLVE
-------------------------------------------------------------------------------------------------*/
// Tikhonov filter factors: sigma_i / (sigma_i^2 + lambda). Port of
// tools/regularization/common.py:tikhonov_filter().
inline VectorXd tikhonovFilter(const VectorXd& S, double lambda){
    return (S.array() / (S.array().square() + lambda)).matrix();
}

// I = V * diag(filter) * U^T * r. Port of tools/regularization/common.py:solve_tikhonov().
inline VectorXd solveTikhonov(const JacobiSVD<MatrixXd>& svd, const VectorXd& r, double lambda){
    VectorXd filt = tikhonovFilter(svd.singularValues(), lambda);
    VectorXd Utr = svd.matrixU().transpose() * r;
    return svd.matrixV() * (filt.array() * Utr.array()).matrix();
}

/*-------------------------------------------------------------------------------------------------
MENGER CURVATURE AND L-CURVE CORNER
-------------------------------------------------------------------------------------------------*/
// Signed Menger curvature of the circle through P1,P2,P3 (arXiv:1608.04571 eq. 4):
// 2*(signed area)/(product of the three pairwise distances). Positive for a left turn
// P1->P2->P3. Port of tools/regularization/lcurve_analysis.py:_menger_curvature().
inline double mengerCurvature(double x1, double y1, double x2, double y2, double x3, double y3){
    double signedArea2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    double d12 = std::hypot(x2 - x1, y2 - y1);
    double d23 = std::hypot(x3 - x2, y3 - y2);
    double d31 = std::hypot(x1 - x3, y1 - y3);
    double denom = d12 * d23 * d31;
    if(denom < 1e-300){
        return 0.0;
    }
    return 2.0 * signedArea2 / denom;
}

struct LCurvePoint {
    double log_residual;
    double log_norm;
};

inline LCurvePoint evaluateLCurvePoint(const JacobiSVD<MatrixXd>& svd, const MatrixXd& J,
                                        const VectorXd& r, double lambda){
    VectorXd I = solveTikhonov(svd, r, lambda);
    double residual = (J * I - r).norm();
    double normI = I.norm();
    LCurvePoint p;
    p.log_residual = std::log10(residual);
    p.log_norm = std::log10(normI);
    return p;
}

inline double curvatureAtLogLambda(const JacobiSVD<MatrixXd>& svd, const MatrixXd& J,
                                    const VectorXd& r, double x, double h){
    LCurvePoint p1 = evaluateLCurvePoint(svd, J, r, std::pow(10.0, x - h));
    LCurvePoint p2 = evaluateLCurvePoint(svd, J, r, std::pow(10.0, x));
    LCurvePoint p3 = evaluateLCurvePoint(svd, J, r, std::pow(10.0, x + h));
    return mengerCurvature(p1.log_residual, p1.log_norm,
                            p2.log_residual, p2.log_norm,
                            p3.log_residual, p3.log_norm);
}

// Golden-section search maximizing the Menger curvature of the L-curve, as a function of
// x=log10(lambda), over [log10(lambda_lo)+h, log10(lambda_hi)-h]. Port of
// tools/regularization/lcurve_analysis.py:find_lcurve_corner() (there operating on a
// cubic-spline interpolant of a pre-swept grid; here evaluating the L-curve exactly at each
// queried lambda -- see the file header).
inline double findLCurveCorner(const JacobiSVD<MatrixXd>& svd, const MatrixXd& J,
                                const VectorXd& r, double lambda_lo, double lambda_hi,
                                double h = 0.05, double tol = 1e-3){
    double a = std::log10(lambda_lo) + h;
    double b = std::log10(lambda_hi) - h;
    const double invphi = 2.0 / (1.0 + std::sqrt(5.0));

    double c = b - (b - a) * invphi;
    double d = a + (b - a) * invphi;
    double fc = curvatureAtLogLambda(svd, J, r, c, h);
    double fd = curvatureAtLogLambda(svd, J, r, d, h);

    while((b - a) > tol){
        if(fc > fd){
            b = d; d = c; fd = fc;
            c = b - (b - a) * invphi;
            fc = curvatureAtLogLambda(svd, J, r, c, h);
        } else {
            a = c; c = d; fc = fd;
            d = a + (b - a) * invphi;
            fd = curvatureAtLogLambda(svd, J, r, d, h);
        }
    }
    double xOpt = 0.5 * (a + b);
    return std::pow(10.0, xOpt);
}

/*-------------------------------------------------------------------------------------------------
ORCHESTRATOR
-------------------------------------------------------------------------------------------------*/
struct RegularizationResult {
    double kappa_J;
    double lambda_used;      // standard Tikhonov convention (see file header)
    bool lambda_was_auto;
    VectorXd currents;       // normalized units, same convention as J,r
};

// lambda_override: pass a negative value to mean "auto" (find lambda via the L-curve corner).
// A non-negative value is used directly as the standard-convention Tikhonov lambda.
inline RegularizationResult solveWithAutoRegularization(const MatrixXd& J, const VectorXd& r,
                                                          double lambda_override = -1.0){
    JacobiSVD<MatrixXd> svd = computeSVD(J);
    const VectorXd& S = svd.singularValues();
    double kappaJ = conditionNumber(S);

    double lambdaUsed;
    bool isAuto;
    if(lambda_override >= 0.0){
        lambdaUsed = lambda_override;
        isAuto = false;
    } else {
        double sigmaMin = S(S.size() - 1);
        double sigmaMax = S(0);
        double lambdaLo = sigmaMin * sigmaMin * 1e-4;
        double lambdaHi = sigmaMax * sigmaMax * 1e4;
        lambdaUsed = findLCurveCorner(svd, J, r, lambdaLo, lambdaHi);
        isAuto = true;
    }

    VectorXd currents = solveTikhonov(svd, r, lambdaUsed);

    RegularizationResult result;
    result.kappa_J = kappaJ;
    result.lambda_used = lambdaUsed;
    result.lambda_was_auto = isAuto;
    result.currents = currents;
    return result;
}

/*-------------------------------------------------------------------------------------------------
NULL-SPACE METHOD FOR EQUALITY-CONSTRAINED LEAST SQUARES (Nx>0)
-------------------------------------------------------------------------------------------------*/
// Reduction of an equality constraint C*I=d to a particular solution plus a null-space basis:
// I = I_particular + Z*y, with C*Z=0 and y free. C is 3*Nx x Nc (Nx X-points, Nc coils), typically
// with 3*Nx << Nc, so the SVD of C is taken with FULL V (Eigen::ComputeFullV) rather than thin --
// the null-space basis lives exactly in the columns of V that a thin SVD would discard.
struct NullSpaceReduction {
    VectorXd I_particular;   // I_p = C^+ d (minimum norm)
    MatrixXd Z;              // orthonormal basis of null(C), Nc x (Nc - rank(C))
    double kappa_C;          // sigma_max(C) / sigma_min(C), among retained singular values only
    int rank_C;               // numerical rank of C (can be < 3*Nx if X-points are near-degenerate)
};

// rank_tol: singular values of C smaller than rank_tol*sigma_max(C) are treated as zero (that
// direction becomes part of Z instead of being treated as a genuine constraint).
inline NullSpaceReduction reduceEqualityConstraint(const MatrixXd& C, const VectorXd& d,
                                                    double rank_tol = 1e-10){
    JacobiSVD<MatrixXd> svdC(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const VectorXd& Sc = svdC.singularValues();   // length m = min(rows,cols) = 3*Nx (assuming Nc >= 3*Nx)
    int m = Sc.size();
    double sigmaMax = Sc(0);

    int rank = 0;
    for(int i = 0; i < m; ++i){
        if(Sc(i) > rank_tol * sigmaMax) rank++;
        else break;   // Eigen sorts singular values in descending order
    }
    if(rank == 0){
        throw std::runtime_error("reduceEqualityConstraint: C is numerically zero "
                                  "(X-points have no detectable influence from the coils)");
    }

    VectorXd Ip = VectorXd::Zero(C.cols());
    for(int i = 0; i < rank; ++i){
        Ip += (svdC.matrixU().col(i).dot(d) / Sc(i)) * svdC.matrixV().col(i);
    }

    int Nc = C.cols();
    int nullDim = Nc - rank;
    MatrixXd Z(Nc, nullDim);
    int col = 0;
    for(int i = rank; i < m; ++i)  Z.col(col++) = svdC.matrixV().col(i);  // near-zero within the first m
    for(int i = m; i < Nc; ++i)    Z.col(col++) = svdC.matrixV().col(i);  // exact-null columns

    NullSpaceReduction result;
    result.I_particular = Ip;
    result.Z = Z;
    result.kappa_C = sigmaMax / Sc(rank - 1);
    result.rank_C = rank;
    return result;
}

struct RegularizationResultConstrained {
    double kappa_C;
    double kappa_J_reduced;       // kappa(J*Z) -- plays the same role as kappa_J in the Nx=0 case
    double lambda_used;
    bool lambda_was_auto;
    VectorXd currents;             // full I, mapped back: I = I_p + Z*y
    double constraint_residual;    // ||C*I - d||, should be ~1e-10 or better
    int rank_C;
};

// lambda_override: same convention as solveWithAutoRegularization (negative = "auto").
inline RegularizationResultConstrained solveConstrainedWithAutoRegularization(
        const MatrixXd& J, const VectorXd& r,
        const MatrixXd& C, const VectorXd& d,
        double lambda_override = -1.0, double rank_tol = 1e-10){
    if(C.cols() <= C.rows()){
        throw std::runtime_error("solveConstrainedWithAutoRegularization: Nc <= 3*Nx "
                                  "(more X-point constraints than coils) -- null space would be empty");
    }

    NullSpaceReduction ns = reduceEqualityConstraint(C, d, rank_tol);
    MatrixXd Jt = J * ns.Z;
    VectorXd rt = r - J * ns.I_particular;

    RegularizationResult reg = solveWithAutoRegularization(Jt, rt, lambda_override);  // UNCHANGED

    VectorXd I = ns.I_particular + ns.Z * reg.currents;

    RegularizationResultConstrained out;
    out.kappa_C = ns.kappa_C;
    out.kappa_J_reduced = reg.kappa_J;
    out.lambda_used = reg.lambda_used;
    out.lambda_was_auto = reg.lambda_was_auto;
    out.currents = I;
    out.constraint_residual = (C * I - d).norm();
    out.rank_C = ns.rank_C;
    return out;
}

} // namespace Regularization

#endif // REGULARIZATION_H
