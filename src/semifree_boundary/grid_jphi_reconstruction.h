/*=============================================================================
  Jphi RECONSTRUCTION FROM MESH DATA (value + gradient) AT ARBITRARY POINTS
  =============================================================================
  Port of tools/singularity/grid_jphi.py. The centroid rule (cutcell_geometry.h)
  needs Jphi at cut-cell centroids (in general not mesh nodes), and the
  singularity subtraction (singularity_subtraction.h) needs Jphi(P) and
  grad(Jphi)(P) exactly at the evaluation point P -- always on the boundary
  Gamma in this project. Requires cutcell_geometry.h to be included first
  (uses CutCell::Geometry).
  =============================================================================*/
#ifndef GRID_JPHI_RECONSTRUCTION_H
#define GRID_JPHI_RECONSTRUCTION_H

#include "cutcell_geometry.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>

namespace JphiGrid {

struct Field {
    std::vector<double> R, Z;                      // same grid as Jphi
    std::vector<std::vector<double>> repaired;      // ghost-filled, valid everywhere
    std::vector<std::vector<bool>> wasGhost;        // true if the node was extrapolated
};

namespace detail {

// 8-connected binary dilation, `iterations` steps.
inline std::vector<std::vector<bool>> dilate8(const std::vector<std::vector<bool>>& mask, int iterations){
    int npr = (int)mask.size();
    int npz = npr > 0 ? (int)mask[0].size() : 0;
    std::vector<std::vector<bool>> cur = mask;
    for(int it = 0; it < iterations; ++it){
        std::vector<std::vector<bool>> next = cur;
        for(int i = 0; i < npr; ++i){
            for(int k = 0; k < npz; ++k){
                if(cur[i][k]) continue;
                bool any = false;
                for(int di = -1; di <= 1 && !any; ++di){
                    for(int dk = -1; dk <= 1 && !any; ++dk){
                        if(di == 0 && dk == 0) continue;
                        int ii = i+di, kk = k+dk;
                        if(ii >= 0 && ii < npr && kk >= 0 && kk < npz && cur[ii][kk]) any = true;
                    }
                }
                if(any) next[i][k] = true;
            }
        }
        cur = next;
    }
    return cur;
}

// Ghost-fill rule for one exterior node (a,c): try the 4 cardinal directions
// for two consecutive interior neighbors J1 (1 step), J2 (2 steps) and
// extrapolate linearly, Jghost = 2*J1 - J2; if none of the 4 directions
// qualifies, fall back to the nearest interior node in a 5x5 block. If that
// block has no interior node either, this is a pathological isolated
// exterior region -- throw rather than silently corrupting the result.
inline double ghostValueAtNode(const std::vector<std::vector<bool>>& inside,
                                const std::vector<std::vector<double>>& Jgrid,
                                int a, int c){
    int npr = (int)inside.size();
    int npz = npr > 0 ? (int)inside[0].size() : 0;
    static const int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for(const auto& d : dirs){
        int a1 = a+d[0], c1 = c+d[1];
        int a2 = a+2*d[0], c2 = c+2*d[1];
        if(a1 >= 0 && a1 < npr && c1 >= 0 && c1 < npz &&
           a2 >= 0 && a2 < npr && c2 >= 0 && c2 < npz &&
           inside[a1][c1] && inside[a2][c2]){
            return 2.0*Jgrid[a1][c1] - Jgrid[a2][c2];
        }
    }
    double best = 0.0, bestD2 = 1e300;
    bool found = false;
    for(int di = -2; di <= 2; ++di){
        for(int dk = -2; dk <= 2; ++dk){
            int ii = a+di, kk = c+dk;
            if(ii < 0 || ii >= npr || kk < 0 || kk >= npz) continue;
            if(!inside[ii][kk]) continue;
            double d2 = (double)(di*di + dk*dk);
            if(d2 < bestD2){ bestD2 = d2; best = Jgrid[ii][kk]; found = true; }
        }
    }
    if(!found)
        throw std::runtime_error("JphiGrid::build: ghost-fill found no interior "
                                  "node in the 5x5 block around an exterior node "
                                  "-- pathologically isolated exterior region");
    return best;
}

} // namespace detail

// Repairs, ONCE per mesh, the whole band of exterior nodes within 2 mesh
// steps of any interior node (8-connected dilation), then bilinear
// interpolation on the repaired grid is valid for any (r,z), including on
// Gamma. Nodes further outside than that band are left untouched (never
// queried, since evaluation points are always on or near Gamma in this
// project -- see docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf section 1).
inline Field build(const CutCell::Geometry& geom,
                    const std::vector<std::vector<double>>& JphiMasked,
                    const std::vector<double>& R, const std::vector<double>& Z){
    Field f;
    f.R = R; f.Z = Z;
    int npr = (int)R.size(), npz = (int)Z.size();
    f.repaired = JphiMasked;
    f.wasGhost.assign(npr, std::vector<bool>(npz, false));

    std::vector<std::vector<bool>> band = detail::dilate8(geom.insideNode, 2);
    for(int i = 0; i < npr; ++i){
        for(int k = 0; k < npz; ++k){
            if(band[i][k] && !geom.insideNode[i][k]){
                f.repaired[i][k] = detail::ghostValueAtNode(geom.insideNode, JphiMasked, i, k);
                f.wasGhost[i][k] = true;
            }
        }
    }
    return f;
}

// Ordinary bilinear interpolation on the repaired grid. Fractional index is
// clamped to [0, n-1-epsilon] (mirrors cutcell.bilinear) so a point exactly
// on the last node/right at the mesh edge never overflows the i+1 lookup.
inline double valueAt(const Field& field, double r, double z){
    int npr = (int)field.R.size(), npz = (int)field.Z.size();
    double dR = field.R[1]-field.R[0], dZ = field.Z[1]-field.Z[0];
    double fi = (r-field.R[0]) / dR;
    double fk = (z-field.Z[0]) / dZ;
    fi = std::max(0.0, std::min(fi, npr - 1.0000001));
    fk = std::max(0.0, std::min(fk, npz - 1.0000001));

    int i0 = (int)std::floor(fi), i1 = i0+1;
    int k0 = (int)std::floor(fk), k1 = k0+1;
    double tR = fi - i0, tZ = fk - k0;

    double v00 = field.repaired[i0][k0], v10 = field.repaired[i1][k0];
    double v01 = field.repaired[i0][k1], v11 = field.repaired[i1][k1];
    double v0 = v00*(1.0-tR) + v10*tR;
    double v1 = v01*(1.0-tR) + v11*tR;
    return v0*(1.0-tZ) + v1*tZ;
}

struct Gradient {
    double dR = 0.0, dZ = 0.0;
    int nPointsUsed = 0;
    bool usedLinearFallback = false;
    bool farFromPlasma = false; // true: zero interior nodes even at maxHalfWidth
};

// Local quadratic fit by least squares (moving-least-squares / Savitzky-Golay
// style): Jhat(r,z) = a0 + a1(r-RP) + a2(z-ZP) + a3(r-RP)^2 + a4(z-ZP)^2 +
// a5(r-RP)(z-ZP), fit over a window of GENUINELY interior nodes (never
// ghost/extrapolated -- critical restriction found during the original
// Python validation: mixing them in gives gradients wrong by orders of
// magnitude near a convex boundary, where a small symmetric window is
// locally a half-plane dominated by fill, not real data). The window grows
// (half-width 2 up to maxHalfWidth) until at least minFitPoints genuinely
// interior nodes are gathered. If even the largest window has fewer than 6
// (the quadratic's own unknown count), falls back to a linear fit (3
// unknowns); NOT using field.wasGhost but geom.insideNode directly, since
// that is the actual point-in-polygon fact the restriction is about.
inline Gradient gradientAt(const CutCell::Geometry& geom, const Field& field,
                            double RP, double ZP,
                            int minFitPoints = 15, int maxHalfWidth = 8){
    int npr = geom.npr, npz = geom.npz;
    double dR = field.R[1]-field.R[0], dZ = field.Z[1]-field.Z[0];
    int i0 = (int)std::round((RP-field.R[0]) / dR);
    int k0 = (int)std::round((ZP-field.Z[0]) / dZ);
    i0 = std::max(0, std::min(i0, npr-1));
    k0 = std::max(0, std::min(k0, npz-1));

    std::vector<int> idxI, idxK;
    int hw = 2;
    while(true){
        idxI.clear(); idxK.clear();
        int ilo = std::max(0, i0-hw), ihi = std::min(npr-1, i0+hw);
        int klo = std::max(0, k0-hw), khi = std::min(npz-1, k0+hw);
        for(int i = ilo; i <= ihi; ++i){
            for(int k = klo; k <= khi; ++k){
                if(geom.insideNode[i][k]){
                    idxI.push_back(i);
                    idxK.push_back(k);
                }
            }
        }
        if((int)idxI.size() >= minFitPoints || hw >= maxHalfWidth) break;
        hw++;
    }

    int n = (int)idxI.size();
    Gradient g;
    g.nPointsUsed = n;

    // n==0 at maxHalfWidth means P sits genuinely far from ANY plasma --
    // found 2026-09-12 while integrating this into the psi_check.txt sweep,
    // which (unlike buildLeastSquaresSystem/buildXPointConstraints) queries
    // P over the WHOLE mesh, not just on Gamma; a deep-exterior point is a
    // real case there. With zero interior evidence anywhere nearby, the
    // TRUE Jphi and its gradient both ARE exactly zero in that whole
    // neighborhood (Jt.txt is zero outside the LCFS by construction) -- this
    // is not an approximation of missing data, it is the physically correct
    // answer, so return it directly instead of treating it as pathological.
    if(n == 0){
        g.farFromPlasma = true;
        return g; // dR=dZ=0.0 already
    }

    if(n < 3)
        throw std::runtime_error("JphiGrid::gradientAt: 1-2 genuinely interior "
                                  "nodes even at maxHalfWidth -- pathologically "
                                  "thin interior region near P (e.g. an X-point "
                                  "cusp tip), genuinely ambiguous unlike n==0");

    if(n < 6){
        g.usedLinearFallback = true;
        Eigen::MatrixXd A(n, 3);
        Eigen::VectorXd b(n);
        for(int m = 0; m < n; ++m){
            double r = field.R[idxI[m]]-RP, z = field.Z[idxK[m]]-ZP;
            A(m,0) = 1.0; A(m,1) = r; A(m,2) = z;
            b(m) = field.repaired[idxI[m]][idxK[m]];
        }
        Eigen::VectorXd coef = A.colPivHouseholderQr().solve(b);
        g.dR = coef(1); g.dZ = coef(2);
        return g;
    }

    Eigen::MatrixXd A(n, 6);
    Eigen::VectorXd b(n);
    for(int m = 0; m < n; ++m){
        double r = field.R[idxI[m]]-RP, z = field.Z[idxK[m]]-ZP;
        A(m,0) = 1.0; A(m,1) = r; A(m,2) = z; A(m,3) = r*r; A(m,4) = z*z; A(m,5) = r*z;
        b(m) = field.repaired[idxI[m]][idxK[m]];
    }
    Eigen::VectorXd coef = A.colPivHouseholderQr().solve(b);
    g.dR = coef(1); g.dZ = coef(2);
    return g;
}

} // namespace JphiGrid
#endif // GRID_JPHI_RECONSTRUCTION_H
