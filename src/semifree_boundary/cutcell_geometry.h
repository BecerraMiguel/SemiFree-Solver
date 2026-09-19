/*=============================================================================
  CUT-CELL GEOMETRY: exact area/centroid cell classification for a boundary
  given as a polygon (list of Points, with or without an X-point cusp).
  =============================================================================
  Port of tools/cutcell/xpoint_geometry.py (GeneralCutGeometry, clip_cell,
  _edge_cell_mask) -- the general-polygon variant, since in production Gamma
  is always a list of points (docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf
  section 4). Each mesh cell (centered on a node, [Ri +/- dR/2] x
  [Zk +/- dZ/2]) is classified Interior / Exterior / Cut and integrated
  accordingly:
    - Interior: full weight dR*dZ, sampled at the node (identical to the
      current naive method there).
    - Exterior: contributes zero.
    - Cut: weight = exact area of cell INTERSECT polygon, evaluated at the
      centroid of that intersection.
  The clipping works entirely in Cartesian (R,Z) coordinates, cell by cell --
  it never reduces area to a polar integral centered on a point P, so it has
  no "star-shaped w.r.t. P" requirement (that condition only appears later,
  in singularity_subtraction.h's exact geometric constants).

  Requires `struct Point{double R,Z;};` to already be visible (defined in
  SemiFree_Solver.cpp before this file is included).
  =============================================================================*/
#ifndef CUTCELL_GEOMETRY_H
#define CUTCELL_GEOMETRY_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <functional>

namespace CutCell {

enum class CellState : int8_t { Exterior = 0, Interior = 1, Cut = 2 };

struct CellInfo {
    CellState state = CellState::Exterior;
    double area = 0.0;       // area(cell INTERSECT boundary)
    double centroidR = 0.0;  // quadrature evaluation point for this cell
    double centroidZ = 0.0;
};

// ---------------------------------------------------------------------------
// Point-in-polygon classification: ported VERBATIM from
// src/grad_shafranov/GS_solver.h lines 860-870 (inside_polygon), adapted
// from two parallel vector<double> to vector<Point> -- logic unchanged.
// GS_solver.h itself is NOT included here (it has no include guard/namespace
// and carries ~15 mutable globals of the GS solver's own state; linking it
// directly into SemiFree_Solver.cpp would risk silent global-name collisions
// -- see the design decisions recorded for this feature). Cross-checked for
// bit-identical output against GS_solver.h's own inside_polygon in
// test_inside_polygon_matches_gs_solver.cpp.
inline bool insidePolygon(double r0, double z0, const std::vector<Point>& poly){
    int M = (int)poly.size();
    bool in = false;
    for(int j = 0, k = M-1; j < M; k = j++){
        bool cross = ((poly[j].Z > z0) != (poly[k].Z > z0)) &&
                     (r0 < (poly[k].R-poly[j].R)*(z0-poly[j].Z)/(poly[k].Z-poly[j].Z) + poly[j].R);
        if(cross) in = !in;
    }
    return in;
}

// ---------------------------------------------------------------------------
// Shoelace area/centroid
// ---------------------------------------------------------------------------

// 2x the SIGNED area (no fabs, no 1/2) -- kept internal since callers that
// need the centroid must divide by the SIGNED (not absolute) area.
inline double signedArea2x(const std::vector<Point>& poly){
    double sum = 0.0;
    int n = (int)poly.size();
    for(int i = 0; i < n; ++i){
        const Point& cur = poly[i];
        const Point& nxt = poly[(i+1)%n];
        sum += cur.R*nxt.Z - nxt.R*cur.Z;
    }
    return sum;
}

// |signed area| via Gauss's shoelace formula.
inline double shoelaceArea(const std::vector<Point>& poly){
    return 0.5*std::fabs(signedArea2x(poly));
}

// Centroid via the standard polygon-centroid formula. Takes the SIGNED area
// (as returned by 0.5*signedArea2x, not shoelaceArea's absolute value) so
// callers that already computed it (clipCell) need not recompute it.
// Degenerate (near-zero-area, e.g. collinear) polygons fall back to the
// arithmetic mean of vertices rather than dividing by ~0.
inline void polygonCentroid(const std::vector<Point>& poly, double signedArea,
                             double& cx, double& cz){
    int n = (int)poly.size();
    if(std::fabs(signedArea) < 1e-300){
        double sr = 0.0, sz = 0.0;
        for(const Point& p : poly){ sr += p.R; sz += p.Z; }
        cx = (n > 0) ? sr/n : 0.0;
        cz = (n > 0) ? sz/n : 0.0;
        return;
    }
    double sxr = 0.0, sxz = 0.0;
    for(int i = 0; i < n; ++i){
        const Point& cur = poly[i];
        const Point& nxt = poly[(i+1)%n];
        double cross = cur.R*nxt.Z - nxt.R*cur.Z;
        sxr += (cur.R+nxt.R)*cross;
        sxz += (cur.Z+nxt.Z)*cross;
    }
    cx = sxr/(6.0*signedArea);
    cz = sxz/(6.0*signedArea);
}

// ---------------------------------------------------------------------------
// Sutherland-Hodgman clipping against the (always-convex) cell rectangle
// ---------------------------------------------------------------------------

// One pass against a single half-plane: axis=0 clips on R, axis=1 on Z;
// keepGreaterEqual selects which side of `value` is kept (true: >=value,
// false: <=value). Standard Sutherland-Hodgman: walk the (possibly
// non-convex, but simple) subject polygon's edges, keep vertices on the
// "inside" side, and insert the edge/line intersection wherever an edge
// crosses the half-plane boundary.
inline std::vector<Point> clipHalfPlane(const std::vector<Point>& poly, int axis,
                                         double value, bool keepGreaterEqual){
    std::vector<Point> out;
    int n = (int)poly.size();
    if(n == 0) return out;
    out.reserve(n+2);
    for(int i = 0; i < n; ++i){
        const Point& cur = poly[i];
        const Point& nxt = poly[(i+1)%n];
        double curVal = (axis == 0) ? cur.R : cur.Z;
        double nxtVal = (axis == 0) ? nxt.R : nxt.Z;
        bool curIn = keepGreaterEqual ? (curVal >= value) : (curVal <= value);
        bool nxtIn = keepGreaterEqual ? (nxtVal >= value) : (nxtVal <= value);
        if(curIn) out.push_back(cur);
        if(curIn != nxtIn){
            double t = (value - curVal) / (nxtVal - curVal);
            Point ip;
            ip.R = cur.R + t*(nxt.R - cur.R);
            ip.Z = cur.Z + t*(nxt.Z - cur.Z);
            out.push_back(ip);
        }
    }
    return out;
}

struct ClipResult { double area, centroidR, centroidZ; };

// Sutherland-Hodgman, four passes (R>=rlo, R<=rhi, Z>=zlo, Z<=zhi), against
// the cell's rectangle. Valid for a simple (possibly non-convex) subject
// polygon as long as cell INTERSECT polygon stays connected, which holds
// cell-by-cell (empirically verified for the X-point scenarios of this
// project in xpoint_geometry.py). If any pass leaves fewer than 3 vertices,
// the intersection is empty -- return zero area immediately rather than
// feeding a degenerate polygon to the shoelace formula.
inline ClipResult clipCell(const std::vector<Point>& subject,
                            double rlo, double rhi, double zlo, double zhi){
    std::vector<Point> poly = subject;
    poly = clipHalfPlane(poly, 0, rlo, true);
    if((int)poly.size() >= 3) poly = clipHalfPlane(poly, 0, rhi, false);
    if((int)poly.size() >= 3) poly = clipHalfPlane(poly, 1, zlo, true);
    if((int)poly.size() >= 3) poly = clipHalfPlane(poly, 1, zhi, false);

    if((int)poly.size() < 3){
        ClipResult cr;
        cr.area = 0.0;
        cr.centroidR = 0.5*(rlo+rhi);
        cr.centroidZ = 0.5*(zlo+zhi);
        return cr;
    }

    double signedA = 0.5*signedArea2x(poly);
    double cx, cz;
    polygonCentroid(poly, signedA, cx, cz);

    ClipResult cr;
    cr.area = std::fabs(signedA);
    cr.centroidR = cx;
    cr.centroidZ = cz;
    return cr;
}

// ---------------------------------------------------------------------------
// Candidate-cell detection (mirrors xpoint_geometry.GeneralCutGeometry.
// _edge_cell_mask): rasterizing each polygon edge over the mesh (at steps of
// 0.25*min(dR,dZ)) and dilating the result 1 cell in each cardinal
// direction, instead of testing every cell against the full polygon
// (O(N*M), M = number of vertices), reduces the cells that actually need
// clipping to O(sqrt(N)) (the discretized perimeter).
// ---------------------------------------------------------------------------
inline std::vector<std::vector<bool>> edgeCellMask(const std::vector<Point>& boundary,
                                                    const std::vector<double>& R,
                                                    const std::vector<double>& Z){
    int npr = (int)R.size(), npz = (int)Z.size();
    double dR = R[1]-R[0], dZ = Z[1]-Z[0];
    double Redge0 = R[0] - 0.5*dR;
    double Zedge0 = Z[0] - 0.5*dZ;
    double step = 0.25*std::min(dR, dZ);

    std::vector<std::vector<bool>> cand(npr, std::vector<bool>(npz, false));
    int n = (int)boundary.size();
    for(int e = 0; e < n; ++e){
        const Point& a = boundary[e];
        const Point& b = boundary[(e+1)%n];
        double segLen = std::hypot(b.R-a.R, b.Z-a.Z);
        int nsub = std::max(2, (int)std::ceil(segLen/step));
        for(int t = 0; t <= nsub; ++t){
            double frac = (double)t/(double)nsub;
            double rr = a.R + frac*(b.R-a.R);
            double zz = a.Z + frac*(b.Z-a.Z);
            int ii = (int)std::floor((rr-Redge0)/dR);
            int kk = (int)std::floor((zz-Zedge0)/dZ);
            if(ii >= 0 && ii < npr && kk >= 0 && kk < npz) cand[ii][kk] = true;
        }
    }

    std::vector<std::vector<bool>> dil = cand;
    for(int i = 0; i < npr; ++i){
        for(int k = 0; k < npz; ++k){
            if(!cand[i][k]) continue;
            if(i > 0)       dil[i-1][k] = true;
            if(i < npr-1)   dil[i+1][k] = true;
            if(k > 0)       dil[i][k-1] = true;
            if(k < npz-1)   dil[i][k+1] = true;
        }
    }
    return dil;
}

// ---------------------------------------------------------------------------
// Full mesh geometry: classify every cell exactly once per mesh (cached and
// reused for every subsequent evaluation on that mesh -- see the
// performance note in docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf
// section 7).
// ---------------------------------------------------------------------------
struct Geometry {
    int npr = 0, npz = 0;
    double dR = 0.0, dZ = 0.0;
    std::vector<std::vector<CellInfo>> cell;      // [i][k], same layout as Jphi[i][k]
    std::vector<std::vector<bool>> insideNode;    // node inside the polygon
    double areaSum = 0.0, areaExact = 0.0;        // diagnostic (acceptance test #1)
    int nCut = 0, nCandidates = 0;
};

inline Geometry build(const std::vector<Point>& boundary,
                       const std::vector<double>& R, const std::vector<double>& Z){
    Geometry g;
    g.npr = (int)R.size();
    g.npz = (int)Z.size();
    g.dR = R[1]-R[0];
    g.dZ = Z[1]-Z[0];
    double full = g.dR*g.dZ;
    double tol = 1e-10*full;

    g.insideNode.assign(g.npr, std::vector<bool>(g.npz, false));
    g.cell.assign(g.npr, std::vector<CellInfo>(g.npz));

    for(int i = 0; i < g.npr; ++i)
        for(int k = 0; k < g.npz; ++k)
            g.insideNode[i][k] = insidePolygon(R[i], Z[k], boundary);

    std::vector<std::vector<bool>> cand = edgeCellMask(boundary, R, Z);

    double areaSum = 0.0;
    g.nCandidates = 0;
    g.nCut = 0;
    for(int i = 0; i < g.npr; ++i){
        for(int k = 0; k < g.npz; ++k){
            CellInfo& c = g.cell[i][k];
            if(cand[i][k]){
                g.nCandidates++;
                double rlo = R[i]-0.5*g.dR, rhi = R[i]+0.5*g.dR;
                double zlo = Z[k]-0.5*g.dZ, zhi = Z[k]+0.5*g.dZ;
                ClipResult cr = clipCell(boundary, rlo, rhi, zlo, zhi);
                if(cr.area <= tol){
                    c.state = CellState::Exterior; c.area = 0.0;
                    c.centroidR = R[i]; c.centroidZ = Z[k];
                } else if(cr.area >= full - tol){
                    c.state = CellState::Interior; c.area = full;
                    c.centroidR = R[i]; c.centroidZ = Z[k];
                } else {
                    c.state = CellState::Cut; c.area = cr.area;
                    c.centroidR = cr.centroidR; c.centroidZ = cr.centroidZ;
                    g.nCut++;
                }
            } else if(g.insideNode[i][k]){
                c.state = CellState::Interior; c.area = full;
                c.centroidR = R[i]; c.centroidZ = Z[k];
            } else {
                c.state = CellState::Exterior; c.area = 0.0;
                c.centroidR = R[i]; c.centroidZ = Z[k];
            }
            areaSum += c.area;
        }
    }
    g.areaSum = areaSum;
    g.areaExact = shoelaceArea(boundary);
    return g;
}

// Cut-cell quadrature: sum over interior cells (full weight, sampled at the
// node -- identical to the current naive method there) plus cut cells
// (exact intersection area, sampled at the centroid of that intersection).
// `valuesAtNodes` supplies the integrand at every mesh node (used for
// Interior cells); `valueAt` supplies it at an arbitrary (r,z) (used for Cut
// cells' centroids, which are in general not mesh nodes).
inline double integrate(const Geometry& geom,
                         const std::vector<std::vector<double>>& valuesAtNodes,
                         const std::function<double(double,double)>& valueAt){
    double sum = 0.0;
    for(int i = 0; i < geom.npr; ++i){
        for(int k = 0; k < geom.npz; ++k){
            const CellInfo& c = geom.cell[i][k];
            if(c.state == CellState::Exterior) continue;
            if(c.state == CellState::Interior) sum += valuesAtNodes[i][k]*c.area;
            else sum += c.area * valueAt(c.centroidR, c.centroidZ);
        }
    }
    return sum;
}

} // namespace CutCell
#endif // CUTCELL_GEOMETRY_H
