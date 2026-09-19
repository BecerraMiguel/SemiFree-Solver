/*=============================================================================
  UNIT TESTS: cutcell_geometry.h (Modulo 3: sub-punto + especificacion completa)
  =============================================================================
  Standalone (cutcell_geometry.h only needs `struct Point` to be visible).

  Compile:
    g++ -O3 -Isrc/third_party -Isrc/semifree_boundary -o build/test_cutcell_geometry \
        src/semifree_boundary/tests/test_cutcell_geometry.cpp -lm -std=c++11
  Run (from src/semifree_boundary/tests/fixtures): ../../../../build/test_cutcell_geometry
  =============================================================================*/
struct Point{ double R,Z; };
#include "cutcell_geometry.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace CutCell;

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

static vector<double> linspace(double a, double b, int n){
    vector<double> v(n);
    for(int i = 0; i < n; ++i) v[i] = a + (b-a)*i/(n-1);
    return v;
}

static vector<Point> loadPointsFile(const string& path){
    ifstream f(path);
    vector<Point> pts;
    double r, z;
    while(f >> r >> z) pts.push_back({r,z});
    return pts;
}

int main(){

    // ---- Sub-punto: clasificacion punto-en-poligono -----------------------
    {
        vector<Point> square = {{0,0},{1,0},{1,1},{0,1}};
        check(insidePolygon(0.5,0.5,square), "test_inside_polygon_square: (0.5,0.5) inside");
        check(!insidePolygon(1.5,0.5,square), "test_inside_polygon_square: (1.5,0.5) outside");

        vector<Point> tri = {{0,0},{2,0},{0,2}};
        check(insidePolygon(0.5,0.5,tri), "test_inside_polygon_triangle: (0.5,0.5) inside");
        check(!insidePolygon(1.5,1.5,tri), "test_inside_polygon_triangle: (1.5,1.5) outside");

        // Concave shape (anticipates an X-point cusp): an "L" polygon, whose
        // notch at (1.5,1.5) must classify as outside despite being inside
        // the L's bounding box.
        vector<Point> Lshape = {{0,0},{2,0},{2,1},{1,1},{1,2},{0,2}};
        check(insidePolygon(0.5,0.5,Lshape), "test_inside_polygon_concave: (0.5,0.5) inside the L");
        check(!insidePolygon(1.5,1.5,Lshape), "test_inside_polygon_concave: (1.5,1.5) in the notch, outside");
    }

    // ---- Sub-punto: shoelace area ------------------------------------------
    {
        vector<Point> unitSquare = {{0,0},{1,0},{1,1},{0,1}};
        checkClose(shoelaceArea(unitSquare), 1.0, 1e-14, "test_shoelace_area_unit_square");
        vector<Point> tri = {{0,0},{4,0},{0,3}};
        checkClose(shoelaceArea(tri), 6.0, 1e-14, "test_shoelace_area_triangle_4_3");
    }

    // ---- Sub-punto: Sutherland-Hodgman clipping ---------------------------
    {
        // Triangle (0,0),(2,0),(1,2) clipped against x>=1: keeps the right
        // half. The triangle is mirror-symmetric about x=1 (full area 2.0),
        // so the clipped half has exactly half that area, 1.0.
        vector<Point> tri = {{0,0},{2,0},{1,2}};
        vector<Point> half = clipHalfPlane(tri, 0, 1.0, true);
        checkClose(shoelaceArea(half), 1.0, 1e-12, "test_clip_half_plane_single_pass: right half area");

        // Same triangle clipped against the cell [0.5,1.5]x[0,1]. Compute
        // the expected area/centroid independently via the shoelace formula
        // on the analytically-derived clipped polygon: for x in [0.5,1.5],
        // the triangle's left edge is x=2y (from (0,0)-(1,2)) i.e. y=x/2,
        // and right edge x=2-... wait use direct edges: edge (0,0)-(1,2):
        // x=t, y=2t -> y=2x for x in[0,1]. edge (2,0)-(1,2): x=2-t,y=2t ->
        // y=2(2-x)=4-2x for x in [1,2]. Clipped to y<=1 and x in [0.5,1.5]:
        // region is bounded below by y=0, above by min(2x,4-2x,1), for
        // x in [0.5,1.5]. This is a hexagon; verified numerically below
        // against an independent fine Riemann sum instead of a hand
        // derivation (less error-prone for a hexagon).
        double rlo=0.5, rhi=1.5, zlo=0.0, zhi=1.0;
        ClipResult cr = clipCell(tri, rlo, rhi, zlo, zhi);
        // Independent reference: fine Riemann sum with point-in-triangle test.
        int N = 4000;
        long inCount = 0;
        double sumR=0, sumZ=0;
        for(int a=0;a<N;++a){
            for(int b=0;b<N;++b){
                double x = rlo + (a+0.5)*(rhi-rlo)/N;
                double y = zlo + (b+0.5)*(zhi-zlo)/N;
                if(insidePolygon(x,y,tri)){ inCount++; sumR+=x; sumZ+=y; }
            }
        }
        double cellArea = (rhi-rlo)*(zhi-zlo)/((double)N*N);
        double refArea = inCount*cellArea;
        double refCx = sumR/inCount, refCz = sumZ/inCount;
        checkClose(cr.area, refArea, 2e-3, "test_sutherland_hodgman_full_cell: area vs fine Riemann reference");
        checkClose(cr.centroidR, refCx, 2e-3, "test_sutherland_hodgman_full_cell: centroid R vs fine Riemann reference");
        checkClose(cr.centroidZ, refCz, 2e-3, "test_sutherland_hodgman_full_cell: centroid Z vs fine Riemann reference");

        // Degenerate: cell entirely outside the polygon.
        ClipResult crOut = clipCell(tri, 10.0, 11.0, 10.0, 11.0);
        checkClose(crOut.area, 0.0, 1e-14, "test_clip_cell_degenerate_returns_zero_area");

        // Full containment: cell entirely inside a big triangle.
        vector<Point> bigTri = {{-10,-10},{10,-10},{0,10}};
        ClipResult crIn = clipCell(bigTri, -1.0, 1.0, -1.0, 1.0);
        checkClose(crIn.area, 4.0, 1e-12, "test_clip_cell_full_containment: area == full cell area");
    }

    // ---- Sub-punto: deteccion de celdas candidatas -------------------------
    {
        // An 8x8 unit-spaced grid on [0,7]x[0,7] (nodes at integers, cell
        // edges at half-integers). Square [0.6,6.4]^2: node (3,3)'s cell
        // [2.5,3.5]^2 sits with a full extra cell of margin (1.9) from the
        // nearest edge -- far enough to survive both direct rasterization
        // AND the +1-cell dilation, a genuine deep-interior cell. Node
        // (7,7)'s cell [6.5,7.5]^2 is symmetric-opposite, genuinely
        // deep-exterior.
        vector<double> R = linspace(0.0,7.0,8);
        vector<double> Z = linspace(0.0,7.0,8);
        vector<Point> sq = {{0.6,0.6},{6.4,0.6},{6.4,6.4},{0.6,6.4}};
        auto mask = edgeCellMask(sq, R, Z);
        check(!mask[7][7], "test_edge_cell_mask_excludes_far_cell: deep-exterior (7,7) not a candidate");
        check(!mask[3][3], "test_edge_cell_mask_excludes_deep_interior_cell: deep-interior (3,3) not a candidate");
        // At least one boundary-adjacent node should be flagged.
        bool anyCandidate = false;
        for(int i=0;i<4;++i) for(int k=0;k<4;++k) if(mask[i][k]) anyCandidate = true;
        check(anyCandidate, "test_edge_cell_mask_flags_some_boundary_adjacent_cell");
    }

    // ---- Sub-punto: clasificacion completa en malla conocida --------------
    {
        // Square R in [0.3,0.7], Z in [0.3,0.7] on a 5x5 grid over [0,1]^2
        // (dR=dZ=0.25, nodes at 0,0.25,0.5,0.75,1.0). Classify by hand:
        //   node (0.5,0.5): deep inside -> Interior, area=full=0.0625
        //   node (0,0): far outside -> Exterior
        //   node (0.25,0.25): cell [0.125,0.375]x[0.125,0.375] overlaps
        //     [0.3,0.7]^2 in [0.3,0.375]x[0.3,0.375] -> Cut, area=0.075^2=0.005625
        vector<double> R = linspace(0.0,1.0,5);
        vector<double> Z = linspace(0.0,1.0,5);
        vector<Point> sq = {{0.3,0.3},{0.7,0.3},{0.7,0.7},{0.3,0.7}};
        Geometry g = build(sq, R, Z);
        double full = g.dR*g.dZ;
        checkClose(full, 0.0625, 1e-12, "test_classification_square_on_grid: full cell area sanity");
        check(g.cell[2][2].state == CellState::Interior, "test_classification_square_on_grid: (0.5,0.5) Interior");
        checkClose(g.cell[2][2].area, full, 1e-12, "test_classification_square_on_grid: (0.5,0.5) area==full");
        check(g.cell[0][0].state == CellState::Exterior, "test_classification_square_on_grid: (0,0) Exterior");
        check(g.cell[1][1].state == CellState::Cut, "test_classification_square_on_grid: (0.25,0.25) Cut");
        checkClose(g.cell[1][1].area, 0.075*0.075, 1e-10, "test_classification_square_on_grid: (0.25,0.25) area");
        checkClose(g.areaSum, shoelaceArea(sq), 1e-10, "test_classification_square_on_grid: areaSum == shoelace(sq)");
    }

    // ---- Especificacion completa (nivel 2): regresion geometrica ----------
    // Acceptance test #1 of the briefing: for the smooth Soloviev D-shape
    // (no X-point), the C++ cut-cell classification must reproduce the
    // exact polygon area (Dshape.txt) to ~1e-10 or better relative.
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        check(boundary.size() > 50, "level2 regression: loaded Dshape.txt (sanity: >50 points)");
        if(boundary.size() > 50){
            double r_min=0.15, r_max=1.65, z_min=-1.3, z_max=1.3;
            vector<double> R76 = linspace(r_min, r_max, 76);
            vector<double> Z131 = linspace(z_min, z_max, 131);
            Geometry g76 = build(boundary, R76, Z131);
            checkClose(g76.areaSum, g76.areaExact, 1e-10,
                       "level2 regression: areaSum matches shoelace(Dshape.txt) at 76x131");

            vector<double> R151 = linspace(r_min, r_max, 151);
            vector<double> Z261 = linspace(z_min, z_max, 261);
            Geometry g151 = build(boundary, R151, Z261);
            checkClose(g151.areaSum, g151.areaExact, 1e-10,
                       "level2 regression: areaSum matches shoelace(Dshape.txt) at 151x261");
        }
    }

    // ---- Especificacion completa (nivel 2): orden de convergencia --------
    // Case A of CUT_CELL_VALIDATION.tex: integrand f=exp(0.5R+0.3Z), no
    // singularity. cut-cell should recover order ~2.0 (naive order ~1.0-1.1).
    // Self-contained 3-level Richardson ratio test (no external reference
    // needed): order = log2(|I(h)-I(h/2)| / |I(h/2)-I(h/4)|).
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        double r_min=0.15, r_max=1.65, z_min=-1.3, z_max=1.3;
        auto f = [](double r, double z){ return exp(0.5*r+0.3*z); };

        int sizesR[3] = {76, 151, 301};
        int sizesZ[3] = {131, 261, 521};
        double Icut[3], Inaive[3];
        for(int lvl = 0; lvl < 3; ++lvl){
            vector<double> R = linspace(r_min, r_max, sizesR[lvl]);
            vector<double> Z = linspace(z_min, z_max, sizesZ[lvl]);
            Geometry g = build(boundary, R, Z);
            vector<vector<double>> fNodes(R.size(), vector<double>(Z.size()));
            for(size_t i=0;i<R.size();++i)
                for(size_t k=0;k<Z.size();++k)
                    fNodes[i][k] = g.insideNode[i][k] ? f(R[i],Z[k]) : 0.0;
            Icut[lvl] = integrate(g, fNodes, f);

            // naive: full-mesh sum, node mask only, weight dR*dZ everywhere
            // (mirrors the CURRENT PsiContributionPlasma-style method).
            double dR = R[1]-R[0], dZ = Z[1]-Z[0];
            double s = 0.0;
            for(size_t i=0;i<R.size();++i)
                for(size_t k=0;k<Z.size();++k)
                    if(g.insideNode[i][k]) s += f(R[i],Z[k]);
            Inaive[lvl] = s*dR*dZ;
        }
        double orderCut = log2(fabs(Icut[0]-Icut[1]) / fabs(Icut[1]-Icut[2]));
        double orderNaive = log2(fabs(Inaive[0]-Inaive[1]) / fabs(Inaive[1]-Inaive[2]));
        printf("[INFO] Case A convergence order: cut-cell=%.3f naive=%.3f (Python 5-level reference: naive~1.09-1.11)\n",
               orderCut, orderNaive);
        check(orderCut > 1.8 && orderCut < 2.3, "level2 orderCut ~2.0 (cut-cell recovers second order)");
        // The naive (0/1 node-mask) method's order is NOT asserted against a
        // fixed bound here: a 3-level Richardson ratio is a noisy estimator
        // for a first-order "staircase" method on a curved boundary (the
        // documented Python reference used a 5-level log-log LEAST-SQUARES
        // fit, not a 3-level ratio, to get a stable ~1.09-1.11). What matters
        // for this test is the comparison: cut-cell's order must land much
        // closer to the true value 2.0 than naive's does.
        check(fabs(orderCut-2.0) < fabs(orderNaive-2.0),
              "level2: cut-cell order is markedly closer to 2.0 than naive's");
    }

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
