/*=============================================================================
  UNIT TESTS: singularity_subtraction.h (Modulo 5: sub-punto + especificacion
  completa)
  =============================================================================
  Uses the #define UNIT_TESTING + #include "SemiFree_Solver.cpp" pattern
  (test_regularization.cpp's own pattern) to get real GPsi/GBr/GBz.

  Compile:
    g++ -O3 -Isrc/third_party -Isrc/semifree_boundary -o build/test_singularity_subtraction \
        src/semifree_boundary/tests/test_singularity_subtraction.cpp -lm -std=c++11
  Run (from src/semifree_boundary/tests/fixtures): ../../../../build/test_singularity_subtraction
  =============================================================================*/
#define UNIT_TESTING
#include "SemiFree_Solver.cpp"
#include "cutcell_geometry.h"
#include "grid_jphi_reconstruction.h"
#include "singularity_kernels.h"
#include "adaptive_quadrature.h"
#include "singularity_subtraction.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>

using namespace CutCell;
using namespace JphiGrid;
using namespace Singularity;

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

static vector<Point> loadPointsFile(const string& path){
    ifstream f(path);
    vector<Point> pts; double r,z;
    while(f >> r >> z) pts.push_back({r,z});
    return pts;
}

static vector<double> linspace(double a, double b, int n){
    vector<double> v(n);
    for(int i = 0; i < n; ++i) v[i] = a + (b-a)*i/(n-1);
    return v;
}

int main(){

    // ---- Sub-punto: chequeo de estrellado (diagnostico) --------------------
    {
        // Regular octagon, P at the center: every ray crosses exactly once.
        vector<Point> octagon;
        for(int i = 0; i < 8; ++i){
            double a = 2.0*M_PI*i/8.0;
            octagon.push_back({2.0+cos(a), 2.0+sin(a)});
        }
        StarShapedReport rep = checkStarShaped(octagon, 2.0, 2.0, 500, 5.0);
        check(rep.nViolations == 0, "test_check_star_shaped_convex_polygon_is_star_shaped");

        // A narrow "V" wedge cusp: P placed just outside the cusp tip, along
        // the bisector, so a ray roughly along either flank crosses the
        // double-branch twice before reaching the wedge interior.
        vector<Point> wedge = {
            {0.0, 0.0}, {1.0, 0.05}, {3.0, 3.0}, {-1.0, 3.0}, {-1.0, 0.05}
        };
        // P near the tip (0,0), slightly outside along -Z.
        StarShapedReport rep2 = checkStarShaped(wedge, 0.0, -0.02, 500, 5.0);
        check(rep2.nViolations > 0, "test_check_star_shaped_concave_polygon_detects_violation");
    }

    // ---- Sub-punto: row intervals -------------------------------------------
    {
        vector<Point> rect = {{0,0},{2,0},{2,1},{0,1}};
        auto iv = rowIntervals(rect, 0.5);
        check(iv.size() == 1, "test_row_intervals_matches_hand_calc_rectangle: one interval");
        if(iv.size() == 1){
            checkClose(iv[0].first, 0.0, 1e-12, "test_row_intervals_matches_hand_calc_rectangle: left edge");
            checkClose(iv[0].second, 2.0, 1e-12, "test_row_intervals_matches_hand_calc_rectangle: right edge");
        }

        // "M" shaped polygon crossed by a row that hits it 4 times (two
        // separate humps): (0,0)-(1,2)-(2,0.5)-(3,2)-(4,0)-(0,0), row z=1.
        vector<Point> mshape = {{0,0},{1,2},{2,0.5},{3,2},{4,0}};
        auto iv2 = rowIntervals(mshape, 1.0);
        check(iv2.size() == 2, "test_row_intervals_handles_multiple_crossings: two intervals at z=1");
    }

    // ---- Sub-punto: verticalmente simple ------------------------------------
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        check(verifyVerticallySimple(boundary), "test_verify_vertically_simple_true_for_dshape");
        vector<Point> mshape = {{0,0},{1,2},{2,0.5},{3,2},{4,0}};
        check(!verifyVerticallySimple(mshape), "test_verify_vertically_simple_false_for_M_shape");
    }

    // ---- Sub-punto: constantes geometricas vs forma cerrada (rectangulo) --
    {
        // Rectangle [-2,2]x[-1,1], P at the exact center (0,0): by odd
        // symmetry, dipZ, dipR, dZ_R, dR_Z must vanish EXACTLY (integrand
        // odd about P over a domain symmetric about P).
        vector<Point> rect = {{-2,-1},{2,-1},{2,1},{-2,1}};
        GeomConstants C = geomConstantsRowScan(0.0, 0.0, rect, nullptr, 1e-10);
        checkClose(C.dipZ, 0.0, 1.0, "test_geom_constants_rowscan_symmetry: dipZ ~ 0 (odd integrand)");
        check(fabs(C.dipZ) < 1e-8, "test_geom_constants_rowscan_symmetry: dipZ magnitude < 1e-8");
        check(fabs(C.dipR) < 1e-8, "test_geom_constants_rowscan_symmetry: dipR magnitude < 1e-8");
        check(fabs(C.dZ_R) < 1e-8, "test_geom_constants_rowscan_symmetry: dZ_R magnitude < 1e-8");
        check(fabs(C.dR_Z) < 1e-8, "test_geom_constants_rowscan_symmetry: dR_Z magnitude < 1e-8");
        // Cross-check the non-symmetric ones against a much tighter
        // "quasi-analytic" row-scan (epsrel=1e-12) as an independent
        // higher-precision reference.
        GeomConstants Cref = geomConstantsRowScan(0.0, 0.0, rect, nullptr, 1e-12);
        checkClose(C.lnD,  Cref.lnD,  1e-6, "test_geom_constants_rowscan: lnD vs tighter reference");
        checkClose(C.dR_R, Cref.dR_R, 1e-6, "test_geom_constants_rowscan: dR_R vs tighter reference");
        checkClose(C.dZ_Z, Cref.dZ_Z, 1e-6, "test_geom_constants_rowscan: dZ_Z vs tighter reference");
    }

    // ---- Sub-punto: VSimpleExtent vs ruta general --------------------------
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        VSimpleExtent ve = buildVSimpleExtent(boundary);
        double RP=1.1, ZP=0.4;
        auto integrand = [&](double R,double Z){
            double d2 = max((R-RP)*(R-RP)+(Z-ZP)*(Z-ZP), 1e-300);
            return -0.5*log(d2);
        };
        double withVSimple = nestedQuadRowScan(integrand, RP, ZP, boundary, &ve, 1e-9);
        double general     = nestedQuadRowScan(integrand, RP, ZP, boundary, nullptr, 1e-9);
        checkClose(withVSimple, general, 1e-6, "test_vsimple_extent_matches_general_rowscan");
    }

    // ---- Sub-punto: switch orden 1 vs 2 -------------------------------------
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        int npr=76, npz=131;
        vector<double> R = linspace(0.15,1.65,npr), Z = linspace(-1.3,1.3,npz);
        // Linear Jphi with a genuinely nonzero gradient in both directions.
        vector<vector<double>> Jt(npr, vector<double>(npz));
        for(int i=0;i<npr;++i) for(int k=0;k<npz;++k) Jt[i][k] = 1.0 + 2.0*R[i] + 3.0*Z[k];
        PlasmaContext ctx = build(boundary, Jt, R, Z);
        double RP=0.923, ZP=0.317;
        Result r1 = evaluateAll(ctx, R, Z, RP, ZP, 1);
        Result r2 = evaluateAll(ctx, R, Z, RP, ZP, 2);
        check(fabs(r1.psi-r2.psi) > 1e-8, "test_order1_vs_order2_switch: psi differs between order 1 and 2 (nonzero gradient case)");
        check(fabs(r1.br-r2.br) > 1e-8, "test_order1_vs_order2_switch: br differs between order 1 and 2");
    }

    // ---- Sub-punto: evaluateAll consistente con psiAt/brAt/bzAt ------------
    {
        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");
        int npr=76, npz=131;
        vector<double> R = linspace(0.15,1.65,npr), Z = linspace(-1.3,1.3,npz);
        vector<vector<double>> Jt(npr, vector<double>(npz));
        for(int i=0;i<npr;++i) for(int k=0;k<npz;++k) Jt[i][k] = 1.0 + 2.0*R[i] + 3.0*Z[k];
        PlasmaContext ctx = build(boundary, Jt, R, Z);
        double RP=1.2, ZP=0.4;
        Result all = evaluateAll(ctx, R, Z, RP, ZP, 2);
        checkClose(psiAt(ctx,R,Z,RP,ZP,2), all.psi, 1e-14, "test_evaluateAll_matches_psiAt");
        checkClose(brAt(ctx,R,Z,RP,ZP,2), all.br, 1e-14, "test_evaluateAll_matches_brAt");
        checkClose(bzAt(ctx,R,Z,RP,ZP,2), all.bz, 1e-14, "test_evaluateAll_matches_bzAt");
    }

    // ---- Especificacion completa (nivel 2) ---------------------------------
    // Uses the ANALYTIC Soloviev Jphi (isolating Module C from Module B, as
    // verify_xpoint_endtoend.py's "Track A" does): Jphi=R*P'+FF'/(mu0*R),
    // dJphi/dR = P'-FF'/(mu0*R^2), dJphi/dZ=0.
    {
        const double MU0 = 4.0*M_PI*1e-7;
        const double A1 = -0.7491216684475385, A2 = 0.24684511262846784;
        const double Pprime = -A1/MU0, FFprime = A2;
        auto Jfun = [&](double R,double Z){ return R*Pprime + FFprime/(MU0*R); };
        auto dJdR = [&](double R){ return Pprime - FFprime/(MU0*R*R); };

        vector<Point> boundary = loadPointsFile("cases/soloviev/Dshape.txt");

        struct Pt { const char* name; double R,Z; double ref_psi, ref_br, ref_bz; };
        // results_boundary.json reference values (code-normalized units,
        // Ro=Lo=0.9 -- normalize R,Z by Lo here so the whole pipeline runs
        // in the SAME units the reference values were computed in).
        Pt pts[] = {
            {"LCFS_aligned_1.5_0", 1.5, 0.0, 233506.76162670413, 2.5363746678592486e-11, -116932.69516480943},
            {"LCFS_theta0.7_offgrid", 1.2089, 0.7344, 167099.87562089055, 163366.49504038828, -4693.841206810044},
            {"interior_control_0.923", 0.923, 0.317, 178096.54325252902, 55872.900112288444, 248587.87000006315},
        };

        int npr=301, npz=521; // finest ladder level for the level-2 absolute-value check
        vector<double> R = linspace(0.15,1.65,npr), Z = linspace(-1.3,1.3,npz);
        vector<vector<double>> Jt(npr, vector<double>(npz));
        for(int i=0;i<npr;++i) for(int k=0;k<npz;++k) Jt[i][k] = Jfun(R[i],Z[k]);
        PlasmaContext ctx = build(boundary, Jt, R, Z);

        for(auto& p : pts){
            auto JfunL = [&](double r,double z){ return Jfun(r,z); };
            Result res = evaluateAllWithJfun(ctx, R, Z, p.R, p.Z, JfunL, Jfun(p.R,p.Z), dJdR(p.R), 0.0, 2);
            string label = string("level2 absolute value @ ") + p.name;
            checkClose(res.psi, p.ref_psi, 5e-3, (label+" psi_p").c_str());
            if(fabs(p.ref_br) > 1e-6) checkClose(res.br, p.ref_br, 5e-3, (label+" B_R").c_str());
            checkClose(res.bz, p.ref_bz, 5e-3, (label+" B_Z").c_str());
        }

        // Convergence order (crudo/sustr1/sustr2) at interior_control_0.923,
        // vs SINGULAR_KERNEL_VALIDATION.tex: psi_p crudo~0.85, sustr2~1.99.
        {
            // Self-contained 3-level RICHARDSON RATIO (not a comparison
            // against the external results_boundary.json reference): found
            // that this method's own value is already stable to ~0.06%
            // across ALL three mesh levels (i.e. it has already converged
            // well within that band), so comparing each level's error
            // against the external reference just measures that fixed
            // ~0.06% offset (itself presumably the external reference's own
            // finite quadrature precision) rather than this method's
            // mesh-refinement order -- exactly the same pitfall as the
            // naive-method order estimate in test_cutcell_geometry.cpp.
            // I(h),I(h/2),I(h/4) from this method's OWN 3 levels isolates
            // the actual convergence rate instead.
            double RP=0.923, ZP=0.317;
            int sizesR[3]={76,151,301}, sizesZ[3]={131,261,521};
            double valCrude[3], valSub2[3];
            for(int lvl=0; lvl<3; ++lvl){
                int nr=sizesR[lvl], nz=sizesZ[lvl];
                vector<double> Rl = linspace(0.15,1.65,nr), Zl = linspace(-1.3,1.3,nz);
                vector<vector<double>> Jl(nr, vector<double>(nz));
                for(int i=0;i<nr;++i) for(int k=0;k<nz;++k) Jl[i][k]=Jfun(Rl[i],Zl[k]);
                PlasmaContext ctxl = build(boundary, Jl, Rl, Zl);
                auto Jl2 = [&](double r,double z){ return Jfun(r,z); };
                // crudo = Q_h[J*raw kernel], no subtraction at all
                auto kpsi=[&](double r,double z){ return GPsi(r,z,RP,ZP); };
                valCrude[lvl] = cutCellIntegral(ctxl, Rl, Zl, [&](double r,double z){ return Jl2(r,z)*kpsi(r,z); });
                valSub2[lvl] = evaluateAllWithJfun(ctxl, Rl, Zl, RP, ZP, Jl2, Jfun(RP,ZP), dJdR(RP), 0.0, 2).psi;
            }
            double orderCrude = log2(fabs(valCrude[0]-valCrude[1]) / fabs(valCrude[1]-valCrude[2]));
            double orderSub2  = log2(fabs(valSub2[0]-valSub2[1])   / fabs(valSub2[1]-valSub2[2]));
            printf("[INFO] psi_p order at interior_control_0.923: crudo=%.3f sustr2=%.3f (Python ref: 0.85 / 1.99)\n",
                   orderCrude, orderSub2);
            check(orderSub2 > 1.5, "level2 order: sustr2 clearly recovers a high order (>1.5) at interior point");
            check(orderSub2 > orderCrude, "level2 order: sustr2 order exceeds crudo order");
        }
    }

    printf("\n=== Summary: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
