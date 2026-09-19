#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>



#include <cstdlib>
#include <time.h>
#include <chrono>
#include <iomanip>
#include "json.hpp"
#include <Eigen/Dense>
#include "regularization.h"

/*-----------------------------------------------------------------------------------------------------------*/
using namespace std;
using json = nlohmann::json;
/*-----------------------------------------------------------------------------------------------------------*/
struct Point{
public:
    double R,Z;
};
/*-------------------------------------------------------------------------------------------------------------
NORMALIZATION
-------------------------------------------------------------------------------------------------------------*/
const double muo = 4*M_PI* 1e-7;
double Lo, Io, Jo, Psi_o, GPsi_o;
/*-------------------------------------------------------------------------------------------------------------
CUT-CELL + SINGULARITY SUBTRACTION (docs/FEATURE_CUTCELL_SINGULARITY_CPP_BRIEFING.pdf)

Included here (right after `struct Point`, before any forward declaration
that needs Singularity::PlasmaContext) rather than after GBz() as originally
sketched in the briefing: singularity_kernels.h forward-declares GPsi/GBr/
GBz itself, so it does not need their real definitions yet, but
PsiContributionPlasma()/BRContribution()/BZContribution()'s own forward
declarations just below DO need Singularity::PlasmaContext to already be a
complete type.
-------------------------------------------------------------------------------------------------------------*/
#include "cutcell_geometry.h"
#include "grid_jphi_reconstruction.h"
#include "singularity_kernels.h"
#include "adaptive_quadrature.h"
#include "singularity_subtraction.h"
/*-------------------------------------------------------------------------------------------------------------
MAIN SUBROUTINES
-------------------------------------------------------------------------------------------------------------*/
vector<vector<double>> LoadJphi(const string& filename, int npr, int npz);
vector<Point> LoadPoints(const string& filename);
vector<double> generateGrid(double min, double max, int numPoints);

double Gpsi(double Rsrc, double Zsrc, double R, double Z);
double GPsi(double Rsrc, double Zsrc, double R, double Z);
double GBr (double Rsrc, double Zsrc, double R, double Z);
double GBz (double Rsrc, double Zsrc, double R, double Z);

// PsiContributionPlasma()/BRContribution()/BZContribution() now go through
// cut-cell + singularity subtraction UNCONDITIONALLY (Singularity::psiAt/
// brAt/bzAt) instead of the O(h) full-mesh sum -- see the design decisions
// for this feature (unconditional replacement, no config flag/parallel
// route). `ctx` is the geometry+Jphi-reconstruction bundle built once per
// run in main() right after `boundary` is finalized (Singularity::build()).
double PsiContributionPlasma(Point evalPoint, const Singularity::PlasmaContext& ctx, int order = 2);
double PsiContributionCoils(Point evalPoint,
                         const vector<Point>& coils,
                         const vector<double>& currents);
double computePsiTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
                         const vector<Point>& coils,
                         const vector<double>& currents,
                         int order = 2);
double BZContribution(Point evalPoint, const Singularity::PlasmaContext& ctx, int order = 2);
double BZContributionCoils(Point evalPoint,
                    const vector<Point>& coils,
                    const vector<double>& currents);
double computeBZTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
                    const vector<Point>& coils,
                    const vector<double>& currents,
                    int order = 2);
double BRContribution(Point evalPoint, const Singularity::PlasmaContext& ctx, int order = 2);
double BRContributionCoils(Point evalPoint,
                    const vector<Point>& coils,
                    const vector<double>& currents);
double computeBRTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
                    const vector<Point>& coils,
                    const vector<double>& currents,
                    int order = 2);
void buildLeastSquaresSystem(const vector<Point>& coils,
                const vector<Point>& boundary,
                const Singularity::PlasmaContext& ctx,
                double psi_in,
                Eigen::MatrixXd& J, Eigen::VectorXd& r);

void buildXPointConstraints(const vector<Point>& coils,
                const vector<Point>& Xpoints,
                const Singularity::PlasmaContext& ctx,
                double psi_ref,
                Eigen::MatrixXd& C, Eigen::VectorXd& d);

vector<Point> transformBoundaryWithXpoints(
    const vector<Point>& boundary,
    const vector<Point>& Xpoints,
    int method);

/*-------------------------------------------------------------------------------------------------------------
CONFIGURATION
-------------------------------------------------------------------------------------------------------------*/
struct Config {
    string name, description;
    double r_min, r_max, z_min, z_max;
    int npr, npz;
    double Psi_b;
    string jt_file, boundary_file, coils_file;
    // Optional "lambda" field (absent, or "auto": lambda_is_auto=true).
    // A numeric override is interpreted in the buildSystem()-diagonal
    // convention (A_ii += lambda, matching the historical hardcoded
    // 2.976351e-05) -- see docs/FEATURE_AUTO_REGULARIZATION_BRIEFING.md.
    bool lambda_is_auto = true;
    double lambda_value = 0.0;
    // Ro, I_plasma, P_axis, B_axis: kept here (in addition to the globals
    // Lo/Io derived from Ro/I_plasma) so that regenerateConsistentJt() can
    // hand the SAME numeric values to gs_solver_xpoint's temporary config,
    // guaranteeing the two processes normalize with an identical Lo/Io
    // (see docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf section 8).
    double Ro, I_plasma, P_axis, B_axis;
};

Config loadConfig(const string& filename) {
    ifstream f(filename);
    if (!f.is_open())
        throw runtime_error("Cannot open config file: " + filename);
    json cfg = json::parse(f);

    Lo     = cfg["geometry"]["Ro"];
    Io     = cfg["constraints"]["I_plasma"];
    Jo     = Io / pow(Lo, 2);
    Psi_o  = muo * Lo * Io;
    GPsi_o = Psi_o / Io;

    Config c;
    c.name          = cfg["name"];
    c.description   = cfg["description"];
    c.r_min         = cfg["mesh"]["r_min"];
    c.r_max         = cfg["mesh"]["r_max"];
    c.z_min         = cfg["mesh"]["z_min"];
    c.z_max         = cfg["mesh"]["z_max"];
    c.npr           = cfg["mesh"]["npr"];
    c.npz           = cfg["mesh"]["npz"];
    c.Psi_b         = cfg["constraints"]["Psi_b"];
    c.jt_file       = cfg["files"]["Jt"];
    c.boundary_file = cfg["files"]["boundary"];
    c.coils_file    = cfg["files"]["coils"];

    c.Ro       = cfg["geometry"]["Ro"];
    c.I_plasma = cfg["constraints"]["I_plasma"];
    c.P_axis   = cfg["constraints"]["P_axis"];
    c.B_axis   = cfg["constraints"]["B_axis"];

    c.lambda_is_auto = true;
    c.lambda_value = 0.0;
    if(cfg.contains("lambda")){
        if(cfg["lambda"].is_string()){
            string lam_str = cfg["lambda"];
            if(lam_str != "auto"){
                throw runtime_error("Unrecognized \"lambda\" config value: \"" + lam_str +
                                    "\" (expected \"auto\" or a number)");
            }
            c.lambda_is_auto = true;
        } else if(cfg["lambda"].is_number()){
            c.lambda_is_auto = false;
            c.lambda_value = cfg["lambda"];
        } else {
            throw runtime_error("\"lambda\" config field must be \"auto\" or a number");
        }
    }
    return c;
}

// Regenerates a Jt consistent with a boundary already transformed toward one
// or more X-points, by running the polygonal fixed-boundary Grad-Shafranov
// solver (gs_solver_xpoint) as a subprocess. See the definition below (near
// transformBoundaryWithXpoints(), which it follows the same pattern of) and
// docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf section 8.
vector<vector<double>> regenerateConsistentJt(
    const Config& cfg,
    const vector<Point>& boundary_norm,
    int npr, int npz);

/*-------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
MAIN PROGRAM
---------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------*/
#ifndef UNIT_TESTING
int main(){
    auto start = std::chrono::high_resolution_clock::now();

    string config_path;
    cout << "Enter path to configuration file: ";
    cin >> config_path;
    Config cfg = loadConfig(config_path);
    cout << "Loaded configuration: " << cfg.name << endl;
    cout << cfg.description << endl;

    printf("***EL PROGRAMA HA INICIADO***\n");
    printf("Calculando...\n");


    //1. Define uniform grid
    double r_min = cfg.r_min / Lo;
    double r_max = cfg.r_max / Lo;
    int    npr   = cfg.npr;

    double z_min = cfg.z_min / Lo;
    double z_max = cfg.z_max / Lo;
    int    npz   = cfg.npz;

    vector<double> Rgrid = generateGrid(r_min, r_max, npr);
    vector<double> Zgrid = generateGrid(z_min, z_max, npz);
    printf("Paso 1 listo\n");


    //2. Read toroidal plasma current density Jphi[i][k]
    vector<vector<double>> Jphi = LoadJphi(cfg.jt_file, npr, npz);
    printf("Paso 2 listo\n");


    //3. Read points on the plasma boundary (Rb,Zb)
    vector<Point> boundary = LoadPoints(cfg.boundary_file);
    printf("Paso 3 listo\n");


    //4. Read external coil positions (Rc,Zc)
    vector<Point> coils = LoadPoints(cfg.coils_file);
    printf("Paso 4 listo\n");



    //5. Define X-points positions
    int num_xpoints;
    cout << "\n cuantos x-points quieres?" << endl;
    cin >> num_xpoints; 
    vector<Point> XPoints(num_xpoints);
    for(int i = 0; i < num_xpoints; ++i){
        cout << "Ingresa las coordenadas noormalizadas para el punto " << i + 1 << ":\n" << endl;
        cout << "Coordenada R: ";
        cin >> XPoints[i].R;
        XPoints[i].R = XPoints[i].R/Lo;
        cout << "Coordenada Z: ";
        cin >> XPoints[i].Z;
        XPoints[i].Z = XPoints[i].Z/Lo;
    }
    //Point Xpoint = {0.4/Lo, 1.14/Lo};
    printf("Paso 5 listo\n");


    //5b. Transform boundary with X-points (optional)
    if(num_xpoints > 0){
        int transform_method;
        cout << "\nBoundary transformation method:" << endl;
        cout << "  0: No transformation (original D-shape)" << endl;
        cout << "  1: Multi-parameter adaptive" << endl;
        cout << "  2: Dual-parameter adaptive" << endl;
        cout << "  3: Dual-parameter adaptive, C1 cusp (Bezier, smooth join, recommended)" << endl;
        cout << "Select method: ";
        cin >> transform_method;

        if(transform_method == 1 || transform_method == 2 || transform_method == 3){
            boundary = transformBoundaryWithXpoints(boundary, XPoints, transform_method);

            FILE *f_dshape = fopen("Dshape_xpoint.txt", "w");
            for(const auto& p : boundary){
                fprintf(f_dshape, "%.10e %.10e\n", p.R * Lo, p.Z * Lo);
            }
            fclose(f_dshape);

            printf("Paso 5b listo: boundary transformed with %d points (saved to Dshape_xpoint.txt)\n", (int)boundary.size());

            // 5c. Re-solve Jt on the boundary that was just transformed, so the
            // plasma current is genuinely consistent with the new (X-point)
            // separatrix instead of remaining masked to the original D-shape
            // (see docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf). Only runs
            // here (transform_method 1 or 2, i.e. right after this same
            // process produced boundary); transform_method==0 keeps whatever
            // Jt the user already loaded, unchanged.
            Jphi = regenerateConsistentJt(cfg, boundary, npr, npz);
            printf("Paso 5c listo: Jt regenerado consistente con la frontera transformada (Jt_xpoint.txt)\n");
        }
    }


    //6-7. Construct and solve for coil currents.
    // Both Nx==0 and Nx>0 now go through the SVD/Tikhonov path (regularization.h),
    // built on the same rectangular (J,r) least-squares system (boundary fit).
    // Nx==0: solved directly via Regularization::solveWithAutoRegularization().
    // Nx>0: the X-point conditions (B_R=0, B_Z=0, Psi=Psi_ref at each X-point) are
    // encoded as an equality constraint C*I=d and reduced to an unconstrained
    // problem via the null-space method (Regularization::solveConstrainedWithAutoRegularization()),
    // which reuses solveWithAutoRegularization() unmodified on the reduced system.
    // Both report kappa/lambda diagnostics via regularization_diagnostics.txt.
    // See docs/FEATURE_AUTO_REGULARIZATION_BRIEFING.md and
    // docs/FEATURE_XPOINT_NULLSPACE_REGULARIZATION_BRIEFING.md for the derivations.
    int Nc = coils.size();
    vector<double> currents(Nc);

    // A manual "lambda" override in the config is in the buildSystem()-diagonal
    // convention (A_ii += lambda); the SVD/null-space paths use the standard
    // Tikhonov convention min||JI-r||^2 + lambda_standard*||I||^2, i.e.
    // lambda_standard = lambda/2 (see the derivation in the report).
    double lambda_override_standard = cfg.lambda_is_auto ? -1.0 : (cfg.lambda_value / 2.0);

    // Cut-cell geometry + Jphi reconstruction, built ONCE for this run, right
    // here -- the one point that covers every combination (with/without
    // X-points, with/without boundary transformation), since `boundary` and
    // `Jphi` are already in their final form by this line regardless of
    // which branch above produced them. Passed by const reference to every
    // consumer (buildLeastSquaresSystem, buildXPointConstraints, and the
    // psi_check.txt sweep below) -- see the design decisions for this
    // feature (cut-cell + singularity subtraction).
    Singularity::PlasmaContext ctx = Singularity::build(boundary, Jphi, Rgrid, Zgrid);
    printf("Geometria cut-cell + reconstruccion Jphi cacheadas (nCut=%d, nCandidatas=%d, areaSum=%.6e, areaExacta=%.6e)\n",
           ctx.geom.nCut, ctx.geom.nCandidates, ctx.geom.areaSum, ctx.geom.areaExact);

    Eigen::MatrixXd J;
    Eigen::VectorXd r;
    buildLeastSquaresSystem(coils, boundary, ctx, cfg.Psi_b, J, r);

    if(num_xpoints == 0){
        Regularization::RegularizationResult reg =
            Regularization::solveWithAutoRegularization(J, r, lambda_override_standard);

        for(int i = 0; i < Nc; ++i){
            currents[i] = reg.currents(i);
        }

        printf("Paso 6-7 listo (SVD/Tikhonov): kappa(J)=%.6e, lambda=%.6e (%s)\n",
               reg.kappa_J, reg.lambda_used, reg.lambda_was_auto ? "auto" : "manual");

        FILE *out_diag = fopen("regularization_diagnostics.txt", "w");
        fprintf(out_diag, "kappa_J %.10e\n", reg.kappa_J);
        fprintf(out_diag, "lambda_standard %.10e\n", reg.lambda_used);
        fprintf(out_diag, "lambda_buildSystem_diagonal_convention_equivalent %.10e\n", reg.lambda_used * 2.0);
        fprintf(out_diag, "lambda_source %s\n", reg.lambda_was_auto ? "auto" : "manual_override");
        fclose(out_diag);
    } else {
        Eigen::MatrixXd C;
        Eigen::VectorXd d;
        buildXPointConstraints(coils, XPoints, ctx, cfg.Psi_b, C, d);

        Regularization::RegularizationResultConstrained reg =
            Regularization::solveConstrainedWithAutoRegularization(J, r, C, d, lambda_override_standard);

        for(int i = 0; i < Nc; ++i){
            currents[i] = reg.currents(i);
        }

        printf("Paso 6-7 listo (null-space/Tikhonov): kappa(C)=%.6e, kappa(J~)=%.6e, "
               "lambda=%.6e (%s), rank(C)=%d, ||C*I-d||=%.6e\n",
               reg.kappa_C, reg.kappa_J_reduced, reg.lambda_used,
               reg.lambda_was_auto ? "auto" : "manual", reg.rank_C, reg.constraint_residual);

        FILE *out_diag = fopen("regularization_diagnostics.txt", "w");
        fprintf(out_diag, "kappa_C %.10e\n", reg.kappa_C);
        fprintf(out_diag, "kappa_J_reduced %.10e\n", reg.kappa_J_reduced);
        fprintf(out_diag, "lambda_standard %.10e\n", reg.lambda_used);
        fprintf(out_diag, "lambda_buildSystem_diagonal_convention_equivalent %.10e\n", reg.lambda_used * 2.0);
        fprintf(out_diag, "lambda_source %s\n", reg.lambda_was_auto ? "auto" : "manual_override");
        fprintf(out_diag, "rank_C %d\n", reg.rank_C);
        fprintf(out_diag, "constraint_residual %.10e\n", reg.constraint_residual);
        fclose(out_diag);
    }

    for(int i = 0; i < Nc; ++i){
        cout << "I[" << i << "] = " << currents[i] << endl;
    }
    printf("Paso 8 listo\n");



    //------------------------------------------------
    // This loop is, measured, the dominant cost of the whole pipeline
    // (CLAUDE.md, "Performance note"): with the unconditional cut-cell +
    // singularity-subtraction replacement, EVERY one of the npr*npz mesh
    // points now pays the full row-scan (9 geometric constants) cost, not
    // just the O(Nb+Nx) boundary/X-point points buildLeastSquaresSystem/
    // buildXPointConstraints evaluate -- see the design decisions for this
    // feature. Each point is independent given `ctx`/`currents` (both
    // const), so this is parallelized with OpenMP; compile with -fopenmp.
    // An exception escaping an OpenMP parallel region is undefined behavior
    // and calls std::terminate() immediately (SIGABRT), unrecoverable by any
    // try/catch outside the region -- found the hard way (2026-09-12) via a
    // std::runtime_error thrown from inside JphiGrid::gradientAt()/
    // ghostValueAtNode() at some mesh point aborting this whole process with
    // no diagnostic. Every point must therefore be wrapped INSIDE the
    // parallel loop; a failing point is marked NaN and counted rather than
    // crashing the run.
    vector<vector<double>> psiGrid(npr, vector<double>(npz, 0.0));
    int nPointFailures = 0;
    #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:nPointFailures)
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            Point point = {Rgrid[i], Zgrid[k]};
            try {
                psiGrid[i][k] = computePsiTotal(point, ctx, coils, currents);
            } catch(const std::exception& e){
                psiGrid[i][k] = std::nan("");
                nPointFailures++;
                #pragma omp critical
                fprintf(stderr, "WARNING: computePsiTotal failed at (R=%.6f, Z=%.6f): %s\n",
                        point.R, point.Z, e.what());
            }
        }
    }
    if(nPointFailures > 0)
        printf("WARNING: %d of %d psi_check.txt points failed and were written as NaN "
               "(see stderr warnings above for coordinates)\n", nPointFailures, npr*npz);
    // psi_check.txt is written with %.15e (full double precision), not %e
    // (6 decimals): a validation pass cross-checking this field's own
    // finite-difference derivatives (B_R=-(1/R)dpsi/dZ, B_Z=(1/R)dpsi/dR)
    // against the closed-form BR_check.txt/BZ_check.txt below would
    // otherwise inherit the same precision floor documented and fixed for
    // GS_solver.h's field outputs ("Bug 14", CLAUDE.md) -- %e's 6 decimals,
    // amplified by dividing by h^2 in the finite difference, puts an
    // artificial floor of ~1e-3 on any residual recomputed from the file,
    // masking genuine agreement/disagreement between the two methods.
    FILE *out1;
    out1 = fopen("psi_check.txt","w");
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            fprintf(out1,"%.15e ", psiGrid[i][k]*Psi_o);
        }
        fprintf(out1,"\n");
    }
    fclose(out1);

    FILE *out_coils;
    out_coils = fopen("corrientes.txt","w");
    for(int i = 0; i < Nc; ++i){
        fprintf(out_coils,"%.10e %.10e %.10e\n",
                coils[i].R * Lo,
                coils[i].Z * Lo,
                currents[i] * Io);
    }
    fclose(out_coils);
    printf("Coil data written to corrientes.txt (%d coils)\n", Nc);

    // psi_boundary_check.txt: computePsiTotal() evaluated EXACTLY at each
    // Dshape.txt boundary point -- the same function call, on the same
    // points, that buildLeastSquaresSystem() used to build r(j) (Nc, in
    // PsiContributionPlasma()) when fitting the currents just solved above.
    // This is the direct, non-interpolated boundary-condition check
    // (psi(boundary) should equal Psi_b): unlike reading it off psi_check.txt
    // via interpolation, it carries no extra interpolation error and is not
    // a duplicate of the cut-cell + singularity-subtraction machinery in a
    // second language -- it just re-queries the same evaluator.
    double maxBoundaryResidual = 0.0;
    FILE *out_boundary = fopen("psi_boundary_check.txt","w");
    fprintf(out_boundary, "# R[m] Z[m] psi_computed[Wb/rad] Psi_b[Wb/rad] abs_diff[Wb/rad]\n");
    for(size_t j = 0; j < boundary.size(); ++j){
        double psi_computed_phys = computePsiTotal(boundary[j], ctx, coils, currents) * Psi_o;
        double diff = fabs(psi_computed_phys - cfg.Psi_b);
        maxBoundaryResidual = max(maxBoundaryResidual, diff);
        fprintf(out_boundary, "%.10e %.10e %.15e %.10e %.15e\n",
                boundary[j].R * Lo, boundary[j].Z * Lo,
                psi_computed_phys, cfg.Psi_b, diff);
    }
    fclose(out_boundary);
    printf("Boundary residual: max |psi_computed - Psi_b| = %.6e Wb/rad over %zu points "
           "(psi_boundary_check.txt)\n", maxBoundaryResidual, boundary.size());

    // sample_points_check.txt: computePsiTotal()/computeBRTotal()/computeBZTotal()
    // evaluated EXACTLY (not interpolated off psi_check.txt/BR_check.txt/
    // BZ_check.txt) at a small, fixed list of (R,Z) points -- added for the
    // DIII-D grid-convergence validation study
    // (docs/FEATURE_DIIID_GRID_CONVERGENCE_VALIDATION_BRIEFING.md), which
    // needs psi/B_R/B_Z at IDENTICAL physical points across four mesh
    // resolutions with no interpolation error contaminating the comparison.
    // Six fixed points: near the magnetic axis; interior/exterior just
    // inside/outside the LCFS on both the inboard and outboard sides,
    // anchored at Z=0.35 m (not the midplane -- chosen inside the Z in
    // [0,0.5] band where docs/reports/BFIELD_LCFS_ACCURACY_INVESTIGATION.md
    // found the Green's-function/finite-difference B disagreement
    // strongest), offset 3% of the DIII-D minor radius a=0.67 m inward/
    // outward along each anchor's own local outward normal (NOT purely
    // radial off-midplane -- derived in
    // docs/FEATURE_DIIID_GRID_CONVERGENCE_VALIDATION_BRIEFING.md section 5.3
    // from the D-shape parametrization rboundary()/zboundary(), matching
    // GS_solver.h:1855/1864); and one far-exterior deep-vacuum point on the
    // outboard midplane, well clear of both the LCFS and every one of the 18
    // coils -- plus, when X-points are in use, one extra row per X-point
    // (already available, normalized, in XPoints[]). Negligible added cost
    // (a handful of extra evaluator calls); placed here, before the timing
    // checkpoint below, so it falls inside the "through psi_check.txt"
    // timing bucket rather than the BZ/BR sweep bucket. The six physical
    // coordinates are hardcoded (not config-driven) because they are tied to
    // this specific DIII-D geometry (Ro=1.67, a=0.67, kappa=1.77,
    // delta=0.30) -- for any other case/config run through this same binary
    // they are simply extra, harmless evaluation points.
    {
        vector<Point> samplePointsPhysicalM = {
            {1.670, 0.000},   // 1: near magnetic axis (geometric center)
            {1.035, 0.348},   // 2: interior, near LCFS, inboard
            {2.271, 0.344},   // 3: interior, near LCFS, outboard
            {0.995, 0.352},   // 4: exterior, just outside LCFS, inboard
            {2.309, 0.356},   // 5: exterior, just outside LCFS, outboard
            {2.750, 0.000},   // 6: far exterior (deep vacuum)
        };
        for(const auto& xp : XPoints){
            samplePointsPhysicalM.push_back({xp.R * Lo, xp.Z * Lo});
        }

        double Bo_local = Psi_o / (Lo * Lo);
        FILE *out_samples = fopen("sample_points_check.txt", "w");
        fprintf(out_samples, "# R[m] Z[m] psi[Wb/rad] B_R_green[T] B_Z_green[T]\n");
        for(const auto& p_phys : samplePointsPhysicalM){
            Point p_norm = {p_phys.R / Lo, p_phys.Z / Lo};
            double psi_val = std::nan(""), br_val = std::nan(""), bz_val = std::nan("");
            try { psi_val = computePsiTotal(p_norm, ctx, coils, currents) * Psi_o; }
            catch(const std::exception& e){
                fprintf(stderr, "WARNING: sample_points_check.txt: computePsiTotal failed at "
                        "(R=%.6f, Z=%.6f): %s\n", p_phys.R, p_phys.Z, e.what());
            }
            try { br_val = computeBRTotal(p_norm, ctx, coils, currents) * Bo_local; }
            catch(const std::exception& e){
                fprintf(stderr, "WARNING: sample_points_check.txt: computeBRTotal failed at "
                        "(R=%.6f, Z=%.6f): %s\n", p_phys.R, p_phys.Z, e.what());
            }
            try { bz_val = computeBZTotal(p_norm, ctx, coils, currents) * Bo_local; }
            catch(const std::exception& e){
                fprintf(stderr, "WARNING: sample_points_check.txt: computeBZTotal failed at "
                        "(R=%.6f, Z=%.6f): %s\n", p_phys.R, p_phys.Z, e.what());
            }
            fprintf(out_samples, "%.10e %.10e %.15e %.15e %.15e\n",
                    p_phys.R, p_phys.Z, psi_val, br_val, bz_val);
        }
        fclose(out_samples);
        printf("Sample points evaluated and written to sample_points_check.txt (%zu points)\n",
               samplePointsPhysicalM.size());
    }

    // Timing checkpoint: elapsed time up to and including psi_check.txt, separate from
    // the total pipeline time (printed at the very end of main()) -- lets a grid-convergence
    // study report "time to solve + psi field" separately from the extra cost of the
    // BZ_check.txt/BR_check.txt sweeps below, which roughly triple the total runtime.
    {
        auto checkpoint = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<60>> elapsed_to_psi_min = checkpoint - start;
        printf("Elapsed time through psi_check.txt (excludes BZ_check.txt/BR_check.txt): "
               "%.6f minutes\n", elapsed_to_psi_min.count());
    }

    //------------------------------------------------
    // BZ_check.txt / BR_check.txt: closed-form B_Z, B_R (GBz/GBr Green's
    // functions, via computeBZTotal()/computeBRTotal()) over the same grid
    // as psi_check.txt. Re-enabled 2026-09-14 for the physical-validation
    // section of the validation studies, updated to the current ctx-based signature
    // (the previous version of this block, calling
    // computeBZTotal(point, Jphi, Rgrid, Zgrid, coils, currents), predates
    // the cut-cell + singularity-subtraction refactor and no longer
    // compiles as written). Parallelized identically to the psi_check.txt
    // loop above, including the same per-point try/catch: an exception
    // escaping an OpenMP parallel region is undefined behavior and calls
    // std::terminate() immediately, so every point must be wrapped INSIDE
    // the parallel loop rather than around it.
    vector<vector<double>> bzGrid(npr, vector<double>(npz, 0.0));
    int nBzFailures = 0;
    #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:nBzFailures)
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            Point point = {Rgrid[i], Zgrid[k]};
            try {
                bzGrid[i][k] = computeBZTotal(point, ctx, coils, currents);
            } catch(const std::exception& e){
                bzGrid[i][k] = std::nan("");
                nBzFailures++;
                #pragma omp critical
                fprintf(stderr, "WARNING: computeBZTotal failed at (R=%.6f, Z=%.6f): %s\n",
                        point.R, point.Z, e.what());
            }
        }
    }
    if(nBzFailures > 0)
        printf("WARNING: %d of %d BZ_check.txt points failed and were written as NaN "
               "(see stderr warnings above for coordinates)\n", nBzFailures, npr*npz);
    // B_0 = Psi_0/L_0^2 (= mu_0*I_0/L_0): the normalized G^{B_R}/G^{B_Z}
    // drop mu_0 the same way GPsi does (normalized Green functions of B_R and
    // B_Z), so computeBZTotal()/computeBRTotal() return normalized
    // B_hat and must be rescaled by B_0 here to get physical Tesla, matching
    // psi_check.txt's own rescaling by Psi_o just above.
    double Bo = Psi_o / (Lo*Lo);
    FILE *out2;
    out2 = fopen("BZ_check.txt","w");
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            fprintf(out2,"%.15e ", bzGrid[i][k]*Bo);
        }
        fprintf(out2,"\n");
    }
    fclose(out2);

    vector<vector<double>> brGrid(npr, vector<double>(npz, 0.0));
    int nBrFailures = 0;
    #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:nBrFailures)
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            Point point = {Rgrid[i], Zgrid[k]};
            try {
                brGrid[i][k] = computeBRTotal(point, ctx, coils, currents);
            } catch(const std::exception& e){
                brGrid[i][k] = std::nan("");
                nBrFailures++;
                #pragma omp critical
                fprintf(stderr, "WARNING: computeBRTotal failed at (R=%.6f, Z=%.6f): %s\n",
                        point.R, point.Z, e.what());
            }
        }
    }
    if(nBrFailures > 0)
        printf("WARNING: %d of %d BR_check.txt points failed and were written as NaN "
               "(see stderr warnings above for coordinates)\n", nBrFailures, npr*npz);
    FILE *out3;
    out3 = fopen("BR_check.txt","w");
    for(int k=0;k<npz;k++){
        for(int i=0;i<npr;i++){
            fprintf(out3,"%.15e ", brGrid[i][k]*Bo);
        }
        fprintf(out3,"\n");
    }
    fclose(out3);
   //------------------------------------------------

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<60>> duration_min = end - start;
    std::cout << "El codigo se demoro: " << duration_min.count() << " minutos" << std::endl;
    printf("***EL PROGRAMA HA TERMINADO***\n");
    return 0;
}
#endif // UNIT_TESTING

/*-------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
SUBROUTINES DECLARATION
---------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------*/


//------------------------------------------------------------------------------
//Green's function for poloidal flux
//------------------------------------------------------------------------------
// Takes p = 1-kappa^2 DIRECTLY (not kappa itself), mirroring
// tools/singularity/kernels.py's use of scipy.special.ellipkm1(p) /
// ellipe(1-p). This matters numerically: computing b=sqrt(1-kappa*kappa)
// from a kappa that is already very close to 1 re-derives the tiny
// quantity p via a subtraction that discards essentially all of p's
// precision (the round-trip kappa=sqrt(p_accurate) -> kappa*kappa can differ
// from p_accurate in its last surviving bits, and "1 minus a number extremely
// close to 1" amplifies that into a large relative error) -- exactly the
// cancellation this project has already had to fix twice for GBr/GBz's own
// external divisions (see the p=d^2/gamma^2 comments in GPsi/GBr/GBz below).
// Taking the already-accurately-computed p directly sidesteps the
// kappa->kappa^2 round-trip entirely: b=sqrt(p) keeps all of p's precision,
// and c=sqrt(1-p) (~kappa, needed only for the AGM's correction sum, not for
// recovering p) loses no information that mattered in the first place.
void ellipticKE(double p, double& K, double& E){

    double a = 1.;
    double b = sqrt(p);
    double c = sqrt(fabs(1.0 - p));
   
    double aux;
   
    double tol = 1.e-9;
    double err = 1.0;
   
    int n = 0;
   
    double sum = pow(2.,n-1)*c*c;
   
    while(err>tol){
    
        n++;
    
        aux = a;
    
        a = 0.5*(aux+b);
        b = sqrt(aux*b);
        c = 0.25*c*c/a;
        sum = sum + pow(2.,n-1)*c*c;

        err = fabs(a-b);
    
    }
   
    K = 0.5*M_PI/a;
    E = K*(1.-sum);
   
  }


double GPsi(double Rs, double Zs, double R, double Z){

    // Same p=1-kappa^2 exact-identity fix as GBr()/GBz() (see the comment
    // there): although GPsi() never divides externally by p (so it cannot
    // itself produce the 0*inf=NaN failure mode those two have), its
    // ACCURACY still depends on kappa2 being precise, because
    // ellipticKE(kappa,...) computes b=sqrt(1-kappa*kappa) internally --
    // if kappa2 was already imprecise from cancellation in 4*R*Rs/D, that
    // subtraction discards whatever precision was left, silently degrading
    // GPsi (and therefore greg=GPsi-gpsiSing, singularity_kernels.h) for
    // small d without crashing. Found empirically 2026-09-11: greg's
    // convergence to its analytic d->0 limit stalled at d~1e-7 with the old
    // naive kappa2 (relative error 2%-40% instead of the expected <1e-6).
    double D  = pow(R + Rs, 2) + pow(Z - Zs, 2);
    double d2 = pow(R - Rs, 2) + pow(Z - Zs, 2);
    double p  = max(d2 / D, 1e-300);
    double kappa2 = 1.0 - p;
    double kappa  = sqrt(kappa2);

    double K, E;
    ellipticKE(p, K, E);
    double G = sqrt(R*Rs)*( (2. - kappa2)*K - 2.*E )/(2.*M_PI*kappa);

    return G;
  }
//------------------------------------------------------------------------------
//Green's function for B_R
//------------------------------------------------------------------------------
double GBr(double Rs, double Zs, double R, double Z){

    // p = 1-kappa^2 is computed via the exact geometric identity d^2/gamma^2
    // (d^2=(R-Rs)^2+(Z-Zs)^2, gamma^2=D) instead of the naive 1-4*R*Rs/D,
    // which suffers catastrophic cancellation as the source approaches the
    // evaluation point (d->0). p is then floored at 1e-300 -- matching the
    // same floor already used inside ellipticKE's own convergence -- and
    // kept as its own variable through to the division below (2*p), rather
    // than re-derived from kappa2=1-p, because kappa2 itself rounds to
    // exactly 1.0 in double precision once p drops below ~1.1e-16, which
    // would otherwise make (2-2*kappa2) an exact zero and, combined with a
    // numerator that can independently underflow to exact zero along
    // near-tangent directions, produce 0*inf=NaN (see CLAUDE.md, "Data File
    // Formats" section on GBr/GBz singularity handling).
    double D  = pow(R + Rs, 2) + pow(Z - Zs, 2);
    double d2 = pow(R - Rs, 2) + pow(Z - Zs, 2);
    double p  = max(d2 / D, 1e-300);
    double kappa2 = 1.0 - p;
    double kappa  = sqrt(kappa2);

    double K, E;
    ellipticKE(p, K, E);
    //dGdz = (Z-Zs)*kappa*(2.*K - (2.-kappa2)*E/(1.-kappa2))/(8.*M_PI*sqrt(R*Rs));
    double dGdz = ((Z-Zs)/(2.*M_PI))*sqrt(kappa2/(4*R*Rs))*(E*(2-kappa2)/(2.*p)-K);

    return dGdz/R;
  }
//------------------------------------------------------------------------------
//Green's function for B_Z
//------------------------------------------------------------------------------
double GBz(double Rs, double Zs, double R, double Z){

    // Same p=1-kappa^2 floor/identity fix as GBr() above -- see the comment
    // there for the full derivation.
    double D  = pow(R + Rs, 2) + pow(Z - Zs, 2);
    double d2 = pow(R - Rs, 2) + pow(Z - Zs, 2);
    double p  = max(d2 / D, 1e-300);
    double kappa2 = 1.0 - p;
    double kappa  = sqrt(kappa2);

    double K, E;
    ellipticKE(p, K, E);
    //dGdr = sqrt(Rs/R)*((2.-kappa2)*K - 2.*E +
    //                  ((2.-kappa2)*E/(1.-kappa2)-2.*K)*
    //                  (1.-(kappa2*(R+Rs)/(2.*Rs))))/(4.*M_PI*kappa);
    // (Rs*kappa2 - (2-kappa2)*R), expanded with kappa2=1-p, algebraically
    // equals (Rs-R) - p*(Rs+R): computing it via kappa2 subtracts two O(1)
    // quantities (kappa2*(Rs+R) and 2R) to recover something as small as
    // (Rs-R), losing precision proportional to how close p is to 0 (the
    // same cancellation pattern already fixed elsewhere in this file).
    // The (Rs-R)-p*(Rs+R) form computes the O(d) difference directly from
    // the inputs (exact) plus a genuinely small correction, with no
    // cancellation. Found 2026-09-11 while chasing a ~1-2% residual error
    // in GBz's own d->0 limit that survived the p=d^2/gamma^2 identity fix.
    double dGdr = sqrt(kappa2/(4*Rs*R))*( R*K + E*((Rs-R) - p*(Rs+R))/(2.*p)) /(2.*M_PI);

    return dGdr/R;
  }

/*-------------------------------------------------------------------------------------------------------------
JPHI LECTURE
-------------------------------------------------------------------------------------------------------------*/
vector<vector<double>> LoadJphi(const string& filename, int npr, int npz){
   
   ifstream file(filename);
   if(!file.is_open()){
    throw runtime_error("No se pudo abrir el archivo " + filename);
   }
   
   vector<vector<double>> Jphi(npr, vector<double>(npz, 0.0));
   
   string line;
   int k = 0;
   while(getline(file, line) && k < npz){
    istringstream iss(line);
    for(int i = 0; i < npr; ++i){
     double val;
     if(!(iss >> val)){
      throw runtime_error("Error de lectura en la línea " + to_string(k) +
                          ", columna " + to_string(i));
     }
     
     Jphi[i][k] = val/Jo;//Para normalizar... cambiar
    }
    
    ++k;
    
   }
   
   if(k != npz){
    throw runtime_error("Número de filas leídas no coincide con npz.");
   }
   
   file.close();
   
   //Chequeo de lectura------------------------------
     FILE *out1;
     out1 = fopen("Jphi_check.txt","write");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%e ",Jphi[i][k]*Jo);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //------------------------------------------------
   return Jphi;
  }
/*-------------------------------------------------------------------------------------------------------------
POINTS LECTURE
-------------------------------------------------------------------------------------------------------------*/
vector<Point> LoadPoints(const string& filename){

    vector<Point> points;
    ifstream file(filename);
    double R, Z;
    while (file >> R >> Z) {
        points.push_back({R/Lo, Z/Lo});
    }
    file.close();
    return points;
}
/*-------------------------------------------------------------------------------------------------------------
GENERATE GRID
-------------------------------------------------------------------------------------------------------------*/
vector<double> generateGrid(double min, double max, int numPoints){

    vector<double> grid(numPoints);
    double h = (max - min) / (numPoints - 1);
    for(int i = 0; i < numPoints; ++i){
        grid[i] = min + i*h;
    }
    return grid;
}
/*-------------------------------------------------------------------------------------------------------------
PSI CONTRIBUTION DUE TO PLASMA
-------------------------------------------------------------------------------------------------------------*/
double PsiContributionPlasma(Point evalPoint, const Singularity::PlasmaContext& ctx, int order){
    return Singularity::psiAt(ctx, ctx.jfield.R, ctx.jfield.Z, evalPoint.R, evalPoint.Z, order);
}
/*------------------------------------------------------------------------------
PSI CONTRIBUTION DUE TO COILS
------------------------------------------------------------------------------*/
double PsiContributionCoils(Point evalPoint,
                         const vector<Point>& coils,
                         const vector<double>& currents){
    double psi_c = 0.0;

    for(size_t i = 0; i < coils.size(); ++i){
        psi_c += currents[i]*GPsi(coils[i].R,coils[i].Z,evalPoint.R,evalPoint.Z);
    }

    return psi_c;
}
/*------------------------------------------------------------------------------
PSI/BR/BZ CONTRIBUTION DUE TO PLASMA -- FAST PATH FOR P OUTSIDE THE PLASMA

Used only by computePsiTotal()/computeBZTotal()/computeBRTotal() below (the
psi_check.txt/BR_check.txt/BZ_check.txt dense-map paths, which query P over the
WHOLE mesh, including deep-exterior points) -- NOT by PsiContributionPlasma()/
BRContribution()/BZContribution() themselves, which stay untouched because
buildLeastSquaresSystem() also calls PsiContributionPlasma() directly for
boundary points sitting exactly on Gamma (the same polygon insidePolygon()
tests against), where the in/out classification of a point ON the polygon is
not reliable enough to risk silently downgrading the coil-current fit's
accuracy -- see docs/reports/GREEN_FUNCTION_REGULAR_TERM_SERIES.tex.

When P is not inside the plasma boundary, the source s (which ranges only over
Omega_p during the integral, ecuacion (1) of GREEN_FUNCTION_SINGULARITY) can
never equal P: G^psi(s;P)/G^{B_R}(s;P)/G^{B_Z}(s;P) are smooth over the whole
integration domain, so the cut-cell + singularity-subtraction machinery
(Taylor patch of Jphi*c, gradientAt()) is mathematically unnecessary --
ordinary cut-cell quadrature of Jphi(s)*G(s;P) already converges at full order
there. This also sidesteps gradientAt()'s "1-2 genuinely interior nodes"
exception entirely for exterior points, which is what produced the ring of NaN
points found around the LCFS in psi_check.txt (CLAUDE.md, "Data File Formats").
-------------------------------------------------------------------------------*/
inline double PsiContributionPlasma_direct(Point evalPoint, const Singularity::PlasmaContext& ctx){
    return Singularity::cutCellIntegral(ctx, ctx.jfield.R, ctx.jfield.Z,
        [&](double r, double z){
            return JphiGrid::valueAt(ctx.jfield, r, z) * GPsi(r, z, evalPoint.R, evalPoint.Z);
        });
}
inline double BZContributionPlasma_direct(Point evalPoint, const Singularity::PlasmaContext& ctx){
    return Singularity::cutCellIntegral(ctx, ctx.jfield.R, ctx.jfield.Z,
        [&](double r, double z){
            return JphiGrid::valueAt(ctx.jfield, r, z) * GBz(r, z, evalPoint.R, evalPoint.Z);
        });
}
inline double BRContributionPlasma_direct(Point evalPoint, const Singularity::PlasmaContext& ctx){
    return Singularity::cutCellIntegral(ctx, ctx.jfield.R, ctx.jfield.Z,
        [&](double r, double z){
            return JphiGrid::valueAt(ctx.jfield, r, z) * GBr(r, z, evalPoint.R, evalPoint.Z);
        });
}
/*------------------------------------------------------------------------------
TOTAL PSI
-------------------------------------------------------------------------------*/
double computePsiTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
                         const vector<Point>& coils,
                         const vector<double>& currents,
                         int order){

 double psi_p = CutCell::insidePolygon(evalPoint.R, evalPoint.Z, ctx.boundary)
              ? PsiContributionPlasma(evalPoint, ctx, order)
              : PsiContributionPlasma_direct(evalPoint, ctx);
 return psi_p + PsiContributionCoils(evalPoint, coils, currents);
}




/*-------------------------------------------------------------------------------------------------------------
BZ CONTRIBUTION DUE TO PLASMA
-------------------------------------------------------------------------------------------------------------*/
double BZContribution(Point evalPoint, const Singularity::PlasmaContext& ctx, int order){
    return Singularity::bzAt(ctx, ctx.jfield.R, ctx.jfield.Z, evalPoint.R, evalPoint.Z, order);
}
/*-------------------------------------------------------------------------------------------------------------
BZ CONTRIBUTION DUE TO COILS
-------------------------------------------------------------------------------------------------------------*/
double BZContributionCoils(Point evalPoint,
    const vector<Point>& coils,
    const vector<double>& currents){
        double bz_c = 0.0;

        for(size_t i = 0; i < coils.size(); ++i){
            bz_c += currents[i]*GBz(coils[i].R,coils[i].Z,evalPoint.R,evalPoint.Z);
        }
    
        return bz_c;
}
/*------------------------------------------------------------------------------
TOTAL Bz
-------------------------------------------------------------------------------*/
double computeBZTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
    const vector<Point>& coils,
    const vector<double>& currents,
    int order){

double bz_p = CutCell::insidePolygon(evalPoint.R, evalPoint.Z, ctx.boundary)
            ? BZContribution(evalPoint, ctx, order)
            : BZContributionPlasma_direct(evalPoint, ctx);
return bz_p + BZContributionCoils(evalPoint, coils, currents);
}


/*-------------------------------------------------------------------------------------------------------------
BR CONTRIBUTION DUE TO PLASMA
-------------------------------------------------------------------------------------------------------------*/
double BRContribution(Point evalPoint, const Singularity::PlasmaContext& ctx, int order){
    return Singularity::brAt(ctx, ctx.jfield.R, ctx.jfield.Z, evalPoint.R, evalPoint.Z, order);
}

/*-------------------------------------------------------------------------------------------------------------
BR CONTRIBUTION DUE TO COILS
-------------------------------------------------------------------------------------------------------------*/
double BRContributionCoils(Point evalPoint,
    const vector<Point>& coils,
    const vector<double>& currents){
        double br_c = 0.0;

        for(size_t i = 0; i < coils.size(); ++i){
            br_c += currents[i]*GBr(coils[i].R,coils[i].Z,evalPoint.R,evalPoint.Z);
        }
    
        return br_c;
}
/*------------------------------------------------------------------------------
TOTAL BR
-------------------------------------------------------------------------------*/
double computeBRTotal(Point evalPoint, const Singularity::PlasmaContext& ctx,
    const vector<Point>& coils,
    const vector<double>& currents,
    int order){

double br_p = CutCell::insidePolygon(evalPoint.R, evalPoint.Z, ctx.boundary)
            ? BRContribution(evalPoint, ctx, order)
            : BRContributionPlasma_direct(evalPoint, ctx);
return br_p + BRContributionCoils(evalPoint, coils, currents);
}

/*-------------------------------------------------------------------------------------------------------------
LEAST-SQUARES SYSTEM (J, r) AND X-POINT EQUALITY CONSTRAINTS (C, d)

Replaces the historical buildSystem()+solveLinearSystem() (dense normal-equations,
A=2*J^T*J(+bordering)+lambda*I, solved by Gaussian elimination). Both the Nx=0 and Nx>0 branches
of main() now build the same rectangular (J,r) least-squares system here; the Nx>0 branch
additionally builds the X-point equality constraints (C,d) and reduces them via the null-space
method (Regularization::solveConstrainedWithAutoRegularization(), regularization.h) instead of
folding them into a bordered normal-equations matrix. See
docs/FEATURE_XPOINT_NULLSPACE_REGULARIZATION_BRIEFING.md for the derivation.
-------------------------------------------------------------------------------------------------------------*/
void buildLeastSquaresSystem(const vector<Point>& coils,
                const vector<Point>& boundary,
                const Singularity::PlasmaContext& ctx,
                double psi_in,
                Eigen::MatrixXd& J, Eigen::VectorXd& r){

    int Nc = coils.size();
    int Nb = boundary.size();
    J.resize(Nb, Nc);
    r.resize(Nb);
    for(int j = 0; j < Nb; ++j){
        double psi_p = PsiContributionPlasma(boundary[j], ctx);
        r(j) = psi_in - psi_p;
        for(int i = 0; i < Nc; ++i){
            J(j, i) = GPsi(coils[i].R, coils[i].Z, boundary[j].R, boundary[j].Z);
        }
    }
}

// Computes B_R=0, B_Z=0, Psi=psi_ref at each X-point via a single
// Singularity::evaluateAll() call per X-point rather than three separate
// BRContribution()/BZContribution()/PsiContributionPlasma() calls: all three
// share the same 9 exact geometric constants (geomConstantsRowScan()), so
// evaluating them together avoids recomputing that row-scan three times.
void buildXPointConstraints(const vector<Point>& coils,
                const vector<Point>& Xpoints,
                const Singularity::PlasmaContext& ctx,
                double psi_ref,
                Eigen::MatrixXd& C, Eigen::VectorXd& d){

    int Nc = coils.size();
    int Nx = Xpoints.size();
    C.resize(3*Nx, Nc);
    d.resize(3*Nx);
    for(int n = 0; n < Nx; ++n){
        for(int i = 0; i < Nc; ++i){
            C(3*n + 0, i) = GBr (coils[i].R, coils[i].Z, Xpoints[n].R, Xpoints[n].Z);
            C(3*n + 1, i) = GBz (coils[i].R, coils[i].Z, Xpoints[n].R, Xpoints[n].Z);
            C(3*n + 2, i) = GPsi(coils[i].R, coils[i].Z, Xpoints[n].R, Xpoints[n].Z);
        }
        Singularity::Result xres = Singularity::evaluateAll(ctx, ctx.jfield.R, ctx.jfield.Z,
                                                              Xpoints[n].R, Xpoints[n].Z);
        d(3*n + 0) = -xres.br;
        d(3*n + 1) = -xres.bz;
        d(3*n + 2) = psi_ref - xres.psi;
    }
}

/*-------------------------------------------------------------------------------------------------------------
TRANSFORM BOUNDARY WITH X-POINTS (calls Python subprocess)
-------------------------------------------------------------------------------------------------------------*/
vector<Point> transformBoundaryWithXpoints(
    const vector<Point>& boundary,
    const vector<Point>& Xpoints,
    int method){

    // Write boundary points to temporary file (denormalized to meters)
    FILE *f_boundary = fopen("_boundary_input.txt", "w");
    if(!f_boundary){
        throw runtime_error("Could not create _boundary_input.txt");
    }
    for(const auto& p : boundary){
        fprintf(f_boundary, "%.10e %.10e\n", p.R * Lo, p.Z * Lo);
    }
    fclose(f_boundary);

    // Write X-point positions to temporary file (denormalized to meters)
    FILE *f_xpoints = fopen("_xpoints_input.txt", "w");
    if(!f_xpoints){
        throw runtime_error("Could not create _xpoints_input.txt");
    }
    for(const auto& xp : Xpoints){
        fprintf(f_xpoints, "%.10e %.10e\n", xp.R * Lo, xp.Z * Lo);
    }
    fclose(f_xpoints);

    // Call Python script (path assumes the solver is run from within a
    // cases/<name>/ directory, two levels below the repository root).
    // method==3 (menu choice "Dual-parameter adaptive, C1 cusp") maps to the
    // Bezier/C1 module (xpoint_transform_c1.py), which only exposes its
    // dual-parameter variant here (script_method=2) -- see
    // docs/FEATURE_XPOINT_C1_CUSP_INTEGRATION_BRIEFING.md for why only the
    // dual-parameter variant was wired in, and why its locality cutoff
    // (cutoff_distance_factor) is on by default inside that script's own
    // CLI entry point rather than passed from here. method==1/2 are
    // unchanged, still the original straight-line-cusp module.
    string script = (method == 3) ? "xpoint_transform_c1.py" : "xpoint_transform.py";
    int script_method = (method == 3) ? 2 : method;
    string cmd = "python3 ../../src/semifree_boundary/" + script + " "
                 "_boundary_input.txt _boundary_output.txt "
                 + to_string(script_method) + " _xpoints_input.txt";

    int ret = system(cmd.c_str());
    if(ret != 0){
        throw runtime_error("Python xpoint_transform.py failed with exit code "
                            + to_string(ret));
    }

    // Read transformed boundary (normalize back)
    vector<Point> transformed;
    ifstream infile("_boundary_output.txt");
    if(!infile.is_open()){
        throw runtime_error("Could not open _boundary_output.txt");
    }
    double R, Z;
    while(infile >> R >> Z){
        transformed.push_back({R / Lo, Z / Lo});
    }
    infile.close();

    if(transformed.empty()){
        throw runtime_error("Transformed boundary is empty");
    }

    // Clean up temporary files
    remove("_boundary_input.txt");
    remove("_xpoints_input.txt");
    remove("_boundary_output.txt");

    return transformed;
}

/*-------------------------------------------------------------------------------------------------------------
REGENERATE Jt CONSISTENT WITH A TRANSFORMED (X-POINT) BOUNDARY (calls gs_solver_xpoint subprocess)
-------------------------------------------------------------------------------------------------------------
Runs the polygonal fixed-boundary Grad-Shafranov solver (gs_solver_xpoint,
built from src/grad_shafranov/main_xpoint.cpp + FIXED_GS_SOLVER_POLY() in
GS_solver.h) on the boundary that transformBoundaryWithXpoints() just
produced, so that Jt is genuinely consistent with that new separatrix instead
of remaining masked to the original, untransformed D-shape (the bug this
feature closes -- see docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf).

Same "write temp files, system(), read back" pattern as
transformBoundaryWithXpoints() above, applied to a different subprocess
(a compiled binary instead of a Python script). Only ever called from the
transform_method==1||2 branch of main(), i.e. only right after this same
process has produced a transformed boundary in memory -- never for
transform_method==0, whose user is assumed to already bring a mutually
consistent boundary+Jt pair.

Passes cfg.Ro/I_plasma/P_axis/B_axis/Psi_b and the SAME mesh
(r_min,r_max,z_min,z_max,npr,npz) already used by this process to the
subprocess's temporary config, so both processes normalize with an identical
Lo=Ro, Io=I_plasma, Jo=Io/Lo^2 -- otherwise the physical Jt.txt produced by
one would be normalized differently when read back by the other (see the
Config struct comment above cfg.Ro/I_plasma/P_axis/B_axis).
-------------------------------------------------------------------------------------------------------------*/
vector<vector<double>> regenerateConsistentJt(
    const Config& cfg,
    const vector<Point>& boundary_norm,
    int npr, int npz){

    // 1. Write the transformed boundary (denormalized to meters, same
    // convention as Dshape_xpoint.txt above) to a temporary polygon file.
    const string poly_path = "_xpoint_boundary_poly.txt";
    FILE *f_poly = fopen(poly_path.c_str(), "w");
    if(!f_poly){
        throw runtime_error("Could not create " + poly_path);
    }
    for(const auto& p : boundary_norm){
        fprintf(f_poly, "%.10e %.10e\n", p.R * Lo, p.Z * Lo);
    }
    fclose(f_poly);

    // 2. Write a temporary config JSON in the format main_xpoint.cpp expects
    // (see src/grad_shafranov/main_xpoint.cpp), reusing the values already
    // loaded into cfg by loadConfig() above.
    json jcfg;
    jcfg["name"]        = "auto_generated_xpoint_consistent_jt";
    jcfg["description"] = "Temporary config auto-generated by SemiFree_Solver.cpp "
                          "(regenerateConsistentJt) to re-solve Jt consistently "
                          "with a transformed X-point boundary.";
    jcfg["geometry"]["Ro"]    = cfg.Ro;
    jcfg["geometry"]["a"]     = 0.0;  // unused by FIXED_GS_SOLVER_POLY (polygon replaces a/kappa/delta)
    jcfg["geometry"]["kappa"] = 0.0;  // unused by FIXED_GS_SOLVER_POLY
    jcfg["geometry"]["delta"] = 0.0;  // unused by FIXED_GS_SOLVER_POLY
    jcfg["constraints"]["I_plasma"] = cfg.I_plasma;
    jcfg["constraints"]["P_axis"]   = cfg.P_axis;
    jcfg["constraints"]["B_axis"]   = cfg.B_axis;
    jcfg["constraints"]["Psi_b"]    = cfg.Psi_b;
    jcfg["mesh"]["r_min"] = cfg.r_min;
    jcfg["mesh"]["r_max"] = cfg.r_max;
    jcfg["mesh"]["z_min"] = cfg.z_min;
    jcfg["mesh"]["z_max"] = cfg.z_max;
    jcfg["mesh"]["npr"]   = npr;
    jcfg["mesh"]["npz"]   = npz;
    jcfg["xpoint"]["boundary_file"] = poly_path;

    const string config_path = "_xpoint_gs_config.json";
    ofstream f_cfg(config_path);
    if(!f_cfg.is_open()){
        throw runtime_error("Could not create " + config_path);
    }
    f_cfg << jcfg.dump(2);
    f_cfg.close();

    // 3. Ensure Results/ exists: GS_solver.h writes several files under it
    // unconditionally, relative to cwd (same convention documented in
    // CLAUDE.md, "fopen(Results/...) fails"). Idempotent if it already
    // exists from a prior full-GS validation run in this case directory.
    system("mkdir -p Results");

    // 4. Invoke gs_solver_xpoint as a subprocess. Same two-levels-below-the
    // repository-root working-directory assumption as the xpoint_transform.py
    // call above (../../build/...). main_xpoint.cpp reads the config path
    // from stdin, exactly like main.cpp/semifree_solver do.
    string cmd = "echo " + config_path + " | ../../build/gs_solver_xpoint "
                "> _gs_solver_xpoint.log 2>&1";
    int ret = system(cmd.c_str());
    if(ret != 0){
        throw runtime_error("gs_solver_xpoint failed with exit code " + to_string(ret) +
                            " (see _gs_solver_xpoint.log in the current directory)");
    }

    // 5. gs_solver_xpoint writes the physical Jt to Results/Jt.txt (same
    // format as the ordinary Jt.txt: npz rows x npr columns, A/m^2). Copy it
    // up to the case directory as Jt_xpoint.txt (matching the naming already
    // used by the manual workflow in the design briefing) and load it with
    // the SAME LoadJphi() used for the non-X-point path, so it is normalized
    // by the identical Jo = I_plasma/Ro^2.
    int cp_ret = system("cp Results/Jt.txt Jt_xpoint.txt");
    if(cp_ret != 0){
        throw runtime_error("Could not copy Results/Jt.txt to Jt_xpoint.txt "
                            "after running gs_solver_xpoint");
    }

    vector<vector<double>> Jphi_new = LoadJphi("Jt_xpoint.txt", npr, npz);

    // Clean up orchestration scratch files only; Jt_xpoint.txt and Results/
    // are kept as diagnostic artifacts.
    remove(poly_path.c_str());
    remove(config_path.c_str());

    return Jphi_new;
}

