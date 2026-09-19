/*============================================================================
  FIXED GRAD-SHAFRANOV SOLVER
  ============================================================================
  This code solves the Grad-Shafranov equation using a fixed plasma boundary
  due by analityc expressions:
  
     Rb = ...
     Zb = ...
  
  Second orden finite difference method was implemeted.
  ============================================================================
  AUTHORS
  ============================================================================
  Juan Camilo Sanchez
  Jesus Eduardo Lopez Duran
  Eduardo Alberto Orozco Ospino
  
  UNIVERSIDAD INDUSTRIAL DE SANTANDER
  BUCARAMANGA-SANTANDER-COLOMBIA
  ============================================================================
  Last modification: April 14, 2025
  ============================================================================*/
  
  
/*============================================================================
  LIBRARIES
  ============================================================================*/
  #include <stdio.h>
  #include <math.h>
  #include <time.h>
  #include <omp.h>

  #include <cstdlib>
  #include <ctime>
  #include <iostream>
  #include <vector>
  #include <utility>
  #include <algorithm>
/*============================================================================
  CTES
  ============================================================================*/
  #define  mu_o    1.256637061e-06
/*----------------------------------------------------------------------------
  Fixed RNG seed for the initial guess. The SOR loops seed Psi with random
  values; seeding from time(NULL) made every run produce a different answer,
  which is unacceptable for reproducible results and makes before/after
  comparison impossible. Override at compile time with -DGS_SEED=<n>.
  ----------------------------------------------------------------------------*/
  #ifndef GS_SEED
  #define  GS_SEED  12345u
  #endif
/*----------------------------------------------------------------------------
  Convergence control.

  GS_TOL_RES is the stopping tolerance on the RELATIVE Grad-Shafranov residual
  max|Delta* Psi + r*Jt| / max|r*Jt|, i.e. on how well the equation is actually
  satisfied. The previous criterion tested only the per-iteration change
  max|Psi - Psi_old|, which says nothing about the residual: with the observed
  slow geometric convergence (rate ~0.99965/iteration) a small step size
  understated the true distance to the fixed point by a factor of ~2800, and the
  solver stopped roughly 1% short of the solution while alpha was still drifting.

  Calibrated, not guessed. An 80000-iteration run with the test disabled shows
  the relative residual falling geometrically (~6500 iterations per decade) all
  the way to roundoff (~2e-12); it does NOT plateau at the O(h^2) truncation
  level, because this is the residual of the DISCRETE equation, which the
  discrete solution satisfies exactly at convergence. (The O(h^2) truncation
  error is the separate gap between the discrete and continuum solutions and is
  not visible here.) The tolerance is therefore free to choose:

      residual   iterations (Soloviev, 151x261, FIXED)
      1e-2       10098      <- where the OLD criterion effectively stopped
      1e-4       23082
      1e-6       36083
      1e-8       49084      <- GS_TOL_RES
      1e-10      62113

  1e-8 sits about four orders of magnitude below the O(h^2) ~ 1e-4 discretization
  error, so iteration error is negligible against discretization error and the
  result is limited only by the mesh.

  Set GS_TOL_RES=0 to disable the residual test (used to measure the curve above).
  ----------------------------------------------------------------------------*/
  #ifndef GS_TOL_RES
  #define  GS_TOL_RES  1.0e-8
  #endif

  #ifndef GS_ITMAX
  #define  GS_ITMAX  1000000
  #endif
/*============================================================================
  SPATIAL MESH VARIABLES
  ============================================================================*/
  int npr,npz;
  
  double *r,*z;
  double r_min,r_max;
  double z_min,z_max;
  double hr,hz;
  double Lr,Lz;
/*============================================================================
  VARIABLE FOR INTERPOLATION
  ============================================================================*/
  int npi;
  int **FLAG,*Ii,*Ki;
  double *La,*Lb,*Lc,*Ld;
/*============================================================================
  SUCCESSIVE OVER-RELAXATION VARIABLES
  ============================================================================*/
  int itmax;
  double tol;
/*============================================================================
  PLASMA PARAMETERS
  ============================================================================*/
  double Ro,a,kappa,delta;
  double Ip,Pa,Ba,go;
  double Psia,Psib,alpha;
  double raxis,zaxis;
/*============================================================================
  FIELD ARRAYS
  ============================================================================*/
  double **Psi,**PsiN;
  double **Jt;
  double **P;
/*============================================================================
  AUXILIAR VARIABLES
  ============================================================================*/
  double **Psi_old,*rc,*zc,*Ic; int Nc = 0;
/*============================================================================
  NORMALIZATION VALUES
  ============================================================================*/
  double Io;
  double Jo;
  double Lo;
  double Po;
  double Bo;
  double Go;
  double Psio;
/*============================================================================
  MAIN MODULES
  ============================================================================*/
  void ALPHA();
  void BOBINAS(void);
  void SAVEMESH(void);
  void BUILDMESH(void);
  void DEFARRAYS(void);
  void DELARRAYS(void);
  void SOR_FIXED_BOUNDARY(void);
  void SOR_SEMIFREE_BOUNDARY(void);
  void SAVEDSHAPE(void);
  void INITVALUES(void);
  void INTERPOLATION(void);
  void NORMALIZATION(void);
  void CRITICAL_POINTS(void);
  void CURRENT_DENSITY(void);
  double RESIDUAL(void);
  void PRESSURE(void);
  void POINTSFORINTERPOLATION(void);
  
  double F_GREEN(double rnp,double znp,double rp,double zp);
  double PSI_BOUNDARY(double rp, double zp);
  void CONDICIONES_DE_FRONTERA(void);
  
  double g(double psiN);
  double p(double psiN);
  double dp_dpsiN(double psiN);
  double dg_dpsiN(double psiN);
  double zboundary(double theta);
  double rboundary(double theta);
  double dr(double **F,int i,int k);
  double dz(double **F,int i,int k);
  double thetaboundary(double r,double z);
/*----------------------------------------------------------------------------
  POLYGONAL BOUNDARY GENERALIZATION (added 2026-09-10)

  Generalizes FIXED_GS_SOLVER() to an arbitrary simple closed polygon boundary
  (with or without X-points), instead of only the analytic D-shape
  (Ro,a,kappa,delta). See docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf for
  the full derivation. Used to regenerate a Jt consistent with a boundary that
  has already been deformed toward an X-point by
  src/semifree_boundary/xpoint_transform.py.
  ----------------------------------------------------------------------------*/
  bool inside_polygon(double r0,double z0,
                       const std::vector<double>& Rp,const std::vector<double>& Zp);
  double ray_distance_minus_R(double r0,double z0,double hr_,
                               const std::vector<double>& Rp,const std::vector<double>& Zp);
  double ray_distance_plus_R(double r0,double z0,double hr_,
                              const std::vector<double>& Rp,const std::vector<double>& Zp);
  double ray_distance_minus_Z(double r0,double z0,double hz_,
                               const std::vector<double>& Rp,const std::vector<double>& Zp);
  double ray_distance_plus_Z(double r0,double z0,double hz_,
                              const std::vector<double>& Rp,const std::vector<double>& Zp);
  bool polygon_is_simple(const std::vector<double>& Rp,const std::vector<double>& Zp);
  void POINTSFORINTERPOLATION_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp);
  void BUILDMESH_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp);
  void SAVEDSHAPE_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp);
  void FIXED_GS_SOLVER_POLY(const std::vector<std::pair<double,double>>& boundary_poly_m,
                             double Dshape[4],double constrains[4],double mesh[6]);
/*============================================================================*/
  
  
/*============================================================================
  INICIO PROGRAMA PRINCIPAL
  ============================================================================*/
  void FIXED_GS_SOLVER(double Dshape[4],double constrains[4],double mesh[6]){
   
   srand(GS_SEED);
   
   system("cls");
   system("echo %DATE%");
   printf("EXECUTION IN PROGRESS\n\n");
   
 /*---------------------------------------------------------------------------
   PLASMA PARAMETERS
   ---------------------------------------------------------------------------*/
   Ro    = Dshape[0];
   a     = Dshape[1];
   kappa = Dshape[2];
   delta = Dshape[3];
   
   Ip   = constrains[0];
   Pa   = constrains[1];
   Ba   = constrains[2];
   go   = Ba*Ro;
   Psib = constrains[3];
 /*---------------------------------------------------------------------------
   MESH PARAMETERS
   ---------------------------------------------------------------------------*/
   r_min = mesh[0];
   r_max = mesh[1];
   z_min = mesh[2];
   z_max = mesh[3];
   npr   = mesh[4];
   npz   = mesh[5];
   
   Lr = r_max - r_min;
   Lz = z_max - z_min;
   
   printf("npr = %d  npz = %d\n",npr,npz);
 /*---------------------------------------------------------------------------*/
   
   DEFARRAYS();
   NORMALIZATION();
   BUILDMESH();
   SOR_FIXED_BOUNDARY();
   DELARRAYS();
   
   printf("\nSUCESSFUL EXECUTION\n");
   system("echo %DATE%");
  }
/*============================================================================
   FIN PROGRAMA PRINCIPAL
  ============================================================================*/


/*============================================================================
  INICIO PROGRAMA PRINCIPAL (FRONTERA POLIGONAL ARBITRARIA)
  ============================================================================
  Generalizacion de FIXED_GS_SOLVER() para una frontera dada como poligono
  cerrado simple (R,Z) en METROS, en vez de la forma analitica D-shape. Ver
  docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf.

  Dshape[4]: SOLO Dshape[0]=Ro se usa (referencia de normalizacion, Lo=Ro).
  Dshape[1..3] (a,kappa,delta) se aceptan por firma homogenea con
  FIXED_GS_SOLVER pero se ignoran: no tienen significado para un poligono
  arbitrario y rboundary()/zboundary()/thetaboundary() no se invocan en esta
  ruta.
  ============================================================================*/
  void FIXED_GS_SOLVER_POLY(const std::vector<std::pair<double,double>>& boundary_poly_m,
                             double Dshape[4],double constrains[4],double mesh[6]){

   srand(GS_SEED);

   system("cls");
   system("echo %DATE%");
   printf("EXECUTION IN PROGRESS (POLYGONAL FIXED BOUNDARY)\n\n");

 /*---------------------------------------------------------------------------
   PLASMA PARAMETERS (a,kappa,delta ignored: not meaningful for a polygon)
   ---------------------------------------------------------------------------*/
   Ro    = Dshape[0];
   a     = 0.0;
   kappa = 0.0;
   delta = 0.0;

   Ip   = constrains[0];
   Pa   = constrains[1];
   Ba   = constrains[2];
   go   = Ba*Ro;
   Psib = constrains[3];
 /*---------------------------------------------------------------------------
   MESH PARAMETERS
   ---------------------------------------------------------------------------*/
   r_min = mesh[0];
   r_max = mesh[1];
   z_min = mesh[2];
   z_max = mesh[3];
   npr   = mesh[4];
   npz   = mesh[5];

   Lr = r_max - r_min;
   Lz = z_max - z_min;

   printf("npr = %d  npz = %d\n",npr,npz);
 /*---------------------------------------------------------------------------
   LOAD AND VALIDATE THE INPUT POLYGON (still in meters at this point)
   ---------------------------------------------------------------------------*/
   if(boundary_poly_m.size() < 3){
    printf("ERROR: boundary_poly_m must have at least 3 points (got %zu).\n",
           boundary_poly_m.size());
    exit(1);
   }

   std::vector<double> Rp_m(boundary_poly_m.size()), Zp_m(boundary_poly_m.size());
   for(size_t j=0;j<boundary_poly_m.size();j++){
    Rp_m[j] = boundary_poly_m[j].first;
    Zp_m[j] = boundary_poly_m[j].second;
   }

 //Drop consecutive (including the cyclic closing pair, last-to-first)
 //near-coincident points BEFORE the simplicity check. This is input
 //sanitization of a redundant vertex, not "fixing" a genuinely bad polygon:
 //a first/last point separated by ~1e-14 m (floating-point noise from the
 //analytic-D-shape resampling loop in SAVEDSHAPE(), t in [0,2*pi) landing a
 //last sample a hair short of a full turn) makes the wrap-around edge have
 //near-zero length, which spuriously registers as "intersecting" its
 //neighbor's neighbor under exact orientation/on-segment tests -- found
 //while running the regression test against cases/soloviev/Dshape.txt (see
 //docs/reports/XPOINT_CONSISTENT_JT_IMPLEMENTATION.tex). Threshold 1e-9 m is
 //far below any real geometric feature at this project's mesh scale (~1 m)
 //and far above the ~1e-14 m duplicate observed.
   {
    const double dedup_tol_m = 1.0e-9;
    std::vector<double> Rp_dedup, Zp_dedup;
    int M0 = (int)Rp_m.size();
    for(int j=0;j<M0;j++){
     int jprev = (j==0) ? M0-1 : j-1;
     double d = hypot(Rp_m[j]-Rp_m[jprev], Zp_m[j]-Zp_m[jprev]);
     if(j==0 || d > dedup_tol_m){
      Rp_dedup.push_back(Rp_m[j]);
      Zp_dedup.push_back(Zp_m[j]);
     }
     else{
      printf("NOTE: dropped near-duplicate boundary vertex %d "
             "(%.3e m from its predecessor, tol=%.1e m).\n",j,d,dedup_tol_m);
     }
    }
    //Also check the (possibly new) last point against point 0 cyclically,
    //in case the duplicate was the closing pair itself (point 0 vs old last).
    if(Rp_dedup.size()>=2){
     double d = hypot(Rp_dedup.back()-Rp_dedup.front(), Zp_dedup.back()-Zp_dedup.front());
     if(d <= dedup_tol_m){
      printf("NOTE: dropped near-duplicate closing boundary vertex "
             "(%.3e m from vertex 0, tol=%.1e m).\n",d,dedup_tol_m);
      Rp_dedup.pop_back();
      Zp_dedup.pop_back();
     }
    }
    Rp_m = Rp_dedup;
    Zp_m = Zp_dedup;
   }

   if(boundary_poly_m.size() >= 3 && Rp_m.size() < 3){
    printf("ERROR: fewer than 3 distinct boundary vertices remain after "
           "removing near-duplicates.\n");
    exit(1);
   }

   if(!polygon_is_simple(Rp_m,Zp_m)){
    printf("ERROR: the input boundary polygon is self-intersecting.\n");
    printf("Aborting rather than attempting to silently repair it.\n");
    printf("Review the X-point transform parameters used to generate this\n");
    printf("boundary (src/semifree_boundary/xpoint_transform.py).\n");
    exit(1);
   }
 /*---------------------------------------------------------------------------*/

   DEFARRAYS();
   NORMALIZATION();

 //Normalize the polygon by the SAME Lo=Ro used for the mesh, exactly once,
 //here (the points arrive in meters).
   std::vector<double> Rp(Rp_m.size()), Zp(Zp_m.size());
   for(size_t j=0;j<Rp_m.size();j++){
    Rp[j] = Rp_m[j]/Lo;
    Zp[j] = Zp_m[j]/Lo;
   }

   BUILDMESH_POLY(Rp,Zp);
   SOR_FIXED_BOUNDARY();
   DELARRAYS();

   printf("\nSUCESSFUL EXECUTION\n");
   system("echo %DATE%");
  }
/*============================================================================
   FIN PROGRAMA PRINCIPAL
  ============================================================================*/


/*============================================================================
  INICIO PROGRAMA PRINCIPAL
  ============================================================================*/
  void SEMIFREE_GS_SOLVER(double Dshape[4],double constrains[4],double mesh[6]){
   
   srand(GS_SEED);
   
   system("cls");
   system("echo %DATE%");
   printf("EXECUTION IN PROGRESS\n\n");
   
 /*---------------------------------------------------------------------------
   PLASMA PARAMETERS
   ---------------------------------------------------------------------------*/
   Ro    = Dshape[0];
   a     = Dshape[1];
   kappa = Dshape[2];
   delta = Dshape[3];
   
   Ip   = constrains[0];
   Pa   = constrains[1];
   Ba   = constrains[2];
   go   = Ba*Ro;
   Psib = constrains[3];
 /*---------------------------------------------------------------------------
   MESH PARAMETERS
   ---------------------------------------------------------------------------*/
   r_min = mesh[0];
   r_max = mesh[1];
   z_min = mesh[2];
   z_max = mesh[3];
   npr   = mesh[4];
   npz   = mesh[5];
   
   Lr = r_max - r_min;
   Lz = z_max - z_min;
   
   printf("npr = %d  npz = %d\n",npr,npz);
 /*---------------------------------------------------------------------------*/
   
   DEFARRAYS();
   NORMALIZATION();
   BUILDMESH();
   BOBINAS();
   CONDICIONES_DE_FRONTERA();
   SOR_SEMIFREE_BOUNDARY();
   DELARRAYS();
   
   printf("\nSUCESSFUL EXECUTION\n");
   system("echo %DATE%");
  }
/*============================================================================
   FIN PROGRAMA PRINCIPAL
  ============================================================================*/
  
   
/*============================================================================
  DEFINE ARRAYS
  ============================================================================*/
  void DEFARRAYS(void){
   
   r = new double[npr];
   z = new double[npz];
   
   rc = new double[Nc];
   zc = new double[Nc];
   Ic = new double[Nc];
   
   npi = 10*fmax(npr,npz);
   
   Ii = new int[npi];
   Ki = new int[npi];
   La = new double[npi];
   Lb = new double[npi];
   Lc = new double[npi];
   Ld = new double[npi];
   
   FLAG    = new int*[npr];
   Psi     = new double*[npr];
   PsiN    = new double*[npr];
   Psi_old = new double*[npr];
   Jt      = new double*[npr];
   P       = new double*[npr];
   
   for(int i=0;i<npr;i++){
    
    FLAG[i]    = new int[npz];
    Psi[i]     = new double[npz];
    PsiN[i]    = new double[npz];
    Psi_old[i] = new double[npz];
    Jt[i]      = new double[npz];
    P[i]       = new double[npz];
    
   }
   
   printf("Arrays defined\n");
   
   INITVALUES();
  }
/*============================================================================*/
  
  
/*============================================================================
  INIZIALIZATE ARRAYS
  ============================================================================*/
  void INITVALUES(void){
   
   for(int k=0;k<npz;k++){
    z[k] = 0.0;
   }
   
   for(int i=0;i<npr;i++){
    r[i] = 0.0;
   }
   
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     
     FLAG[i][k]    = 0;
     Psi[i][k]     = 0.0;
     PsiN[i][k]    = 0.0;
     Psi_old[i][k] = 0.0;
     Jt[i][k]      = 0.0;
     P[i][k]       = 0.0;
     
    }
   }
   
   printf("Arrays inicialized\n");
  }
/*============================================================================*/
  
  
/*============================================================================
  DELETE ARRAYS
  ============================================================================*/
  void DELARRAYS(void){
   
   for(int i=0;i<npr;i++){
    delete[] FLAG[i];
    delete[] Psi[i];
    delete[] PsiN[i];
    delete[] Psi_old[i];
    delete[] Jt[i];
    delete[] P[i];
   }
   delete[] FLAG;
   delete[] Psi;
   delete[] PsiN;
   delete[] Psi_old;
   delete[] Jt;
   delete[] P;
   
   delete[] Ii;
   delete[] Ki;
   delete[] La;
   delete[] Lb;
   delete[] Lc;
   delete[] Ld;
   delete[] r;
   delete[] z;
   delete[] rc;
   delete[] zc;
   delete[] Ic;
   
   printf("Arrays deleted\n");
  }
/*============================================================================*/
  
  
/*============================================================================
  NORMALIZATION
  ============================================================================*/
  void NORMALIZATION(void){
   
   Lo   = Ro;
   Io   = Ip;
   Jo   = Io/pow(Lo,2);
   Psio = mu_o*Jo*pow(Lo,3);
   
   Po = Psio*Jo/Lo;
   Bo = Psio/pow(Lo,2);
   Go = Bo*Lo;
   
   r_min = r_min/Lo;
   r_max = r_max/Lo;
   z_min = z_min/Lo;
   z_max = z_max/Lo;
   
   Lr = Lr/Lo;
   Lz = Lz/Lo;
   
   Ro = Ro/Lo;
   a  = a/Lo;
   
   Ip   = Ip/Io;
   Pa   = Pa/Po;
   Ba   = Ba/Bo;
   go   = go/Go;
   Psib = Psib/Psio;
   
   printf("Values Normalized\n");
  }
/*============================================================================*/
  
  
/*============================================================================
  BUILD MESH
  ============================================================================*/
  void BUILDMESH(void){
   
   hr = Lr/(npr-1.);
   
   for(int i=0;i<npr;i++){
    r[i] = r_min + i*hr;
   }
   
   hz = Lz/(npz-1.);
   
   for(int k=0;k<npz;k++){
    z[k] = z_min + k*hz;
   }
   
   POINTSFORINTERPOLATION();
   SAVEMESH();
   SAVEDSHAPE();

   printf("Mesh built\n");
  }
/*============================================================================*/


/*============================================================================
  BUILD MESH -- POLYGONAL BOUNDARY VERSION

  Identical r[]/z[] construction (geometry-independent); dispatches to the
  polygonal classification/interpolation-distance and Dshape-export routines
  instead of the analytic-D-shape ones.
  ============================================================================*/
  void BUILDMESH_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp){

   hr = Lr/(npr-1.);

   for(int i=0;i<npr;i++){
    r[i] = r_min + i*hr;
   }

   hz = Lz/(npz-1.);

   for(int k=0;k<npz;k++){
    z[k] = z_min + k*hz;
   }

   POINTSFORINTERPOLATION_POLY(Rp,Zp);
   SAVEMESH();
   SAVEDSHAPE_POLY(Rp,Zp);

   printf("Mesh built (polygonal boundary)\n");
  }
/*============================================================================*/


/*============================================================================
  SAVE MESH IN TXT FILE
  ============================================================================*/
  void SAVEMESH(void){
   
   FILE *out;
   out = fopen("Results/rz_mesh.txt","w");
   
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     fprintf(out,"%e %e %d\n",r[i]*Lo,z[k]*Lo,FLAG[i][k]);
    }
   }
   
   fclose(out);
  }
/*============================================================================*/
  
  
/*============================================================================
  DETECT POINTS AND VALUES FOR INTERPOLATION
  ============================================================================*/
  void POINTSFORINTERPOLATION(void){
   
 //Identifica puntos dentro (FLAG = 0) y fuera (FLAG = 9) de la frontera......
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     
     if( fabs(z[k]) >= a*kappa ){
      FLAG[i][k] = 9;
     }
     else{
      double theta = asin( z[k]/(a*kappa) );
      double Rbmax = rboundary(theta);
      double Rbmin = rboundary(M_PI-theta);
      if( (r[i] >= Rbmax) | (r[i] <= Rbmin) ){
       FLAG[i][k] = 9;
      }
      else FLAG[i][k] = 0;
     }
     
    }
   }
 //...........................................................................
   
 //Identifica cada unos de los puntos que se deben interpolar y las..........\
   respectivas distancias para la interpolacion...............................
   npi = 0;
   for(int k=1;k<npz-1;k++){
    for(int i=1;i<npr-1;i++){
     
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]!=9)&\
         (FLAG[i-1][k]==9)&(FLAG[i][k+1]!=9)&(FLAG[i][k-1]!=9) ){
     //CASO 1
      double theta = asin( z[k]/(a*kappa) );
      La[npi] = r[i] - rboundary(M_PI-theta);
      Lb[npi] = hr;
      Lc[npi] = hz;
      Ld[npi] = hz;
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 1;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]!=9)&\
         (FLAG[i-1][k]==9)&(FLAG[i][k+1]==9)&(FLAG[i][k-1]!=9) ){
     //CASO 2
      double theta = asin( z[k]/(a*kappa) );
      La[npi] = r[i] - rboundary(M_PI-theta);
      Lb[npi] = hr;
      theta = thetaboundary(r[i],z[k]);
      Lc[npi] = zboundary(theta) - z[k];
      Ld[npi] = hz;
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 2;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]!=9)&\
         (FLAG[i-1][k]!=9)&(FLAG[i][k+1]==9)&(FLAG[i][k-1]!=9) ){
     //CASO 3
      La[npi] = hr;
      Lb[npi] = hr;
      double theta = thetaboundary(r[i],z[k]);
      Lc[npi] = zboundary(theta) - z[k];
      Ld[npi] = hz;
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 3;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]==9)&\
         (FLAG[i-1][k]!=9)&(FLAG[i][k+1]==9)&(FLAG[i][k-1]!=9) ){
     //CASO 4
      La[npi] = hr;
      double theta = asin( z[k]/(a*kappa) );
      Lb[npi] = rboundary(theta) - r[i];
      theta = thetaboundary(r[i],z[k]);
      Lc[npi] = zboundary(theta) - z[k];
      Ld[npi] = hz;
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 4;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]==9)&\
         (FLAG[i-1][k]!=9)&(FLAG[i][k+1]!=9)&(FLAG[i][k-1]!=9) ){
     //CASO 5
      La[npi] = hr;
      double theta = asin( z[k]/(a*kappa) );
      Lb[npi] = rboundary(theta) - r[i];
      Lc[npi] = hz;
      Ld[npi] = hz;
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 5;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]==9)&\
         (FLAG[i-1][k]!=9)&(FLAG[i][k+1]!=9)&(FLAG[i][k-1]==9) ){
     //CASO 6
      La[npi] = hr;
      double theta = asin( z[k]/(a*kappa) );
      Lb[npi] = rboundary(theta) - r[i];
      Lc[npi] = hz;
      theta = thetaboundary(r[i],z[k]);
      Ld[npi] = zboundary(theta) - fabs(z[k]);
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 6;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]!=9)&\
         (FLAG[i-1][k]!=9)&(FLAG[i][k+1]!=9)&(FLAG[i][k-1]==9) ){
     //CASO 7
      La[npi] = hr;
      Lb[npi] = hr;
      double theta = thetaboundary(r[i],z[k]);
      Lc[npi] = hz;
      Ld[npi] = zboundary(theta) - fabs(z[k]);
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 7;
      npi++;
     }
     if( (FLAG[i][k]==0)&(FLAG[i+1][k]!=9)&\
         (FLAG[i-1][k]==9)&(FLAG[i][k+1]!=9)&(FLAG[i][k-1]==9) ){
     //CASO 8
      double theta = asin( z[k]/(a*kappa) );
      La[npi] = r[i] - rboundary(M_PI-theta);
      Lb[npi] = hr;
      Lc[npi] = hz;
      theta = thetaboundary(r[i],z[k]);
      Ld[npi] = zboundary(theta) - fabs(z[k]);
      
      Ii[npi] = i;
      Ki[npi] = k;
      FLAG[i][k] = 8;
      npi++;
     }
     
    }
   }
 //...........................................................................
   
   printf("Puntos a interpolar: npi = %d\n",npi);

  }
/*============================================================================*/


/*============================================================================
  POLYGONAL BOUNDARY HELPERS (added 2026-09-10)
  ============================================================================
  Replace the D-shape-specific classification (fabs(z[k])>=a*kappa + asin +
  rboundary, which assumes exactly one theta and two boundary crossings per
  height z -- a property that breaks near an X-point cusp) with generic,
  shape-agnostic point-in-polygon and ray-casting primitives. See
  docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf section 5.
  ============================================================================*/

/*----------------------------------------------------------------------------
  POINT-IN-POLYGON: even-odd / crossing-number rule. Valid for convex and
  non-convex simple polygons alike. The strict ">" comparison on both segment
  endpoints (rather than ">=") is what makes this robust when a horizontal
  ray happens to pass exactly through a polygon vertex: at most one of the two
  segments meeting at that vertex is ever counted as "crossing", so a shared
  vertex is never double-counted. The same convention is reused, for the same
  reason, in the ray_distance_* functions below.
  ----------------------------------------------------------------------------*/
  bool inside_polygon(double r0,double z0,
                       const std::vector<double>& Rp,const std::vector<double>& Zp){
   int M = Rp.size();
   bool in = false;
   for(int j=0,k=M-1;j<M;k=j++){
    bool cross = ((Zp[j] > z0) != (Zp[k] > z0)) &&
                 (r0 < (Rp[k]-Rp[j])*(z0-Zp[j])/(Zp[k]-Zp[j]) + Rp[j]);
    if(cross) in = !in;
   }
   return in;
  }
/*============================================================================*/


/*----------------------------------------------------------------------------
  RAY-CASTING DISTANCES TO THE POLYGON, ONE PER CARDINAL DIRECTION.

  Each returns the distance from (r0,z0) to the NEAREST polygon crossing in
  that direction, capped at h (hr or hz): if the neighboring grid point at
  that distance is still inside the polygon, no crossing closer than h exists
  and h itself is returned (matching the analytic path's convention of
  fixing the distance at exactly hr/hz in directions that do not touch the
  boundary, see INTERPOLATION() below).
  ----------------------------------------------------------------------------*/
  double ray_distance_minus_R(double r0,double z0,double hr_,
                               const std::vector<double>& Rp,const std::vector<double>& Zp){
   double best = 1e300;
   int M = Rp.size();
   for(int j=0,k=M-1;j<M;k=j++){
    double za=Zp[j], zb=Zp[k];
    if((za>z0)==(zb>z0)) continue; //segment does not cross z=z0
    double t = (z0-za)/(zb-za);
    double r_cross = Rp[j] + t*(Rp[k]-Rp[j]);
    double d = r0 - r_cross; //crossing to the LEFT (smaller r)
    if(d>0.0 && d<best) best = d;
   }
   return std::min(best,hr_);
  }

  double ray_distance_plus_R(double r0,double z0,double hr_,
                              const std::vector<double>& Rp,const std::vector<double>& Zp){
   double best = 1e300;
   int M = Rp.size();
   for(int j=0,k=M-1;j<M;k=j++){
    double za=Zp[j], zb=Zp[k];
    if((za>z0)==(zb>z0)) continue;
    double t = (z0-za)/(zb-za);
    double r_cross = Rp[j] + t*(Rp[k]-Rp[j]);
    double d = r_cross - r0; //crossing to the RIGHT (larger r)
    if(d>0.0 && d<best) best = d;
   }
   return std::min(best,hr_);
  }

  double ray_distance_minus_Z(double r0,double z0,double hz_,
                               const std::vector<double>& Rp,const std::vector<double>& Zp){
   double best = 1e300;
   int M = Rp.size();
   for(int j=0,k=M-1;j<M;k=j++){
    double ra=Rp[j], rb=Rp[k];
    if((ra>r0)==(rb>r0)) continue; //segment does not cross r=r0
    double t = (r0-ra)/(rb-ra);
    double z_cross = Zp[j] + t*(Zp[k]-Zp[j]);
    double d = z0 - z_cross; //crossing BELOW (smaller z)
    if(d>0.0 && d<best) best = d;
   }
   return std::min(best,hz_);
  }

  double ray_distance_plus_Z(double r0,double z0,double hz_,
                              const std::vector<double>& Rp,const std::vector<double>& Zp){
   double best = 1e300;
   int M = Rp.size();
   for(int j=0,k=M-1;j<M;k=j++){
    double ra=Rp[j], rb=Rp[k];
    if((ra>r0)==(rb>r0)) continue;
    double t = (r0-ra)/(rb-ra);
    double z_cross = Zp[j] + t*(Zp[k]-Zp[j]);
    double d = z_cross - z0; //crossing ABOVE (larger z)
    if(d>0.0 && d<best) best = d;
   }
   return std::min(best,hz_);
  }
/*============================================================================*/


/*----------------------------------------------------------------------------
  SIMPLE-POLYGON CHECK (no self-intersections), O(M^2) segment-pair test via
  the standard orientation (CCW) criterion -- the same criterion
  xpoint_transform.py already uses internally for its own loop detection
  (_ccw / _segments_intersect), ported to C++ here. Trivial cost for the
  M~100-500 point boundaries this project uses.
  ----------------------------------------------------------------------------*/
  static int gs_poly_orientation(double ax,double ay,double bx,double by,double cx,double cy){
   double val = (by-ay)*(cx-bx) - (bx-ax)*(cy-by);
   if(fabs(val) < 1e-14) return 0; //collinear
   return (val>0.0) ? 1 : 2;
  }

  static bool gs_poly_on_segment(double px,double py,double qx,double qy,double rx,double ry){
   return (qx <= std::max(px,rx)+1e-12 && qx >= std::min(px,rx)-1e-12 &&
           qy <= std::max(py,ry)+1e-12 && qy >= std::min(py,ry)-1e-12);
  }

  static bool gs_poly_segments_intersect(double p1x,double p1y,double p2x,double p2y,
                                          double p3x,double p3y,double p4x,double p4y){
   int o1 = gs_poly_orientation(p1x,p1y,p2x,p2y,p3x,p3y);
   int o2 = gs_poly_orientation(p1x,p1y,p2x,p2y,p4x,p4y);
   int o3 = gs_poly_orientation(p3x,p3y,p4x,p4y,p1x,p1y);
   int o4 = gs_poly_orientation(p3x,p3y,p4x,p4y,p2x,p2y);

   if(o1!=o2 && o3!=o4) return true;

   if(o1==0 && gs_poly_on_segment(p1x,p1y,p3x,p3y,p2x,p2y)) return true;
   if(o2==0 && gs_poly_on_segment(p1x,p1y,p4x,p4y,p2x,p2y)) return true;
   if(o3==0 && gs_poly_on_segment(p3x,p3y,p1x,p1y,p4x,p4y)) return true;
   if(o4==0 && gs_poly_on_segment(p3x,p3y,p2x,p2y,p4x,p4y)) return true;

   return false;
  }

  bool polygon_is_simple(const std::vector<double>& Rp,const std::vector<double>& Zp){
   int M = Rp.size();
   if(M<3) return false;
   for(int i=0;i<M;i++){
    int i2 = (i+1)%M;
    for(int j=i+1;j<M;j++){
     int j2 = (j+1)%M;
     if(i2==j || j2==i) continue; //adjacent segments share an endpoint
     if(gs_poly_segments_intersect(Rp[i],Zp[i],Rp[i2],Zp[i2],Rp[j],Zp[j],Rp[j2],Zp[j2]))
      return false;
    }
   }
   return true;
  }
/*============================================================================*/


/*============================================================================
  DETECT POINTS AND VALUES FOR INTERPOLATION -- POLYGONAL BOUNDARY VERSION

  Classifies FLAG[i][k] into only 0 (interior), 1 (ring -- at least one
  cardinal neighbor outside, needs interpolation), 9 (exterior). FLAG never
  takes values 2..8 on this path (see the design decision recorded in
  docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf section 5.3): all downstream
  consumers of FLAG (RESIDUAL, CURRENT_DENSITY, ALPHA, CRITICAL_POINTS) only
  ever test ==0 or !=9, and the only place that reads FLAG==1..8 as a
  discriminant is the switch removed from INTERPOLATION() below, which is
  replaced there by a formula that reads directly off La/Lb/Lc/Ld instead of
  FLAG.
  ============================================================================*/
  void POINTSFORINTERPOLATION_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp){

   double eps_ring = 1.0e-12; //purely numerical guard against float equality,
                               //unrelated to the eps_geom used in INTERPOLATION()
                               //or to the distance floor below.

 //Piso de distancia (regularizacion Shortley-Weller, decision 3 del brief):
 //epsilon=0.05 por defecto; documentado aqui si se sube a 0.1 (ver Parte A.3).
   const double eps_floor = 0.05;

 //Clasifica FLAG[i][k] en 0 (dentro) / 9 (fuera) via point-in-polygon.........
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     FLAG[i][k] = inside_polygon(r[i],z[k],Rp,Zp) ? 0 : 9;
    }
   }
 //...........................................................................

 //Identifica los nodos de anillo (FLAG=1) y sus 4 distancias..................
   npi = 0;
   int npi_capacity = 10*(int)fmax(npr,npz);

   int n_isolated = 0;

   for(int k=1;k<npz-1;k++){
    for(int i=1;i<npr-1;i++){

     if(FLAG[i][k]!=0) continue;

     double La_v = ray_distance_minus_R(r[i],z[k],hr,Rp,Zp);
     double Lb_v = ray_distance_plus_R (r[i],z[k],hr,Rp,Zp);
     double Lc_v = ray_distance_plus_Z (r[i],z[k],hz,Rp,Zp);
     double Ld_v = ray_distance_minus_Z(r[i],z[k],hz,Rp,Zp);

     bool is_ring = (La_v < hr-eps_ring) || (Lb_v < hr-eps_ring) ||
                    (Lc_v < hz-eps_ring) || (Ld_v < hz-eps_ring);

     if(!is_ring) continue;

     if(La_v>=hr-eps_ring && Lb_v>=hr-eps_ring && Lc_v>=hz-eps_ring && Ld_v>=hz-eps_ring){
      n_isolated++; //defensive: cannot happen given is_ring above, kept for clarity
     }

     //Piso de distancia (Shortley-Weller), SOLO aqui -- nunca en INTERPOLATION()
     //ni en la ruta analitica existente.
     La_v = std::max(La_v, eps_floor*hr);
     Lb_v = std::max(Lb_v, eps_floor*hr);
     Lc_v = std::max(Lc_v, eps_floor*hz);
     Ld_v = std::max(Ld_v, eps_floor*hz);

     if(npi>=npi_capacity){
      printf("ERROR: POINTSFORINTERPOLATION_POLY ring point count exceeded\n");
      printf("the preallocated capacity (%d). Increase npi in DEFARRAYS().\n",npi_capacity);
      exit(1);
     }

     La[npi]=La_v; Lb[npi]=Lb_v; Lc[npi]=Lc_v; Ld[npi]=Ld_v;
     Ii[npi]=i; Ki[npi]=k;
     FLAG[i][k] = 1;
     npi++;
    }
   }
 //...........................................................................

 //Celda aislada: los 4 vecinos cardinales de un nodo de anillo estan fuera....
 //Extremadamente improbable a resoluciones razonables; se advierte, no se
 //aborta (la formula generica de INTERPOLATION() sigue siendo valida:
 //Fa=Fb=Fc=Fd=Psib en ese caso).
   for(int t=0;t<npi;t++){
    int i=Ii[t], k=Ki[t];
    if(FLAG[i-1][k]==9 && FLAG[i+1][k]==9 && FLAG[i][k+1]==9 && FLAG[i][k-1]==9){
     printf("WARNING: isolated ring cell at i=%d k=%d (R=%e m, Z=%e m).\n",
            i,k,r[i]*Lo,z[k]*Lo);
     printf("  All 4 cardinal neighbors are exterior -- mesh may be too\n");
     printf("  coarse for this boundary geometry near this point.\n");
    }
   }

   printf("Puntos a interpolar (poligono): npi = %d\n",npi);

  }
/*============================================================================*/


/*============================================================================
  SUCCESSIVE OVER-RELAXATION METHOD
  ============================================================================*/
  void SOR_FIXED_BOUNDARY(void){
   
   tol = 1.e-3*(pow(hr,2)+pow(hz,2));
   printf("TOL = %e\n",tol);
   
 //Adivina solucion: Asigna valores aleatorios entre 0 y 0.1..................
   double RANDMAX = double(RAND_MAX);
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     if(FLAG[i][k]==0){
      Psi[i][k] = 0.1*double(rand())/RANDMAX;
     }
     else{
      Psi[i][k] = Psib;
     }
    }
   }
 //...........................................................................
   
 //Definir valores propios del metodo y de la discretizacion..................
   double err_local;
   double err_max;
   double res = 1.0;
   itmax = GS_ITMAX;
   FILE *out1;
   out1 = fopen("Results/Error_vs_it_fixed_boundary.txt","w");

   double cte_1 = 2.*(pow(1./hr,2)+pow(1./hz,2));
   double cte_5 = 1./cte_1;
   double cte_2 = cte_5/pow(hr,2);
   double cte_3 = 0.5*cte_5/hr;
   double cte_4 = cte_5/pow(hz,2);
 //...........................................................................
   CRITICAL_POINTS();
   
 //INICIA BUCLE METODO DE RELAJACION..........................................
   
   for(int it=1;it<=itmax;it++){
    
    //Guardar solucion previa
      for(int k=0;k<npz;k++){
       for(int i=0;i<npr;i++){
        Psi_old[i][k] = Psi[i][k];
        if(FLAG[i][k]!=9){
         PsiN[i][k] = (Psib-Psi[i][k])/(Psib-Psia);
        }
        else{
         PsiN[i][k] = 0.0;
        }
       }
      }
    
    ALPHA();
    CURRENT_DENSITY();

    //Calcular nueva solucion excepto en frontera
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        if(FLAG[i][k]==0){
         Psi[i][k] = cte_2*(Psi[i+1][k]+Psi[i-1][k]) - \
                     cte_3*(Psi[i+1][k]-Psi[i-1][k])/r[i] + \
                     cte_4*(Psi[i][k+1]+Psi[i][k-1]) - \
                     cte_5*(-r[i]*Jt[i][k]);
        }
       }
      }
      
    //Proceso de interpolacion
      INTERPOLATION();
      
    //PICARD
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        Psi[i][k] = 0.3*Psi_old[i][k] + 0.7*Psi[i][k];
       }
      }
      
    //Calculo de alpha para garantizar valores de corriente en el plasma
      CRITICAL_POINTS();
      //ALPHA();
    
    //Calcular error local y verificar si este es el error maximo
      err_max = 0.0;//Para determinar maximos, se inicializan en cero
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        err_local = fabs(Psi[i][k] - Psi_old[i][k]);
        if(err_local>err_max){
         err_max = err_local;
        }
       }
      }

    //Residuo relativo de la ecuacion de Grad-Shafranov (criterio real)
      res = RESIDUAL();

    //Guardar iteracion, cambio por iteracion y residuo
      fprintf(out1,"%d %e %e\n",it,err_max,res);

    //Verificar condicion de tolerancia sobre el RESIDUO, no sobre el paso.
    //GS_TOL_RES <= 0 desactiva la prueba (usado para medir el plateau).
      if( (GS_TOL_RES > 0.0) && (res < GS_TOL_RES) ){
       printf("SOLUCION ENCONTRADA EN %d ITERACIONES\n",it);
       printf("  residuo relativo GS = %e  (tol = %e)\n",res,(double)GS_TOL_RES);
       printf("  cambio por iteracion = %e\n",err_max);
       break;
      }

   }
   fclose(out1);

   //Mensaje de NO convergencia
     if( (GS_TOL_RES > 0.0) && (res > GS_TOL_RES) ){
      printf("SOLUCION NO CONVERGE EN %d ITERACIONES\n",itmax);
      printf("  residuo relativo GS = %e  (tol = %e)\n",res,(double)GS_TOL_RES);
      printf("Se recomienda revisar el comportamiento del error\n");
     }

   printf("Eje magnetico: R = %e m,  Z = %e m,  Psi_a = %e Wb/rad\n",
          raxis*Lo, zaxis*Lo, Psia*Psio);
   //.........................................................................

   //GUARDAR POSICION DEL EJE MAGNETICO.......................................
     FILE *outax;
     outax = fopen("Results/axis_fixed_boundary.txt","w");
     fprintf(outax,"# raxis[m] zaxis[m] Psia[Wb/rad] shafranov_shift[m]\n");
     fprintf(outax,"%e %e %e %e\n",
             raxis*Lo, zaxis*Lo, Psia*Psio, (raxis-Ro)*Lo);
     fclose(outax);
   //.........................................................................

   //GUARDAR SOLUCION DE FLUJO POLOIDAL EN ARCHIVOS...........................

     FILE *out2;
     out2 = fopen("Results/Psi_rz_fixed_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       //fprintf(out2,"%e %e %e\n",r[i]*Lo,z[k]*Lo,Psi[i][k]*Psio);
       fprintf(out2,"%.15e ",Psi[i][k]*Psio);
      }
      fprintf(out2,"\n");
     }
     fclose(out2);
     
     out2 = fopen("Results/PsiN_rz_fixed_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       //fprintf(out2,"%e %e %e\n",r[i]*Lo,z[k]*Lo,Psi[i][k]*Psio);
       fprintf(out2,"%.15e ",PsiN[i][k]);
      }
      fprintf(out2,"\n");
     }
     fclose(out2);
   //.........................................................................
     out2 = fopen("Results/Jt.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out2,"%.15e ",Jt[i][k]*Jo);
      }
      fprintf(out2,"\n");
     }
     fclose(out2);
   //.........................................................................
   
   printf("Successive over-relaxation method finished\n");
  }
/*============================================================================*/
  
  
/*============================================================================
  SUCCESSIVE OVER-RELAXATION METHOD
  ============================================================================*/
  void SOR_SEMIFREE_BOUNDARY(void){
   
   tol = 1.e-3*(pow(hr,2)+pow(hz,2));
   printf("TOL = %e\n",tol);
   
 //Adivina solucion: Asigna valores aleatorios entre 0 y 0.1..................
   double RANDMAX = double(RAND_MAX);
   for(int k=1;k<npz-1;k++){
    for(int i=1;i<npr-1;i++){
     if(FLAG[i][k]==0){
      Psi[i][k] = 0.1*double(rand())/RANDMAX;
     }
     else{
      Psi[i][k] = Psib;
     }
    }
   }
   
 //Psib ya viene de la configuracion y fue normalizado en NORMALIZATION().
 //Antes se sobrescribia aqui con 0.0, lo que hacia que el solver semi-libre
 //ignorara silenciosamente un Psi_b distinto de cero que el solver de frontera
 //fija si respetaba.
 //...........................................................................

 //Definir valores propios del metodo y de la discretizacion..................
   double err_local;
   double err_max;
   double res = 1.0;
   itmax = GS_ITMAX;
   FILE *out1;
   out1 = fopen("Results/Error_vs_it_free_boundary.txt","w");

   double cte_1 = 2.*(pow(1./hr,2)+pow(1./hz,2));
   double cte_5 = 1./cte_1;
   double cte_2 = cte_5/pow(hr,2);
   double cte_3 = 0.5*cte_5/hr;
   double cte_4 = cte_5/pow(hz,2);
 //---------------------------------------------------------------------------
   CRITICAL_POINTS();
 
 //INICIA BUCLE METODO DE RELAJACION..........................................
   
   for(int it=1;it<=itmax;it++){
    
    //Guardar solucion previa
      for(int k=0;k<npz;k++){
       for(int i=0;i<npr;i++){
        Psi_old[i][k] = Psi[i][k];
        if(FLAG[i][k]!=9){
         PsiN[i][k] = (Psib-Psi[i][k])/(Psib-Psia);
        }
        else{
         PsiN[i][k] = 0.0;
        }
       }
      }
      
      ALPHA();
      CURRENT_DENSITY();
      if(it % 100 == 0){
       CONDICIONES_DE_FRONTERA();
      }

    //Calcular nueva solucion excepto en frontera
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        if((FLAG[i][k]==0)||(FLAG[i][k]==9)){
         Psi[i][k] = cte_2*(Psi[i+1][k]+Psi[i-1][k]) - \
                     cte_3*(Psi[i+1][k]-Psi[i-1][k])/r[i] + \
                     cte_4*(Psi[i][k+1]+Psi[i][k-1]) - \
                     cte_5*(-r[i]*Jt[i][k]);
        }
       }
      }

    //Proceso de interpolacion (despues del barrido SOR, igual que en
    //SOR_FIXED_BOUNDARY; antes se ejecutaba antes del barrido)
      INTERPOLATION();

    //PICARD
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        Psi[i][k] = 0.3*Psi_old[i][k] + 0.7*Psi[i][k];
       }
      }
      
      CRITICAL_POINTS();
      
    
    //Calcular error local y verificar si este es el error maximo
      err_max = 0.0;//Para determinar maximos, se inicializan en cero
      for(int k=1;k<npz-1;k++){
       for(int i=1;i<npr-1;i++){
        err_local = fabs(Psi[i][k] - Psi_old[i][k]);
        if(err_local>err_max){
         err_max = err_local;
        }
       }
      }

    //Residuo relativo de la ecuacion de Grad-Shafranov (criterio real)
      res = RESIDUAL();

    //Guardar iteracion, cambio por iteracion y residuo
      fprintf(out1,"%d %e %e\n",it,err_max,res);

    //Verificar condicion de tolerancia sobre el RESIDUO, no sobre el paso
      if( (GS_TOL_RES > 0.0) && (res < GS_TOL_RES) ){
       printf("SOLUCION ENCONTRADA EN %d ITERACIONES\n",it);
       printf("  residuo relativo GS = %e  (tol = %e)\n",res,(double)GS_TOL_RES);
       printf("  cambio por iteracion = %e\n",err_max);
       break;
      }

   }
   fclose(out1);

   //Mensaje de NO convergencia
     if( (GS_TOL_RES > 0.0) && (res > GS_TOL_RES) ){
      printf("SOLUCION NO CONVERGE EN %d ITERACIONES\n",itmax);
      printf("  residuo relativo GS = %e  (tol = %e)\n",res,(double)GS_TOL_RES);
      printf("Se recomienda revisar el comportamiento del error\n");
     }

   printf("Eje magnetico: R = %e m,  Z = %e m,  Psi_a = %e Wb/rad\n",
          raxis*Lo, zaxis*Lo, Psia*Psio);
   //.........................................................................

   //GUARDAR POSICION DEL EJE MAGNETICO.......................................
     FILE *outax;
     outax = fopen("Results/axis_free_boundary.txt","w");
     fprintf(outax,"# raxis[m] zaxis[m] Psia[Wb/rad] shafranov_shift[m]\n");
     fprintf(outax,"%e %e %e %e\n",
             raxis*Lo, zaxis*Lo, Psia*Psio, (raxis-Ro)*Lo);
     fclose(outax);
   //.........................................................................

   //GUARDAR DATOS EN ARCHIVOS................................................
     out1 = fopen("Results/Psi_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",Psi[i][k]*Psio);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/PsiN_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",PsiN[i][k]);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/Pressure_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",p(PsiN[i][k])*Po);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/Jt_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",Jt[i][k]*Jo);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/Br_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",-Bo*dz(Psi,i,k)/r[i]);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/Bz_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",Bo*dr(Psi,i,k)/r[i]);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
     out1 = fopen("Results/Bt_rz_free_boundary.txt","w");
     for(int k=0;k<npz;k++){
      for(int i=0;i<npr;i++){
       fprintf(out1,"%.15e ",Bo*go*sqrt(1.+alpha*g(PsiN[i][k]))/r[i]);
      }
      fprintf(out1,"\n");
     }
     fclose(out1);
   //.........................................................................
   
   printf("Successive over-relaxation method finished\n");
  }
/*============================================================================*/
  
  
/*============================================================================
  ASIGNA VALORES DE PSI EN LA FRONTERA COMPUTACIONAL CON FUN. DE GREEN
  ============================================================================*/
  void CONDICIONES_DE_FRONTERA(void){
   
   for(int i=0;i<npr;i++){
    
    Psi[i][0] = PSI_BOUNDARY(r[i],z[0]);
    Psi[i][npz-1] = PSI_BOUNDARY(r[i],z[npz-1]);
   }
   
   for(int k=0;k<npz;k++){
    Psi[0][k] = PSI_BOUNDARY(r[0],z[k]);
    Psi[npr-1][k] = PSI_BOUNDARY(r[npr-1],z[k]);
   }
   
  }
/*============================================================================*/
  
  
/*============================================================================
  RESULEVE LAS INTEGRALES DE GREEN PARA SAINAR PSI EN FRONTERA COMPUTACIONAL
  ============================================================================*/
  double PSI_BOUNDARY(double rp, double zp){
   
   double sum = 0.0;
   
   #pragma omp parallel for collapse(2) reduction(+:sum)
   for(int i=1;i<npr-1;i++){
    for(int k=1;k<npz-1;k++){
     sum = sum + F_GREEN(r[i],z[k],rp,zp)*Jt[i][k];
    }
   }
   
   double Integral = sum*hr*hz;
   
   sum = 0.0;
   for(int l=0;l<Nc;l++){
    sum = sum + F_GREEN(rc[l],zc[l],rp,zp)*Ic[l];
   }
   
   return Integral + sum;
  }
/*============================================================================*/
  
  
/*============================================================================
  FUNCION DE GREEN ASOCIADA AL OPERADOR ELIPTICO TOROIDAL
  ============================================================================*/
  double F_GREEN(double rnp,double znp,double rp,double zp){
   
   double kappa = sqrt( 4.*rp*rnp/( pow(rnp+rp,2)+pow(znp-zp,2) ) );
   
 //CALCULAR INTEGRALES ELIPTICAS----------------------------------------------
   double a = 1.;
   double b = sqrt(1. - kappa*kappa);
   double c = sqrt(fabs(a*a - b*b));
   
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
   
   double K = 0.5*M_PI/a;
   double E = K*(1.-sum);
   //-------------------------------------------------------------------------
   
   return sqrt(rp*rnp)*( (2.-pow(kappa,2))*K - 2.*E )/(2.*M_PI*kappa);
  }
/*============================================================================*/
  

/*============================================================================
  SOURCE TERM IN GRAD-SHAFRANOV EQUATION
  ============================================================================*/
  void CURRENT_DENSITY(void){

   double dpsiN_dpsi = -1./(Psib-Psia);

 //FLAG != 9 (not FLAG == 0): FLAG 1-8 mark points that are INSIDE the plasma,
 //adjacent to the boundary. Restricting to FLAG == 0 left Jt identically zero
 //on that ring, which (a) killed the C5*Jt source term in INTERPOLATION(),
 //(b) made ALPHA()'s Ip integral -- taken over FLAG != 9 -- inconsistent with
 //the current actually deposited, and (c) omitted the ring current from
 //PSI_BOUNDARY() and from the exported Jt.txt.
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     if(FLAG[i][k]!=9){
      Jt[i][k] = r[i]*Pa*dp_dpsiN(PsiN[i][k])*dpsiN_dpsi + \
                 0.5*pow(go,2)*alpha*dg_dpsiN(PsiN[i][k])*dpsiN_dpsi/r[i];
     }
    }
   }

  }
/*============================================================================*/
  
  
/*============================================================================
  ALPHA PARAMETER
  ============================================================================*/
  void ALPHA(){
   
   double sum1,sum2;
   double psiN;
   
   sum1=0.0;
   sum2=0.0;
   
   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     if(FLAG[i][k]!=9){
      sum1 = sum1 + r[i]*dp_dpsiN(PsiN[i][k]);
      sum2 = sum2 + dg_dpsiN(PsiN[i][k])/r[i];
     }
    }
   }
   
   double dpsiN_dpsi = -1./(Psib-Psia);
   
   alpha = (Ip - Pa*dpsiN_dpsi*sum1*hr*hz)/(0.5*go*go*dpsiN_dpsi*sum2*hr*hz);
   
  }
/*============================================================================*/
  
  
/*============================================================================
  GRAD-SHAFRANOV RESIDUAL

  Returns the RELATIVE residual of the discretized GS equation,

      max| Delta* Psi + r*Jt |  /  max| r*Jt |

  evaluated over the plasma interior (FLAG == 0, where the standard 5-point
  stencil applies; the FLAG 1-8 ring uses the irregular stencil in
  INTERPOLATION() and is excluded).

  This is the quantity that actually answers "has the equation been solved?".
  It is used as the stopping criterion; the per-iteration change is still
  logged alongside it for comparison.
  ============================================================================*/
  double RESIDUAL(void){

   double res_max = 0.0;
   double src_max = 0.0;

   for(int k=1;k<npz-1;k++){
    for(int i=1;i<npr-1;i++){
     if(FLAG[i][k]==0){

      double d2r = (Psi[i+1][k] - 2.*Psi[i][k] + Psi[i-1][k])/pow(hr,2);
      double d1r = (Psi[i+1][k] - Psi[i-1][k])/(2.*hr);
      double d2z = (Psi[i][k+1] - 2.*Psi[i][k] + Psi[i][k-1])/pow(hz,2);

      double lap = d2r - d1r/r[i] + d2z;   //Delta* Psi
      double src = r[i]*Jt[i][k];          //  r*Jt   (Delta* Psi = -r*Jt)

      double res = fabs(lap + src);

      if(res>res_max) res_max = res;
      if(fabs(src)>src_max) src_max = fabs(src);

     }
    }
   }

   if(src_max<=0.0) return res_max;   //degenerate: no source yet

   return res_max/src_max;
  }
/*============================================================================*/


/*============================================================================
  CRITICAL POINTS
  ============================================================================*/
  void CRITICAL_POINTS(void){

 //Full 2D scan over the plasma for the magnetic axis (max of Psi).
 //
 //The previous version scanned only the row k = npz/2. That is correct only if
 //(i) the equilibrium is up-down symmetric AND (ii) the z-mesh is symmetric
 //about zero, so that z[npz/2] is actually the midplane. Neither holds in
 //general: an asymmetric z-range makes z[npz/2] != 0 (so an arbitrary row gets
 //scanned), and an up-down asymmetric equilibrium (single-null X-point,
 //asymmetric coils) moves the axis off the midplane entirely. A wrong Psia
 //propagates into PsiN, alpha and Jt.
 //
 //raxis/zaxis are seeded unconditionally so they can never be left at their
 //zero-initialized value if the seed point already holds the maximum.

   int ia = npr/2;
   int ka = npz/2;

   Psia  = Psi[ia][ka];
   raxis = r[ia];
   zaxis = z[ka];

   for(int k=0;k<npz;k++){
    for(int i=0;i<npr;i++){
     if(FLAG[i][k]!=9){
      if(Psia<Psi[i][k]){
       Psia  = Psi[i][k];
       raxis = r[i];
       zaxis = z[k];
      }
     }
    }
   }

  }
/*============================================================================*/
  
  
/*============================================================================
  INTERPOLATION
  ============================================================================*/  
  void INTERPOLATION(void){

   double Fa,Fb,Fc,Fd;

 //Generic formula (added 2026-09-10), replacing the 8-case switch(FLAG) that
 //used to sit here. Decides, per direction, whether the neighbor is across
 //the boundary (use Psib) purely from the already-computed distances
 //La/Lb/Lc/Ld -- never from FLAG -- so it works unmodified for both the
 //analytic D-shape ring (FLAG 1..8, POINTSFORINTERPOLATION()) and the
 //polygonal ring (FLAG==1 only, POINTSFORINTERPOLATION_POLY()).
 //
 //Safe for the existing analytic path by construction: in
 //POINTSFORINTERPOLATION(), any direction NOT adjacent to the boundary has
 //its distance fixed at EXACTLY hr or hz (e.g. Lb[npi]=hr in case 1), while
 //any direction that IS adjacent has a real geometric crossing distance that
 //is always strictly less than hr/hz. eps_geom exists only to avoid a
 //floating-point tie at that exact boundary between "fixed at h" and "real
 //crossing", never to change which branch is taken in practice. Verified by
 //the regression test in docs/reports/XPOINT_CONSISTENT_JT_IMPLEMENTATION.tex
 //(FIXED_GS_SOLVER vs FIXED_GS_SOLVER_POLY on the same D-shape, the latter
 //resampled as a polygon).
   double eps_geom = 1e-9;

   for(int t=0;t<npi;t++){

    Fa = (La[t] < hr-eps_geom) ? Psib : Psi[Ii[t]-1][Ki[t]];
    Fb = (Lb[t] < hr-eps_geom) ? Psib : Psi[Ii[t]+1][Ki[t]];
    Fc = (Lc[t] < hz-eps_geom) ? Psib : Psi[Ii[t]][Ki[t]+1];
    Fd = (Ld[t] < hz-eps_geom) ? Psib : Psi[Ii[t]][Ki[t]-1];

    //Psi[Ii[t]][Ki[t]] = 0.5*( (Fa*Lb[t] + Fb*La[t])/(La[t]+Lb[t]) + \
                              (Fc*Ld[t] + Fd*Lc[t])/(Lc[t]+Ld[t]) );
    
    double Dr = (pow(La[t],2)*Lb[t] + pow(Lb[t],2)*La[t]);
    double Dz = (pow(Lc[t],2)*Ld[t] + pow(Ld[t],2)*Lc[t]);
    
    double A1r = La[t]/(0.5*Dr);
    double A2r = Lb[t]/(0.5*Dr);
    double A3r = pow(La[t],2)/(r[Ii[t]]*Dr);
    double A4r = pow(Lb[t],2)/(r[Ii[t]]*Dr);
    
    double A1z = Ld[t]/(0.5*Dz);
    double A2z = Lc[t]/(0.5*Dz);
    
    double C = A1r + A2r - A3r + A4r + A1z + A2z;
    
    double C1 = (A1r - A3r)/C;
    double C2 = (A2r + A4r)/C;
    double C3 = A1z/C;
    double C4 = A2z/C;
    double C5 = r[Ii[t]]/C;
    
    Psi[Ii[t]][Ki[t]] = C1*Fb + C2*Fa + C3*Fc + C4*Fd + C5*Jt[Ii[t]][Ki[t]];
  
    
   }
   
  }
/*============================================================================*/
  
  
/*============================================================================
  PRESSURE FUNCTION
  ============================================================================*/
  double p(double psiN){
   
   return Pa*( pow(1.-pow(1-psiN,2),2) + 0.2 );
  }
/*============================================================================*/
  
  
/*============================================================================
  DERIVATE PRESSURE FUNCTION
  ============================================================================*/
  double dp_dpsiN(double psiN){
   
   return 4.*(1.-psiN)*(1.-pow(1.-psiN,2));
  }
/*============================================================================*/
  
  
/*============================================================================
  g FUNCTION
  ============================================================================*/
  double g(double psiN){
   
   return  pow(psiN,2);//-0.2*pow(psiN,4) + 1.5*pow(psiN,2);
  }
/*============================================================================*/
  
  
/*============================================================================
  DERIVATE g FUNCTION
  ============================================================================*/
  double dg_dpsiN(double psiN){
   
   return 2.*psiN;//-0.8*pow(psiN,3) + 3.0*pow(psiN,1);
  }
/*============================================================================*/
  
  
/*============================================================================
  SAVE D-SHAPE IN TXT FILE
  ============================================================================*/
  void SAVEDSHAPE(void){
   
   FILE *out;
   out = fopen("Results/Dshape.txt","w");
   
   double dt = 2.*M_PI/100.;
   for(double t=0.0;t<2.*M_PI;t=t+dt){
    fprintf(out,"%e %e\n",rboundary(t)*Lo,zboundary(t)*Lo);
   }

   fclose(out);
  }
/*============================================================================*/


/*============================================================================
  SAVE D-SHAPE IN TXT FILE -- POLYGONAL BOUNDARY VERSION

  Trivial: the polygonal boundary IS the input data, so it is written
  directly (denormalized by Lo) rather than resampled from an analytic curve.
  ============================================================================*/
  void SAVEDSHAPE_POLY(const std::vector<double>& Rp,const std::vector<double>& Zp){

   FILE *out;
   out = fopen("Results/Dshape.txt","w");

   for(size_t j=0;j<Rp.size();j++){
    fprintf(out,"%e %e\n",Rp[j]*Lo,Zp[j]*Lo);
   }

   fclose(out);
  }
/*============================================================================*/


/*============================================================================
  PLASMA BOUNDARY: D-SHAPE R-FUNCTION
  ============================================================================*/
  double rboundary(double theta){
   return Ro + a*cos( theta + asin(delta)*sin(theta) );
  }
/*============================================================================*/
  
  
/*============================================================================
  PLASMA BOUNDARY: D-SHAPE Z-FUNCTION
  ============================================================================*/
  double zboundary(double theta){
   return a*kappa*sin(theta);
  }
/*============================================================================*/
  
  
/*============================================================================
  PARAMETRIC THETA VALUE FROM ANALYTIC PLASMA BOUNDARY
  ============================================================================*/  
  double thetaboundary(double r,double z){
   
   double xa = 0.;
   double xb = 2.*M_PI;
   double xc = 0.5*(xa+xb);
   
   double cte1 = acos( (r - Ro)/a );
   double cte2 = asin(delta);
   
   double err = 1.;
   
   while(err>1.e-9){
    
    double Fa = xa + cte2*sin(xa) - cte1;
    double Fb = xb + cte2*sin(xb) - cte1;
    double Fc = xc + cte2*sin(xc) - cte1;
    
    if(Fa*Fb>0.){
     printf("NO ES POSIBLE DETECTAR RAIZ\n");
     exit(0);
    }
    else{
     if(Fa*Fc<0.) xb = xc;
     else xa = xc;
     err = fabs(xb-xa);
     xc = 0.5*(xa+xb);
    }
   }
   
   return xc;
  }
/*============================================================================*/
  
  
/*============================================================================
  DERIVADAS RESPECTO A r
  ============================================================================*/
  double dr(double **F,int i,int k){
   
   double dF;
   
   if(i==0){
    dF = (-1.5*F[i][k] + 2.*F[i+1][k] - 0.5*F[i+2][k])/hr; //Paso adelante
   }
   else if(i==npr-1){
    dF = ( 1.5*F[i][k] - 2.*F[i-1][k] + 0.5*F[i-2][k])/hr; //Paso atras
   }
   else{
    dF = 0.5*(F[i+1][k] - F[i-1][k])/hr; //Centradas
   }
   
   return dF;
  }
/*============================================================================*/
  
  
/*============================================================================
  DERIVADAS RESPECTO A z
  ============================================================================*/
  double dz(double **F,int i,int k){
   
   double dF;
   
   if(k==0){
    dF = (-1.5*F[i][k] + 2.*F[i][k+1] - 0.5*F[i][k+2])/hz; //Paso adelante
   }
   else if(k==npz-1){
    dF = ( 1.5*F[i][k] - 2.*F[i][k-1] + 0.5*F[i][k-2])/hz; //Paso atras
   }
   else{
    dF = 0.5*(F[i][k+1] - F[i][k-1])/hz; //Centradas
   }
   
   return dF;
  }
/*============================================================================*/
  void BOBINAS(void){

   FILE *input;
   input = fopen("Results/corrientes.txt","r");
   if(!input){
    printf("ERROR: Cannot open Results/corrientes.txt\n");
    exit(1);
   }

   // Count the number of coils in the file
   double tmp1, tmp2, tmp3;
   int Nc_file = 0;
   while(fscanf(input,"%lf %lf %lf",&tmp1,&tmp2,&tmp3) == 3){
    Nc_file++;
   }
   rewind(input);

   // Reallocate coil arrays with the correct size
   delete[] rc;
   delete[] zc;
   delete[] Ic;
   Nc = Nc_file;
   rc = new double[Nc];
   zc = new double[Nc];
   Ic = new double[Nc];

   printf("Coils read from file: Nc = %d\n", Nc);

   double val;

   for(int l=0;l<Nc;l++){

    fscanf(input,"%lf",&val);
    rc[l] = val/Lo;

    fscanf(input,"%lf",&val);
    zc[l] = val/Lo;

    fscanf(input,"%lf",&val);
    Ic[l] = val/Io;

   }

   fclose(input);
  }
/*============================================================================*/
