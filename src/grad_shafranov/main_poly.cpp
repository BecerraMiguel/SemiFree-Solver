/*============================================================================
  FIXED GRAD-SHAFRANOV SOLVER -- POLYGONAL BOUNDARY DRIVER
  ============================================================================
  Companion driver to main.cpp: calls FIXED_GS_SOLVER_POLY() instead of
  FIXED_GS_SOLVER()+SEMIFREE_GS_SOLVER(), so that Jt can be regenerated
  consistently for an X-point-transformed boundary (produced externally by
  src/semifree_boundary/xpoint_transform_c1.py) instead of the analytic
  D-shape formula. See GS_solver.h's own comment above FIXED_GS_SOLVER_POLY()
  and docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf.

  Only Dshape[0]=Ro is used from the config's geometry block (a/kappa/delta
  are meaningless for an arbitrary polygon and are not read here); Results/
  Dshape.txt and Results/Jt.txt are written the same way as main.cpp's
  FIXED_GS_SOLVER() path (BUILDMESH_POLY() -> SAVEDSHAPE_POLY(), then
  SOR_FIXED_BOUNDARY() -> the same Jt.txt export).
  ============================================================================*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include "json.hpp"
#include "GS_solver.h"

using namespace std;
using json = nlohmann::json;

int main(){

  string config_path, poly_path;
  cout << "Enter path to configuration file: ";
  cin >> config_path;
  cout << "Enter path to boundary polygon file (R Z per line, meters): ";
  cin >> poly_path;

  ifstream f(config_path);
  if (!f.is_open()) {
      cerr << "Cannot open config file: " << config_path << endl;
      return 1;
  }
  json cfg = json::parse(f);
  cout << "Loaded configuration: " << cfg["name"] << endl;
  cout << cfg["description"] << endl;

  /****************************************************************
    D-SHAPE PARAMETERS: only Dshape[0]=Ro is used (Lo=Ro normalization
    reference); a/kappa/delta are ignored by FIXED_GS_SOLVER_POLY.
   ****************************************************************/
  double Dshape[4] = {
      cfg["geometry"]["Ro"],
      0.0,
      0.0,
      0.0
  };

  /****************************************************************
    CONSTRAINS:      |  I_plasma  | P_axis | B_axis | Psi_b
   ****************************************************************/
  double constrains[4] = {
      cfg["constraints"]["I_plasma"],
      cfg["constraints"]["P_axis"],
      cfg["constraints"]["B_axis"],
      cfg["constraints"]["Psi_b"]
  };

  /****************************************************************
    MESH PARAMETERS: | r_min | r_max | z_min | z_max | npr | npz
   ****************************************************************/
  double mesh[6] = {
      cfg["mesh"]["r_min"],
      cfg["mesh"]["r_max"],
      cfg["mesh"]["z_min"],
      cfg["mesh"]["z_max"],
      cfg["mesh"]["npr"],
      cfg["mesh"]["npz"]
  };

  /****************************************************************
    BOUNDARY POLYGON (R Z per line, meters -- e.g. the output of
    xpoint_transform_c1.py)
   ****************************************************************/
  ifstream pf(poly_path);
  if (!pf.is_open()) {
      cerr << "Cannot open boundary polygon file: " << poly_path << endl;
      return 1;
  }
  vector<pair<double,double>> boundary_poly_m;
  double r, z;
  while (pf >> r >> z) {
      boundary_poly_m.push_back(make_pair(r, z));
  }
  cout << "Loaded boundary polygon with " << boundary_poly_m.size()
       << " points from " << poly_path << endl;

  FIXED_GS_SOLVER_POLY(boundary_poly_m, Dshape, constrains, mesh);

  return 0;
}
/*============================================================================
   END PROGRAM
  ============================================================================*/
