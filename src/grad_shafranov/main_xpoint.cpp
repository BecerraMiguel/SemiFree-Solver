/*============================================================================
  FIXED GRAD-SHAFRANOV SOLVER -- POLYGONAL (X-POINT) BOUNDARY DRIVER
  ============================================================================
  Parallel to main.cpp: reads a configuration file the same way, but calls
  FIXED_GS_SOLVER_POLY() with a boundary given as an arbitrary polygon (with
  or without X-points) instead of the analytic D-shape. See
  docs/FEATURE_XPOINT_CONSISTENT_JT_BRIEFING.pdf for the full design.

  Added 2026-09-10.
  ============================================================================*/


/*============================================================================
  LIBRARIES
  ============================================================================*/
  #include <stdio.h>
  #include <iostream>
  #include <fstream>
  #include <sstream>
  #include <string>
  #include <vector>
  #include <utility>
  #include "json.hpp"
  #include "GS_solver.h"

  using namespace std;
  using json = nlohmann::json;
/*============================================================================*/


/*============================================================================
  LOAD A POLYGON (R Z per line, in meters) FROM A TEXT FILE
  ============================================================================*/
  vector<pair<double,double>> loadPolygon(const string& path){

   vector<pair<double,double>> poly;
   ifstream f(path);
   if(!f.is_open()){
    cerr << "Cannot open boundary polygon file: " << path << endl;
    exit(1);
   }

   string line;
   while(getline(f,line)){
    if(line.find_first_not_of(" \t\r\n") == string::npos) continue; //blank line
    istringstream iss(line);
    double R,Z;
    if(iss >> R >> Z){
     poly.push_back(make_pair(R,Z));
    }
   }

   if(poly.size() < 3){
    cerr << "Boundary polygon file " << path << " has fewer than 3 points ("
         << poly.size() << ")." << endl;
    exit(1);
   }

   return poly;
  }
/*============================================================================*/


/*============================================================================
  MAIN PROGRAM
  ============================================================================*/
  int main(){

   string config_path;
   cout << "Enter path to configuration file: ";
   cin >> config_path;

   ifstream f(config_path);
   if (!f.is_open()) {
       cerr << "Cannot open config file: " << config_path << endl;
       return 1;
   }
   json cfg = json::parse(f);
   cout << "Loaded configuration: " << cfg["name"] << endl;
   cout << cfg["description"] << endl;

   /****************************************************************
     D-SHAPE PARAMETERS: |   Ro   |   a   |    kappa    |   delta
     Only Ro is used by FIXED_GS_SOLVER_POLY (normalization reference,
     Lo=Ro); a/kappa/delta are read for signature homogeneity with main.cpp
     but ignored downstream.
    ****************************************************************/
   double Dshape[4] = {
       cfg["geometry"]["Ro"],
       cfg["geometry"]["a"],
       cfg["geometry"]["kappa"],
       cfg["geometry"]["delta"]
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
     XPOINT BLOCK: path to the already-transformed boundary polygon
     (R Z per line, meters). "targets"/"transform_method" are only
     relevant to the (separate) xpoint_transform.py step and are not read
     here; "boundary_file" is the only field FIXED_GS_SOLVER_POLY needs.
    ****************************************************************/
   if(!cfg.contains("xpoint") || !cfg["xpoint"].contains("boundary_file")){
    cerr << "Config file must contain an \"xpoint\":{\"boundary_file\":...} block "
            "pointing to the already-transformed boundary polygon." << endl;
    return 1;
   }
   string poly_path = cfg["xpoint"]["boundary_file"];
   vector<pair<double,double>> boundary_poly = loadPolygon(poly_path);
   cout << "Loaded boundary polygon: " << poly_path
        << " (" << boundary_poly.size() << " points)" << endl;

   FIXED_GS_SOLVER_POLY(boundary_poly, Dshape, constrains, mesh);

   return 0;
  }
/*============================================================================
   END PROGRAM
  ============================================================================*/
