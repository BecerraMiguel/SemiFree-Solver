/*============================================================================
  FIXED GRAD-SHAFRANOV SOLVER
  ============================================================================
  A prototipe main program to run the solver where the Dshape, plasma and
  mesh parameters are the input data.
  
  Last modification: Feb 20, 2026
  ============================================================================*/
  
  
/*============================================================================
  LIBRARIES
  ============================================================================*/
  #include <stdio.h>
  #include <iostream>
  #include <fstream>
  #include <string>
  #include "json.hpp"
  #include "GS_solver.h"

  using namespace std;
  using json = nlohmann::json;
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

   FIXED_GS_SOLVER(Dshape, constrains, mesh);
   SEMIFREE_GS_SOLVER(Dshape, constrains, mesh);

   return 0;
  }
/*============================================================================
   END PROGRAM
  ============================================================================*/
