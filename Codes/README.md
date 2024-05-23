
The files in this folder contain the final part of my master thesis. 
The order to read them is the following:

  1. copula_function : class (slightly modified) taken from the following repo in order to simulate copula random numbers   
     https://github.com/rkawsar/ambhas/blob/master/
  2. mc_introduction : function to transform and pre-process data for the Monte Carlo simulation
  3. mc_simulation : Monte Carlo simulation
  4. mc_main : initialization and execution of the last three modules
  5. so_schwartz_estimation : algorithm to estimate the price of a Swing option
  6. so_main : initialization and execution of the previous module
  7. so_sensitivity_ex : example of a sensitivity done on the previous algorithm, in this case the swing contract price is 
     estimated for a different set of possible options to be exercized along the contract period
  8. so_scenario_analyisis : re-work and re-performance of the MC simulation and subsequent swing contract pricing for 
     different scenarios of the European Cabron Emission Allowances prices.
