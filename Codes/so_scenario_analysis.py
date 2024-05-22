import pandas as pd
import numpy as np
from statsmodels.api import OLS 
from scipy.special import laguerre
import matplotlib.pyplot as plt

from so_schwartz_estimation import SchwartzEstimation
from mc_main import pos_delivery_start, dict_out_sim_robust, dict_out_sim
from mc_simulation import McSimulation
from mc_main import dict_EUA, dict_OU, list_seasonality, \
                    dict_dates, dict_copulas,n_sim, dict_copulas_robust

dict_mu = {'SSP5': 0.001696096032,
           'SSP3': 0.001915704702,
           'SSP2': 0.002191407194,
           'SSP1': 0.002283217033,
           'WAM': 0.002461671976,
          }


#preparing for the loop
data_options =[]
EUA_data_list=[]
n=0 
for mu in list(dict_mu.values()):
    dict_EUA['MU'] = mu
    Simulation2 = McSimulation(
                dict_OU = dict_OU,
                dict_EUA =dict_EUA,
                list_seasonality = list_seasonality,
                dict_dates = dict_dates,
                seed = 1, #random seed number
                jump= False #False -> no jump, True -> yes jumps
                )
    #firt and second MC simulation of PUN and EUA prices
    dict_out_sim =Simulation2.starter(num_mc_sim=n_sim, d_copulas=dict_copulas)
    dict_out_sim_robust =Simulation2.starter(num_mc_sim=n_sim, d_copulas=dict_copulas_robust)
    #save the simulation mean value At_m for each iteration:
    EUA_data_list.append( dict_out_sim_robust['EUA_m'] )
    
    """ START THE LOOPS FOR METHOD AND STRIKE"""
    """ PARAMETERS"""
    list_strikes =  [310,330,350,370,400]

    #estimation istance
    estimation_istance = SchwartzEstimation(
                    pos_delivery_start = pos_delivery_start,
                    n_ex_options = 5,
                    ester = 0.015,
                    EUA_denominator = 88,
                    estimation_method = OLS,
                    dict_basis_f = {'POLY FAMILY': laguerre, 'APRX N': 4,'WEIGHT FACTOR': 0 } 
                    #laguerre, hermite
                    )
    """ FOR LOOPS"""              
    #FOR LOOPS for the changing parameter
    price_list=[]
    rmse_list=[]
    price_list_robust=[]
    rmse_list_robust = []
    data =[]
    for strike in list_strikes:
        """ run estimation"""
        price_swing_today, coeff_i_n_k, rmse = estimation_istance.estimation(
                                                dict_out = dict_out_sim, 
                                                strike=strike
                                                )
        price_list.append(price_swing_today)
        rmse_list.append(rmse)
        """ run a second simulation for constistency"""
        price_swing_today, coeff_i_n_k, rmse = estimation_istance.estimation(
                                                dict_out = dict_out_sim_robust, 
                                                strike=strike, 
                                                coeff= coeff_i_n_k
                                                )
        price_list_robust.append(price_swing_today)
        rmse_list_robust.append(rmse)

    """ SHOW OUTPUT DATA IN A SUMMARIZED WAY"""
    data = price_list, rmse_list, price_list_robust, rmse_list_robust
    data_options.append(pd.DataFrame(data=data, 
                                     index=['Swing Price','RMSE','Robust Swing Price','Robust RMSE'], 
                                     columns=list_strikes
                                     ))
    n=n+1

EUA_data = pd.DataFrame(np.array(EUA_data_list).T.tolist(), columns=dict_mu.keys())
EUA_data.to_excel('EUA_mu_swingprice.xlsx')