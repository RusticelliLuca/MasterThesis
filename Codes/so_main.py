
import pandas as pd
import numpy as np
from statsmodels.api import GLS, OLS 
from scipy.special import laguerre,hermite,legendre

from so_schwartz_estimation import SchwartzEstimation
from mc_main import pos_delivery_start, dict_out_sim_robust, dict_out_sim


estimation_istance = SchwartzEstimation(
                    pos_delivery_start = pos_delivery_start,
                    n_ex_options = 5,
                    ester = 0.015,
                    EUA_denominator = 88.0,
                    estimation_method = GLS,
                    dict_basis_f = {'POLY FAMILY': laguerre, 'APRX N': 5,'WEIGHT FACTOR': 0 } 
                    #laguerre, hermite
                    )
price_list=[]
rmse_list=[]
price_list_robust=[]
rmse_list_robust = []
data =[]
list_strikes   = [310,330,350,370,400] 
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
output = pd.DataFrame(data=data, 
                      index=['Swing Price','RMSE','Robust Swing Price','Robust RMSE'], 
                      columns=list_strikes)