import pandas as pd
import numpy as np
from statsmodels.api import OLS 
from scipy.special import laguerre
import matplotlib.pyplot as plt

from so_schwartz_estimation import SchwartzEstimation
from mc_main import pos_delivery_start, dict_out_sim_robust, dict_out_sim

""" CHANGE NUMBER OF EXERCISE OPTIONS """

#FOR LOOPS for the changing parameter

data_options=[]


for n in range(1,32):
    """ PARAMETERS"""
    n_ex_options = n
    EUA_denominators = 88
    estimation_method = OLS
    dict_basis_f = {'POLY FAMILY': laguerre, 'APRX N': 4,'WEIGHT FACTOR': 0 }

    #estimation istance
    estimation_istance = SchwartzEstimation(
                    pos_delivery_start = pos_delivery_start,
                    n_ex_options = n_ex_options,
                    ester = 0.015,
                    EUA_denominator = EUA_denominators,
                    estimation_method = estimation_method,
                    dict_basis_f = dict_basis_f #laguerre, hermite
                    )            
    """ FOR LOOPS""" 
    list_strikes = [310,330,350,370,400]
    
    price_list=[]
    rmse_list=[]
    price_list_robust=[]
    rmse_list_robust = []
    data =[]
    
    for strike in list_strikes :
        """ run estimation"""
        price_swing_today, coeff_i_n_k, rmse = estimation_istance.estimation(dict_out = dict_out_sim, strike=strike)
        price_list.append(price_swing_today)
        rmse_list.append(rmse)
        """ run a second simulation for constistency"""
        price_swing_today, coeff_i_n_k, rmse = estimation_istance.estimation(dict_out = dict_out_sim_robust, strike=strike, coeff= coeff_i_n_k)
        price_list_robust.append(price_swing_today)
        rmse_list_robust.append(rmse)
    
    """ SHOW OUTPUT DATA IN A SUMMARIZED WAY"""
    data = price_list, rmse_list, price_list_robust, rmse_list_robust
    data_options.append(pd.DataFrame(data=data, index=['Swing Price','RMSE','Robust Swing Price','Robust RMSE'], columns=list_strikes))



""" CREATION OF 3D PLOTS"""

# df modification
final_ex = pd.concat([df for df in data_options])
final_ex.to_excel('final_ex_opt.xlsx')
x = final_ex[final_ex.index == 'Robust Swing Price'].reset_index()
x.drop('index', axis=1, inplace=True)
#creation of an 3 columns array
a=[]
for idx in x.index:
    for col in x.columns:
        a.append([(idx),col,x.loc[idx,col]])
d=np.array(a).T

""" plot creation"""

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
X= d[1]
Y = d[0]
Z=d[2]
# Plot the surface.
surf = ax.plot_trisurf(X, Y, Z,cmap='viridis')#cmap=plt.cm.CMRmap)
ax.set_xlabel('Strike')
ax.set_ylabel('Number of options')
ax.set_zlabel('Swing Price')
fig.savefig('3Dplot_ex_opt.png')