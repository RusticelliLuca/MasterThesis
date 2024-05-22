
import pandas as pd
import numpy as np


from copula_function import Copula
from mc_simulation import McSimulation
from mc_introduction import *


""" PARAMETERS FOR THE MONTE CARLO SIMULATION"""

""" REFERENCE DATES"""
start_date = '2022-10-01'
start_delivery_date = '2024-10-01'
end_delivery_date = '2024-10-31'

""" SEASONALITY PARAMETERS"""
dict_coef_seas = {
    's1':-0.10212972841531148, 
    's2':0.08690318718087663, 
    's3':-0.0005061996717078659, 
    's4':0.03341946823435296,
    's5':5.4302485221994633, 
    'wd':0.1468520171034231   
}

"""ENERGY STOCHASTIC MODEL coeff"""
dict_OU = {
'INIT_PRICE': 250,
'ALPHA' : 0.116931791, 
'SIGMA' :0.099123982,
'LAMBDA' : -0.010278569021984, #Market risk premium
#JUMP
'SIGMA_J' : 0.133910073263212,
'MU_J' : -0.006593260104758,
'THETA' : 0.176441978434355 ,#parameter of the poisson distr.
'MRP_J' : -0.002364793030824
}
"""EUAs STOCHASTIC MODEL PARAMETERS"""
dict_EUA ={
'INIT_PRICE': 88,
'SIGMA' : 0.030653342, 
'MRP' : 0.0601216,
'MU' : 0.002191407
}

""" DATA UTILIZED TO CALIBRATE COPULA FUNCTIONS and WORKING DAYS (wdays calendar from 2022-01-01 TO 2025-01-01)"""
eua3 = pd.read_excel(r"C:\Users\ASUS\Desktop\TESI\Dati\eua_spot.xlsx", sheet_name='phase 3',index_col='Date')
eua4 = pd.read_excel(r"C:\Users\ASUS\Desktop\TESI\Dati\eua_spot.xlsx", sheet_name='phase 4',index_col='Date')
pun = pd.read_excel(r'C:\Users\ASUS\Desktop\TESI\Dati\energy_spot.xlsx', sheet_name='2016-2022', index_col = 0)
df_wdays = pd.read_excel(r"C:\Users\ASUS\Desktop\TESI\Dati\working days.xlsx", sheet_name="simulation", index_col=0)



#workload
eua4.rename(columns={'Price':'EUA','LogP':'LogP_EUA'}, inplace=True)
pun.rename(columns={'LogP':'LogP_PUN'}, inplace=True)
corr = pun.join(eua4, how='inner')
pun = np.array(corr['PUN'])
eua = np.array(corr['EUA'])
copula_function = Copula(pun, eua, family='clayton')


""" CALCULATIONS """


"""plot options:
 - None --> no plot visualization
 - 'ALL' --> plot output data from start_date
 - 'DELIVERY' --> plot output data from start_delivery_date
 - 'BOTH' --> plot output data for both previous options
 """
plot = 'BOTH' 
n_sim = 10000

#computations:
dict_dates, list_seasonality, pos_delivery_start = intro_calc(
    df_wdays = df_wdays,
    start_date = start_date, start_delivery_date = start_delivery_date, end_delivery_date = end_delivery_date,
    dict_coef_seas = dict_coef_seas)

dict_copulas =  copula(n_sim, dict_dates, copula_function)   #generate randomess

dict_out_sim =McSimulation(
                dict_OU = dict_OU,
                dict_EUA =dict_EUA,
                list_seasonality = list_seasonality,
                dict_dates = dict_dates,
                seed = 1, #random seed number
                jump= False #False -> no jump, True -> yes jumps
                ).starter(num_mc_sim=n_sim, d_copulas=dict_copulas, plot=plot)


""" ITERATE ESTIMATION FOR DIFFERENT INPUT DATA FIXED BELOW"""
plot = None 
n_sim = 10000

""" SIMULATION 1 & 2 DONE OUTSIDE SINCE WE WANT THEM CONSTANT IN ORDER TO COMPARE REUSLTS"""
#they remain ouside loops until we change a parameter, as MU
dict_copulas =  copula(n_sim, dict_dates, copula_function)
dict_copulas_robust =  copula(n_sim, dict_dates, copula_function)    
#simulation istance to be change based on copulas
Simulation = McSimulation(
                dict_OU = dict_OU,
                dict_EUA =dict_EUA,
                list_seasonality = list_seasonality,
                dict_dates = dict_dates,
                seed = 1, #random seed number
                jump= False #False -> no jump, True -> yes jumps
                )
#firt and second MC simulation of PUN and EUA prices
dict_out_sim =Simulation.starter(num_mc_sim=n_sim, d_copulas=dict_copulas, plot='BOTH')
dict_out_sim_robust =Simulation.starter(num_mc_sim=n_sim, d_copulas=dict_copulas_robust)