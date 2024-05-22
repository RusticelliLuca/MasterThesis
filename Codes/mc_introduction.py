import random 
from datetime import datetime
import pandas as pd
import numpy as np
import time



""" DICTIONARY FOR DATES + PREPOCESSOR"""
def dict_dates_gen(df_wdays, start_date, start_delivery_date, end_delivery_date):
    dict_dates={
        'start_date' : start_date, 
        'start_delivery_date' :start_delivery_date, 
        'end_delivery_date' : end_delivery_date,
        'start_seasonality_date': str(df_wdays.index[0].date())
        }
    # transform in datetime obj
    for k in dict_dates.keys():
        dict_dates[k] = datetime.strptime(dict_dates[k], '%Y-%m-%d')
    return dict_dates


""" SEASONALITY LIST CONSTRUCTION"""
def seasonality(df_wdays,dict_coef_seas, n_days_mc, n_days_before_start): #df_wdays, dict_coef_seas
    """ compute the seasonal term for a sigle date defined in term of daily distance from start_seasonality_date"""
    df_wdays = df_wdays.copy()

    """ year fraction and working_days term calculation""" 
    def gen_coeff(t, df_wdays = df_wdays):
        y_frac = (t)/365.25
        """coeff construction"""
        par_seas =[
                np.sin(2*np.pi*y_frac), 
                np.cos(2*np.pi*y_frac), 
                np.sin(4*np.pi*y_frac), 
                np.cos(4*np.pi*y_frac),
                1, 
                float(df_wdays.iloc[t])
                ]
        return par_seas
    
    range_array = np.arange(n_days_before_start , n_days_before_start + n_days_mc + 1)
    l_coeff =[ gen_coeff(t) for t in range_array ]
    """compute seasonal value"""
    l2=list(dict_coef_seas.values())
    l_values = [np.inner(l1,l2) for l1 in l_coeff]
    return l_values

"""" CALCULCATION OF INTRO FOR THE SIMULATION"""
def intro_calc(
        df_wdays,
        start_date, start_delivery_date, end_delivery_date,
        dict_coef_seas
        ):
    #CALL THE DICT_DATES GENERATOR FUNCTION
    dict_dates = dict_dates_gen(df_wdays, start_date, start_delivery_date, end_delivery_date)
    #DETERMINE NUMBER OF SIMULATION STEPS
    n_days_mc = int((dict_dates['end_delivery_date'] - dict_dates['start_date']).days)
    n_days_before_start = int((dict_dates['start_date'] - dict_dates['start_seasonality_date']).days)
    n_days_delivery = int((dict_dates['end_delivery_date'] - dict_dates['start_delivery_date']).days)
    pos_delivery_start = n_days_mc - n_days_delivery
    #CALL THE SEASONALITY FUNCTION
    list_seasonality = seasonality(df_wdays,dict_coef_seas, n_days_mc, n_days_before_start)

    return dict_dates, list_seasonality, pos_delivery_start

def copula(n_sim, dict_dates, copula_function):
    n_days_mc = int((dict_dates['end_delivery_date'] - dict_dates['start_date']).days)
    #CALL THE DF_RAND GENERATOR FUNCTION
    tic_c =time.perf_counter()
    U1,U2 = copula_function.generate_uv(n= n_days_mc*n_sim)
    dict_copulas ={ 'U1':U1 , 'U2':U2}
    toc_c =time.perf_counter()
    print(f"Simulation of Copulas done in {toc_c - tic_c:0.4f} seconds")
    return dict_copulas