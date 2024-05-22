import random 
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from matplotlib import dates



class McSimulation:
    
    
    def __init__(self,
                dict_OU,
                dict_EUA,
                list_seasonality,
                dict_dates,
                seed=1,
                jump=False
                #seasonality partial così da esssere in funzione solo di n_days????
    ):
        self.pun = dict_OU
        self.eua = dict_EUA
        self.l_seas = list_seasonality
        self.d_dates = dict_dates
        self.seed= seed
        self.jump=jump
        self.n_days_before_start = int((dict_dates['start_date'] - dict_dates['start_seasonality_date']).days)
        self.n_days_mc = int((dict_dates['end_delivery_date'] - dict_dates['start_date']).days) #TODO:+1 --> impact N1,N2 & for loop
        self.n_days_delivery = int((dict_dates['end_delivery_date'] - dict_dates['start_delivery_date']).days)
    

    def rand_constructor(self, num_mc_sim, d_copulas):
        """ CONSTRUCTION OF RANDOM VALUES LISTS"""
        d = d_copulas
        K = num_mc_sim
        T = self.n_days_mc
        tot = K*T
        #generate normalized values
        N1 = [norm.ppf(u) for u in d['U1']]
        N2 = [norm.ppf(u) for u in d['U2']]
        #dict
        dict_rand={'N1': N1 , 'N2': N2}
        #IF statement in case of Jump
        if self.jump==True:
            dict_rand['N_J']=[norm.ppf((random.uniform(0, 1))) for n in range(0,tot)] #normal random value
            dict_rand['j'] = [np.random.binomial(1,self.pun['THETA']) for n in range(0,tot)] #bernoulli random value
        #reshape
        for key in dict_rand.keys():
            dict_rand[key]= pd.DataFrame(data=np.reshape(dict_rand[key], (K,T)))
        
        #classifiy
        self.dict_rand = dict_rand
        return self.dict_rand 


    def mc_sim(self, num_mc_sim):
        """ MC simulation method, based on number of MC simulations"""

        """ initialization"""
        K = num_mc_sim
        T = self.n_days_mc
        d = self.dict_rand.copy()
        pun = self.pun
        eua = self.eua
        dt = 1 #it is fixed in my prooject. It is daily
        t0 = self.n_days_before_start
        #f_0 = self.seasonality(t0) 
        X_0 = np.log(pun['INIT_PRICE']) - self.l_seas[0]

        list_sim_pun =[]
        list_sim_eua =[]
        list_sim_f =[]
        list_sim_x =[]
        """ start iterations """
        for k in range(0,K):
                
            Xt= [] #deseasonalized log price for a single iteration
            Xt.append(X_0) #appended the initial value
            At=[] #eua spot price for a single iteration
            At.append(eua['INIT_PRICE']) #appended the initial value
            St= []
            St.append(pun['INIT_PRICE'])
            ft = self.l_seas
            #for-loop 
            for t in range(0,T): #TODO:o sto conforme a d[] oppure a seasonality
                """ PUN Ieration"""
                X = (1-pun['ALPHA'])*Xt[-1]*dt - pun['LAMBDA']*pun['SIGMA']*dt + pun['SIGMA']*np.sqrt(dt)*d['N1'].iat[k,t]
                if self.jump==True:
                    X = X + d['j'].iat[k,t]*((pun['MU_J'] -pun['MRP_J']*pun['SIGMA_J'])*dt + pun['SIGMA_J']*np.sqrt(dt)*d['N_J'].iat[k,t])
                Xt.append(X)
                """ EUA Ieration"""
                A = At[-1]*np.exp( (eua['MU'] - 0.5*eua['SIGMA']**2 - eua['MRP']*eua['SIGMA'])*dt + eua['SIGMA']*np.sqrt(dt)*d['N2'].iat[k,t])
                At.append(A)
                """ PUN Construction """
                t_p1 = t+1
                St.append( np.exp(ft[t_p1] + 1.41*Xt[t_p1]))     

            #save in a list where each simulation represents a single item/row

            list_sim_x.append(Xt)
            list_sim_f.append(ft)
            list_sim_pun.append(St)
            list_sim_eua.append(At)
        
        """ COMPUTE THE MEAN AMONG ITERATIONS"""
        # append the mean of each point in time among all iterations
        St_m=[]
        At_m=[]
        Xt_m=[]
        ft_m=[]

        for n in range(0,len(list_sim_pun[0])):
            St_m.append(np.mean (np.array(list_sim_pun).T.tolist()[n])) 
            At_m.append(np.mean (np.array(list_sim_eua).T.tolist()[n]))
            Xt_m.append(np.mean (np.array(list_sim_x).T.tolist()[n]  ))
            #ft_m.append(np.mean (np.array(list_sim_f).T.tolist()[n]  ))
        
        """ CREATE A DICTIONARY TO STORE OUTPUT DATA"""
        self.dict_output={
            'SIM_PUN': list_sim_pun,
            'SIM_EUA': list_sim_eua,
            'SIM_Xt': list_sim_x,
            #'SIM_ft': list_sim_f,
            'PUN_m': St_m,
            'EUA_m': At_m,
            'Xt_m': Xt_m
            #'ft_m': ft_m
        }
  
        return self.dict_output
    

    def plots(self, t0=0):
        """ PLOT EACH TIME SERIES AND THE AVERAGED OUTPUTS"""
        d = self.dict_output.copy()
        sim_dates = pd.date_range(start=self.d_dates['start_date'], end=self.d_dates['end_delivery_date'])
        
        fig, ax = plt.subplots(4,1, figsize=(10,10))
        locator = dates.AutoDateLocator()
        formatter = dates.ConciseDateFormatter(locator)
        fig.tight_layout()
        
        ax[0].plot(sim_dates[t0:],d['PUN_m'][t0:], 'black')
        ax[0].set_xlabel('Dates')
        ax[0].set_ylabel('Energy Prices')
        ax[0].xaxis.set_major_formatter(formatter)

        ax[1].plot(sim_dates[t0:],d['EUA_m'][t0:], 'black')
        ax[1].set_xlabel('Dates')
        ax[1].set_ylabel('EUAs Prices')
        ax[1].xaxis.set_major_formatter(formatter)

        ax[2].plot(sim_dates[t0:],d['Xt_m'][t0:], 'black')
        ax[2].set_xlabel('Dates')
        ax[2].set_ylabel('Deseasonalized Part')
        ax[2].xaxis.set_major_formatter(formatter)

        ax[3].plot(sim_dates[t0:],self.l_seas[t0:], 'black')
        ax[3].set_xlabel('Dates')
        ax[3].set_ylabel('Seasonality Part')
        ax[3].xaxis.set_major_formatter(formatter)
        plt.show()
        
    
    def starter(self, num_mc_sim, d_copulas, plot=None):
        
        self.rand_constructor(num_mc_sim, d_copulas)
        tic =time.perf_counter()
        output = self.mc_sim(num_mc_sim)
        toc =time.perf_counter()
        print(f"Simulation done in {toc - tic:0.4f} seconds")
        """ PLOT STARTER"""      
        if plot== 'ALL':
            print('___COMPLETE SIMULATED PATH PLOTS___') 
            self.plots()  
        elif plot== 'DELIVERY':
            print('___ONLY DELIVERY-PERIOD PLOTS___')
            self.plots(self.n_days_mc - self.n_days_delivery)
        elif plot =='BOTH':
            print('___COMPLETE SIMULATED PATH PLOTS___')
            self.plots()
            print('___ONLY DELIVERY-PERIOD PLOTS___')
            self.plots(self.n_days_mc - self.n_days_delivery)
        elif plot== None:
            print('No plots option')
        else:
            print('Inserted a un-corrected PLOTS option')
   
        return output