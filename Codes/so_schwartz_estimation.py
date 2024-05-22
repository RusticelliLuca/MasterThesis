import pandas as pd
import numpy as np
from scipy.special import laguerre,hermite,legendre


class SchwartzEstimation:
    
    def __init__(self,
            pos_delivery_start,
            n_ex_options,
            ester,
            EUA_denominator,
            estimation_method,
            dict_basis_f
            ):
        self.n_days_before_delivery = pos_delivery_start
        self.N = n_ex_options
        self.ester = ester
        self.EUA_denominator = EUA_denominator
        self.estimation_method = estimation_method
        #basis function coeff
        self.basis_f =  dict_basis_f['POLY FAMILY']
        self.n_basis =  dict_basis_f['APRX N']
        self.w_basis = dict_basis_f['WEIGHT FACTOR']
        
    

    def basis_f_gen(self,reg_data, n ,basis_f, c): #where reg is the list of sim St ITM for that time

        def basis_f_single_line(x, n, basis_f, c): #n is the degree of laguerre basis_polinomials
            X=[]
            for j in range(1,n+1):
                r = basis_f(j)(x)*np.exp(-x**c) #MODIFICATO SOLO PER UN CASO
                X.append(r)
            return X

        X_final =[]
        for pos in range(0,len(reg_data)):
            X_final.append( basis_f_single_line(reg_data[pos], n, basis_f, c))
        return X_final
        #we have the regressors on the column line
        #one row for each simulation as wanted by the estimation method 
       
    
    def aprx_max_price(self, dict_out, strike):
        St_m = np.array(dict_out['PUN_m']).T.tolist()[self.n_days_before_delivery:][::-1]
        At_m = np.array(dict_out['EUA_m']).T.tolist()[self.n_days_before_delivery:][::-1] 
        N = self.N
        den = self.EUA_denominator
        payoff=[ max(s-strike,0)*a/den for s,a in zip(St_m,At_m)]
        payoff.sort(reverse=True)
        return sum(payoff[:(N+1)])*np.exp(-self.ester/365*self.n_days_before_delivery)


#_____________________________________________________________________________
    def estimation(self, dict_out, strike, coeff =None):
        #PARAMETERS
        N = self.N #number of exercise options
        ester = self.ester #discount rate
        EUA_den_factor = self.EUA_denominator
        estimation_method = self.estimation_method
        """ PREPARATION OF THE PARAMETERS AND DATA"""
        """ Time Series Data:
        we want them ordered from the farest to the nearest, only in the delivery period
         - .T <- inverts rows and column. so we pass from a K x T to a T x K
            where K=n of simulations, and T = number of steps in delivery
         - [self.n_days_before_delivery:] <-  kepps only the delivery period
         - [::-1] <- serves to invert the list. so we start from the end
        """
        St = np.array(dict_out['SIM_PUN']).T.tolist()[self.n_days_before_delivery:][::-1]
        At = np.array(dict_out['SIM_EUA']).T.tolist()[self.n_days_before_delivery:][::-1] 
        #exstract parameters for iterations
        T = len(St)
        K = len(St[0])
        #discount factor
        DF = np.exp(-ester/365) # daily discount factor
        Final_DF = np.exp(-ester/365*self.n_days_before_delivery)
        """ ITERATION """
        #inizialization
        if not coeff: # 1ST CASE - coeff TO BE ESTIMATED
            coeff_i_n_k = []
        else:
            coeff_i_n_k = coeff
        rmse_i_n_k = []
        V_i_n_k = []
        
        
        #for loop along times steps
        
        for i in range(0,T):
            """1) compute the payoff call option payoff on PUN for each k-iteration -> Zt """
            Zt = []
            At_mod=[]
            for k in range(0,K):
                Zt.append( max( (St[i][k] - strike) ,0) )
                """ #2) compute EUA multiplication factor """
                At_mod.append( At[i][k] / EUA_den_factor)

            """ 3) Preparation for the REGRESSION: """
            """Starting case for i=0 -> initialize V_i_n_k filled with zeros"""
            if i == 0:
                V_2=[]
                for n in range(0,N):
                    V_1=[]
                    for k in range(0,K):
                        V_1.append(0)
                    V_2.append(V_1)
                V_i_n_k.append(V_2)

            """definition of Y & X-variables based on previous and current times "i"""
            reg_data = []
            applied_data =[]
            Y_n_k = []
            Q_k = []
            """ Y definiton"""
            for k in range(0,K):
                Q_k.append(0) #We need it for the calculation of the current V_i_n_k when n=1 to not go out of bounds
                applied_data.append(Zt[k]/strike)
                Y_k = []
                #if statment since we want only ITM obs as input data for regression
                if (not coeff):
                    if Zt[k] >= 0: # 1ST CASE - coeff TO BE ESTIMATED
                        reg_data.append(Zt[k]/strike)
                        #extract the corrisponding Y values for the correct k positions to have the same simulations
                        for n in range(0,N):
                            Y_k.append( V_i_n_k[-1][n][k] ) #value functions at the previous upper time already discounted
                        Y_n_k.append( Y_k ) #to obtain a row for each k
                else: # 2ND CASE - coeff ALREADY PRESENT
                    for n in range(0,N):
                        Y_k.append( V_i_n_k[-1][n][k] ) #value functions at the previous upper time already discounted
                    Y_n_k.append(Y_k)
            #Y has k-iterations on rows and N-steps cols, invert it to have N in rows as regressors:
            Y = np.array(Y_n_k).T.tolist()

            """ X definition"""
            if self.basis_f == hermite:
                w_basis = 2
            else:
                w_basis = 1 #laguerre
            if not coeff:
                X = self.basis_f_gen(reg_data, self.n_basis, self.basis_f, w_basis) #sono gli stessi per ogni n, cambiano rispetto a "i"
            #hanno lungo le righe le k simulazioni. non hanno tutti i valori perchè scremato sopra.
            #calcolo regressors anche per tutti le sim perchè mi servono per stimare tutti i Q
            X_applied = self.basis_f_gen(applied_data, self.n_basis, self.basis_f, w_basis)

            """ ESTIMATION OF DISCOUNTED VALUE FUNCTIONS"""
            #initialization
            coeff_n_k =[]
            rmse_n_k = [] 
            Q_n_k = []
            Q_n_k.append(Q_k) #first list inside, it represents Q(n=0)
            for n in range(0,N): #regressioni per ogni n in N dato che la var dipendente cambia a seconda di n
                if not coeff: # 1ST CASE - coeff TO BE ESTIMATED
                    fit=estimation_method(Y[n] ,X).fit()
                    coeff_n_k.append(fit.params)
                    y_hat = fit.predict(X) #save y_hat for the RMSE
                    Q_n_k.append( fit.predict(X_applied) ) #fit the all sample of k-simulations for the computation of V_i_n_k
                    rmse_n_k.append( np.sqrt(np.sum((np.array(Y[n]) - np.array(y_hat))**2) / len(Y[n]) ) ) #compute RMSE for each single estimation
                else:
                    Q_n_k.append( np.matmul( X_applied, np.array(coeff_i_n_k[i][n]).T.tolist() ) )
                    rmse_n_k.append( np.sqrt(np.sum((np.array(Y[n]) - Q_n_k[-1])**2) / len(Y[n]) ))

            
            """ SAVING OF coeff COEFFICIENTS AND RMSE VAUE"""
            if not coeff: # 1ST CASE - coeff TO BE ESTIMATED
                coeff_i_n_k.append( coeff_n_k) 
            rmse_i_n_k.append(rmse_n_k)

            """ 5) calcolo Value function """
            #initialize
            V_n_k = []
            for n in range(1,N+1): 
                V_k=[]
                for k in range(0,K): 
                    V_k.append( max( Zt[k] + Q_n_k[n-1][k] , Q_n_k[n][k] ) *DF )

                #save values before the next "n"    
                V_n_k.append(V_k) #each row is a different n, each column a different k
                
            
            V_i_n_k.append(V_n_k) #to utilize it for the next i
        #____________________________________________________________________________________    
        #the final price is given by the mean between V(t=0,) of all k pathsfor n=N since it is the beginning:
        
        price_swing_forward = np.mean(V_i_n_k[-1][-1] ) / DF #(because they are discounted by DF before)
        price_swing_today = price_swing_forward * Final_DF
        #____________________________________________________________________________________
        #RMSE total
        rmse=0
        for i in range(0,len(rmse_i_n_k)):
            for n in range(0,len(rmse_i_n_k[i])):
                    rmse = rmse+rmse_i_n_k[i][n]
        """ MAX APPROXIMATED PRICE CALCULATION """
        
        return price_swing_today, coeff_i_n_k, rmse
