import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.optimize import curve_fit
import numpy as np
import matplotlib 
import argparse
import optuna
import pickle as pkl 
import os 
# Evaluation time
from datetime import datetime # Library for datetime format
from dateutil import relativedelta # Library to calculate delta time from date
from sklearn.metrics import mean_squared_error # Score error
from sklearn.preprocessing import StandardScaler # Standarizer for normalization

plt.rc('text', usetex=True)

def getPop(dataDir, filePop):
    ausPop = pd.read_pickle(dataDir + filePop)
    return ausPop

def getSeries(dataDir, fileData):
    # Load data
    series = pd.read_pickle(dataDir + fileData)
    return series

def getParams(dataDir, fileData):
    # Load data
    params = pd.read_pickle(dataDir + fileData)
    return params

# Defining SIIR interactive equations
def SIIR_eqs(SIIR0,t,beta1,beta2, gamma1, gamma2, beta1int, beta2int, gamma1int, gamma2int):
    SSi, ISi, SIi, IIi, RSi, SRi, RIi, IRi, RRi = SIIR0
    
    N = np.sum(SIIR0)
    
    SS = -SSi * beta1 * (ISi + IIi + IRi) / N - SSi * beta2 * (SIi + IIi + RIi) / N
    SI = beta2*SSi*(SIi + IIi + RIi)/N - beta1int*SIi*(ISi + IIi + IRi)/N - gamma2*SIi 
    SR = gamma2 * SIi - SRi * beta1 * (ISi + IIi + IRi)/N
    IS = SSi * beta1 * (ISi + IIi + IRi) / N - gamma1 * ISi - ISi * beta2int * (SIi + IIi + RIi) / N
    II = ISi * beta2int * (SIi + IIi + RIi) / N + SIi * beta1int * (ISi + IIi + IRi) / N - gamma1int * IIi - gamma2int * IIi
    IR = SRi * beta1 * (ISi + IIi + IRi) / N + gamma2int * IIi - gamma1 * IRi
    RS = gamma1 * ISi - RSi * beta2 * (SIi + IIi + RIi) / N
    RI = RSi * beta2 * (SIi + IIi + RIi) / N + gamma1int * IIi - gamma2 * RIi
    RR = gamma1 * IRi + gamma2 * RIi

    return SS, IS, SI, II, RS, SR, RI, IR, RR

# Fitting function from infected data and I state on SIIR model
def fitSIIR(t, beta1,beta2, gamma1, gamma2, beta1int, beta2int, gamma1int, gamma2int):
    sir_res = spi.odeint(SIIR_eqs, SIIR0, t, args=(beta1, beta2,
                                                   gamma1, gamma2,
                                                   beta1int, beta2int,
                                                   gamma1int, gamma2int)) 
    I1 = sir_res[:,1] + sir_res[:,3] + sir_res[:,7] 
    I2 = sir_res[:,2] + sir_res[:,3] + sir_res[:,6]
    return I1, I2

def fitErrorSIIR(trial, fluSeries, menSeries, t_range):
    k1 = 5
    k2 = 100
    beta1 = trial.suggest_float("beta1", betaFlu0/k1, betaFlu0*k1, log=True)
    beta2 = trial.suggest_float("beta2", betaMen0/k1, betaMen0*k1, log=True)
    gamma1 = trial.suggest_float("gamma1", gammaFlu0/k1, gammaFlu0*k1, log=True)
    gamma2 = trial.suggest_float("gamma2", gammaMen0/k1, gammaMen0*k1, log=True)
    beta1int = trial.suggest_float("beta1int",betaFlu0/k2,betaFlu0*k2, log=True)
    beta2int = trial.suggest_float("beta2int", betaMen0/k2, betaMen0*k2, log=True)
    gamma1int = trial.suggest_float("gamma1int", gammaFlu0/k2, gammaFlu0*k2, log=True)
    gamma2int = trial.suggest_float("gamma2int", gammaMen0/k2, gammaMen0*k2, log=True)

    I1,I2 = fitSIIR(t_range, beta1,beta2, gamma1, gamma2, 
                    beta1int, beta2int, gamma1int, gamma2int)

    sc_Inf = StandardScaler()
    sc_Men = StandardScaler()

    fSeries = np.array(fluSeries[startdate:enddate]).reshape(-1, 1)
    mSeries = np.array(menSeries[startdate:enddate]).reshape(-1, 1)
    I1 = I1.reshape(-1,1)
    I2 = I2.reshape(-1,1)

    fSeries_norm = sc_Inf.fit_transform(fSeries).flatten()
    I1_norm = sc_Inf.transform(I1).flatten()
    mSeries_norm = sc_Men.fit_transform(mSeries).flatten()
    I2_norm = sc_Men.transform(I2).flatten()
    
    error = [mean_squared_error(fSeries_norm,I1_norm),
             mean_squared_error(mSeries_norm,I2_norm)]
    return np.sum(error)

def menfluSIIRSim(beta1,beta2,gamma1, gamma2, beta1int, 
                  beta2int, gamma1int, gamma2int):    
    return spi.odeint(SIIR_eqs,SIIR0,t_eval,args=(beta1,beta2,
                                                  gamma1, gamma2,
                                                  beta1int, beta2int,
                                                  gamma1int, gamma2int)) 

if __name__ == '__main__':
    # Extract arguments from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "Input path")
    ap.add_argument("-fi", "--file-influenza", required = True, help = "File input name with infection data for Influenza")
    ap.add_argument("-fpi", "--file-params-influenza", required = True, help = "File input name parameters for Influenza (isolated)")
    ap.add_argument("-fm", "--file-meningococcal", required = True, help = "File input name with infection data for Meningococcal")
    ap.add_argument("-fpm", "--file-params-meningococcal", required = True, help = "File input name parameters for Meningoccocal (isolated)")
    ap.add_argument("-pop", "--population", required = True, help = "File input name with population data")
    ap.add_argument("-i", "--image-path", required = True, help = "Image path")
    ap.add_argument("-n", "--n-trials", required = True, help = "Number of trials")
    args = vars(ap.parse_args())
    dataDir = args["path"]
    filePop = args["population"]
    fileInf = args["file_influenza"]
    fileFluParams = args["file_params_influenza"]
    fileMen = args["file_meningococcal"]
    fileMenParams = args["file_params_meningococcal"]
    image_path = args["image_path"]
    n_trials = int(args["n_trials"])

    # Get Influenza and Meningococcal parameters
    fluMenParams = {}
    ausPop = getPop(dataDir, filePop)
    for year in ausPop.keys():
        try:
            print(year)
            startdate = str(year) + '-01'
            enddate = str(year) + '-12'
            date_format = "%Y-%m"
            sd = datetime.strptime(startdate, date_format)
            ed = datetime.strptime(enddate, date_format)
            # Get data from each disease
            fluSeries = getSeries(dataDir, fileInf)
            menSeries = getSeries(dataDir, fileMen)
            # Get parameters from isolated case
            fluParams = getParams(dataDir, fileFluParams)
            menParams = getParams(dataDir, fileMenParams)
            # Number of months (delta time)
            n_months = relativedelta.relativedelta(ed, sd)
            # Timestamp parameters
            t_start = 0.0; t_end = n_months.months; t_inc = 1
            t_range = np.arange(t_start, t_end+t_inc, t_inc)
            if len(menSeries[startdate:enddate]) != len(t_range) or len(fluSeries[startdate:enddate]) != len(t_range):
               continue
            # Initial conditionss
            SIIR0 = np.zeros(9)
            SIIR0[1] = (fluSeries[startdate])
            SIIR0[2] = (menSeries[startdate])
            SIIR0[0] = ausPop[year] - np.sum(SIIR0[1:8])

            betaFlu0 = fluParams[year][0]
            gammaFlu0 = fluParams[year][1]
            scoreFlu0 = fluParams[year][2]
            betaMen0 = menParams[year][0]
            gammaMen0 = menParams[year][1]
            scoreMen0 = menParams[year][2]

            study = optuna.create_study()
            # Optimization process
            study.optimize( lambda trial: 
                            fitErrorSIIR(trial, fluSeries, menSeries, t_range),
                            n_trials = n_trials)
            print(study.best_params)  # E.g. {'x': 2.002108042})
            beta1,beta2, gamma1, gamma2, beta1int, beta2int, gamma1int, gamma2int = study.best_params.values()
            fluMenParams[year] = [beta1,beta2, gamma1, gamma2,
                                  beta1int, beta2int, gamma1int,
                                  gamma2int, study.best_trial.value]
            # Timestamp parameters
            t_start = 0.0; t_end = n_months.months; t_inc = 0.01
            t_eval = np.arange(t_start, t_end+t_inc, t_inc)
            # Initial conditionss
            SIIR0 = np.zeros(9)
            SIIR0[1] = (fluSeries[startdate])
            SIIR0[2] = (menSeries[startdate])
            SIIR0[0] = ausPop[year] - np.sum(SIIR0[1:8])

            # Evaluation
            SIIR = menfluSIIRSim(beta1,beta2,gamma1, gamma2,
                                 beta1int, beta2int, gamma1int,
                                 gamma2int)
            IfluInt = SIIR[:,1] + SIIR[:,3] + SIIR[:,7] 
            ImenInt = SIIR[:,2] + SIIR[:,3] + SIIR[:,6]

            # Influenza plot
            plt.plot(t_range[:n_months.months+1],
                     fluSeries[startdate:enddate].values,
                     'ok',label="Original data")
            plt.plot(t_eval,IfluInt,'-r',label="Interactive fit")
            plt.legend()
            plt.title("Influenza Year: " + str(year))
            plt.savefig(image_path + "fluInteractive_" + str(year) + ".png" )
            plt.clf()

            # Meningococcal plot
            plt.plot(t_range[:n_months.months+1],
                     menSeries[startdate:enddate].values,
                     'ok',label="Original data")
            plt.plot(t_eval,ImenInt,'-r',label="Interactive fit")
            plt.legend()
            plt.title("Meningococcal Year: " + str(year))
            plt.savefig(image_path + "menInteractive_" + str(year) + ".png" )
            plt.clf()

        except KeyError:
            print("Year " + str(year) + " is not found")

    pkl.dump(fluMenParams, open(dataDir + '/flu_menSIIRParams.pkl','wb'))
    




