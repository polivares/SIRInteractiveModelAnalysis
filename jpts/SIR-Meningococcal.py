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

def getPop(dataDir,filePop):
    ausPop = pd.read_pickle(dataDir + filePop)
    return ausPop

def getMenSeries(dataDir, fileData):
    # Load data
    menSeries = pd.read_pickle(dataDir + fileData)
    return menSeries

# Defining SIR isolated equations
def SIR_eqs(SIR0, t, beta, gamma):
    S0=SIR0[0]
    I0=SIR0[1]
    R0=SIR0[2]

    S = - beta * S0 * I0/ausPop[year]
    I = (beta * S0 * I0/ausPop[year]) - gamma * I0
    R = gamma * I0

    return (S,I,R)

# Fitting function from infected data and I state on SIR model
def fitSIR(t, beta, gamma):
    return spi.odeint(SIR_eqs,SIR0,t_range,args=(beta, gamma))[:,1] 

def fitErrorSIR(trial, menSeries, t_range):
    beta = trial.suggest_float("beta",1,150,step=0.0001)
    gamma = trial.suggest_float("gamma",1,150,step=0.0001)
    SIR_Res = fitSIR(t_range, beta, gamma).reshape(-1, 1)

    scaler = StandardScaler()

    mSeries = np.array(menSeries[startdate:enddate]).reshape(-1, 1)
    mSeries_norm = scaler.fit_transform(mSeries).flatten()
    SIR_Res_norm = scaler.transform(SIR_Res).flatten()

    return mean_squared_error(mSeries, SIR_Res_norm)

def menSIRSim(beta,gamma, t_eval):    
    return spi.odeint(SIR_eqs, SIR0, t_eval, args=(beta, gamma))

if __name__ == '__main__':
    # Extract arguments from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "Input path")
    ap.add_argument("-f", "--file", required = True, help = "File input name with infection data")
    ap.add_argument("-pop", "--population", required = True, help = "File input name with population data")
    ap.add_argument("-i", "--image-path", required = True, help = "Image path")
    ap.add_argument("-n", "--n-trials", required = True, help = "Number of trials")
    args = vars(ap.parse_args())
    dataDir = args["path"]
    filePop = args["population"]
    fileData = args["file"]
    image_path = args["image_path"]
    n_trials = int(args["n_trials"])

    # Get Meningococcal parameters
    fluParams = {}
    ausPop = getPop(dataDir, filePop)
    for year in ausPop.keys():
        try:
            print(year)
            startdate = str(year) + '-01'
            enddate = str(year) + '-12'
            date_format = "%Y-%m"
            sd = datetime.strptime(startdate, date_format)
            ed = datetime.strptime(enddate, date_format)
            menSeries = getMenSeries(dataDir, fileData)
            ausPop = getPop(dataDir, filePop)
            # Number of months (delta time)
            n_months = relativedelta.relativedelta(ed, sd)
            # Timestamp parameters
            t_start = 0.0; t_end = n_months.months; t_inc = 1
            t_range = np.arange(t_start, t_end+t_inc, t_inc)
            if len(menSeries[startdate:enddate]) != len(t_range):
                continue
            # Initial conditionss
            S0 = (ausPop[year] - menSeries[startdate])
            I0 = (menSeries[startdate])
            R0 = 0
            SIR0 = [S0,I0,R0]
            study = optuna.create_study()
            # Optimization process
            study.optimize( lambda trial: 
                            fitErrorSIR(trial, menSeries, t_range),
                            n_trials = n_trials)
            print(study.best_params)  # E.g. {'x': 2.002108042})
            beta, gamma = study.best_params.values()
            fluParams[year] = [beta, gamma, study.best_trial.value]
            # Timestamp parameters
            t_start = 0.0; t_end = n_months.months; t_inc = 0.01
            t_eval = np.arange(t_start, t_end+t_inc, t_inc)
            # Initial conditions
            S0 = (ausPop[year] - menSeries[startdate])
            I0 = (menSeries[startdate])
            R0 = 0
            SIR0 = [S0, I0, R0]

            SIR = menSIRSim(beta, gamma, t_eval)
            S = SIR[:,0]
            I = SIR[:,1]
            R = SIR[:,2]
            plt.plot(t_range[:n_months.months+1], menSeries[startdate:enddate].values,
                    'ok',label="Original data")
            plt.plot(t_eval, I,'-r',label="Infected fit")
            plt.text(15, 60, r'$\gamma = $' + str(gamma))
            plt.text(15, 64, r'$\beta = $' + str(beta))
            plt.title("Year: " + str(year))
            plt.savefig(image_path + "men_" + str(year) + ".png" )
            plt.clf()
        except KeyError:
            print("Year " + str(year) + " is not found")

    pkl.dump(fluParams, open(dataDir + '/menSIRParams.pkl','wb'))
    




