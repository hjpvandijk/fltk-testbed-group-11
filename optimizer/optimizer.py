import joblib
from numpy import average, var
import pandas as pd
from math import factorial, ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
import random

# configs = {
#     'exp1': {
#         'CPU cores (m)': 500,
#         'parallel' : 5,
#         'network': 'Cifar10CNN'}}

def import_data():

    configs_df = pd.read_csv('experiment_configs.csv')

    maskCNN = configs_df.model == 'Cifar10CNN'
    maskResNet = configs_df.model == 'Cifar10ResNet'

    X_to_predict_with_model = configs_df[['model','Paralell', 'CPU']]
    X_to_predict_with_model.loc[maskCNN, 'model'] = 1
    X_to_predict_with_model.loc[maskResNet, 'model'] = 2

    configs_cnn = configs_df[configs_df['model'] == 'Cifar10CNN']
    configs_resnet = configs_df[configs_df['model'] == 'Cifar10ResNet']

    X_to_predict_cnn = configs_cnn[['Paralell', 'CPU' ]]
    X_to_predict_resnet = configs_resnet[['Paralell', 'CPU']]
    return X_to_predict_cnn, X_to_predict_resnet, X_to_predict_with_model

def scale_data(X_to_predict_cnn, X_to_predict_resnet, X_to_predict_with_model):

    scaler_cnn = joblib.load('scaler_cnn.joblib')
    scaler_resnet = joblib.load('scaler_resnet.joblib')
    scaler_combined = joblib.load('scaler_combined.joblib')

    # ... and scale the features
    if len(X_to_predict_cnn) < 1:
        X_to_predict_cnn_scaled = X_to_predict_cnn
    else:
        X_to_predict_cnn_scaled = scaler_cnn.transform(X_to_predict_cnn)

    if len(X_to_predict_resnet) < 1:
        X_to_predict_resnet_scaled = X_to_predict_resnet
    else :
        X_to_predict_resnet_scaled = scaler_resnet.transform(X_to_predict_resnet)

    if len(X_to_predict_with_model) < 1:
        X_to_predict_with_model_scaled = X_to_predict_with_model
    else:    
        X_to_predict_with_model_scaled = scaler_combined.transform(X_to_predict_with_model)
    return X_to_predict_cnn_scaled, X_to_predict_resnet_scaled, X_to_predict_with_model_scaled

def import_regression_models():
    rf_combined_servicetime = joblib.load('rf_combined_servicetime.joblib')
    rf_cnn_cpu = joblib.load('rf_cnn_cpu.joblib')
    rf_resnet_cpu = joblib.load('rf_resnet_cpu.joblib')
    return rf_combined_servicetime, rf_cnn_cpu, rf_resnet_cpu

def predict_service_times(X, rf_combined_servicetime):
    service_times = []

    #Predicting service time per experiment and calculating mu
    for i, row in enumerate(X):
        service_times.append(rf_combined_servicetime.predict([row])[0])
    return service_times

def calculate_mu(service_times):
    mu_val = 1/average(service_times)
    return mu_val

def calculate_power_usage_per_experiment(X_cnn, X_resnet, X_cnn_unscaled, X_resnet_unscaled, rf_cnn_cpu, rf_resnet_cpu):

    # Skylake, Broadwell, Haswell, AMD EPYC Rome, and AMD EPYC Milan
    # 6700: 65/4, 5775c: 65/8, 4770:  84/4, EPYC 7352: 155/24, EPYC 7443: 200/24
    tdp_per_core = sum([65/4, 65/8, 84/4, 155/24, 200/24]) / 5

    power_usages = []
    for (X, X_unscaled, rf) in [(X_cnn, X_cnn_unscaled, rf_cnn_cpu), (X_resnet, X_resnet_unscaled, rf_resnet_cpu)]:
        if len(X_unscaled) != 0:
            for i, exp in enumerate(X):
                unscaled_exp = X_unscaled.iloc[i]
                cores = unscaled_exp['CPU']
                parallel = unscaled_exp["Paralell"]
                total_cores = cores*parallel
                cpu_util = rf.predict([exp])[0]
                power_usage = tdp_per_core * cpu_util * total_cores
                power_usages.append(power_usage)
    return power_usages



# C(k, lambda/mu)
def erlangC(k, rho):
    pt1 = (1-rho)
    pt2 = factorial(k)/((k*rho)**k)
    
    pt3 = 0
    for i in range(k):
        pt3 += ((k*rho)**i)/factorial(i)

    result = 1/(1+pt1*pt2*pt3)
    return result

# E[T_Q]
def Etq(k_servers, lambda_value, mu_value):
    Etq = erlangC(k_servers, lambda_value/mu_value)/(k_servers*mu_value - lambda_value)
    return Etq

# C^2
def Csquared(service_times, mu_value):
    c2 = var(service_times)/((1/mu_value)**2)
    return c2

#E[W^M/G/k]
def EWmgk(service_times, k_servers, lamda_value, mu_value):
    pt1 = (Csquared(service_times, mu_value) + 1)/2
    pt2 = Etq(k_servers, lamda_value, mu_value)
    return pt1*pt2

def calculate_ERP_for_k_servers(k_servers, lambda_value):
    k_servers = int(k_servers)
    X_cnn_unscaled, X_resnet_unscaled, X_combined_unscaled = import_data()
    X_cnn, X_resnet, X_combined = scale_data(X_cnn_unscaled, X_resnet_unscaled, X_combined_unscaled)
    rf_combined_servicetime, rf_cnn_cpu, rf_resnet_cpu = import_regression_models()
    service_times = predict_service_times(X_combined, rf_combined_servicetime)
    mu_value = calculate_mu(service_times)
    if  not(lambda_value <= k_servers * mu_value): # Stability condition
        print("Lambda is not smaller than n_servers * mu. The system is not stable")
        exit()
    avg_response_time = EWmgk(service_times, k_servers, lambda_value, mu_value) + 1/mu_value
    power_usages = calculate_power_usage_per_experiment(X_cnn, X_resnet, X_cnn_unscaled, X_resnet_unscaled, rf_cnn_cpu, rf_resnet_cpu)
    avg_power_usage = average(power_usages)
    avg_energy_usage = avg_power_usage * ((1/mu_value)/36000000) #Wh
    ERP = avg_energy_usage * avg_response_time*len(X_combined_unscaled) # Wh*ms
    ERPhours = ERP/1000/3600000 # kWh*h
    return ERPhours, mu_value

def penalty(k_servers):
    ERPhours = calculate_ERP_for_k_servers(k_servers)
    return ERPhours + (ERPhours*10**-1)*k_servers


# lambda_value = 1/(0.1*36000000) # In ms^-1. Change according to the average lambda of your experiments

def optimize(lambda_value):
    x = np.arange(1, 15)
    y_mu = [calculate_ERP_for_k_servers(x, lambda_value) for x in x]
    y = [tup[0] for tup in y_mu]
    mu = y_mu[1][1]
    ceiling = max(y)/1000
    y_rounded = [ceil(num/ceiling)*ceiling for num in y]
    solution = x[y_rounded.index(min(y_rounded))]
    minERP = min(y_rounded)
    return solution, minERP, mu

# solution, minERP, mu = optimize(lambda_value)
# print("lambda =", lambda_value)
# print("Optimal number of servers:", solut     ion)
# print("With ERP (kWh*h):", minERP)

# print("ERP theirs", calculate_ERP_for_k_servers(1, lambda_value)[0])

lambdas = np.arange(0.05, 2, 0.05)
# print(1/(0.1*3600000))
# print(1/(2*3600000))
# print(1/(0.05*3600000))
# lambdas = np.arange(0, 0.5, 0.1)
X_solution = []
erps_solution = []
X_rot = []
erps_rot = []
rhos = []

for i in lambdas:
    lambda_value = 1/(i*3600000)
    solution, minERP, mu = optimize(lambda_value)
    rho = lambda_value/mu
    print((1/mu)/60000)
    rhos.append(rho)
    rot = int(max(1, rho + sqrt(rho)))
    erp_rot, _ = calculate_ERP_for_k_servers(rot, lambda_value)
    print("solution:", solution, "with min ERP=", minERP)
    print("rot:", rot, "with ERP=", erp_rot)
    X_solution.append(solution)
    erps_solution.append(minERP)
    X_rot.append(rot)
    erps_rot.append(erp_rot)

plt.plot(rhos, erps_solution, label="Optimizer")
plt.plot(rhos, erps_rot, label="Hall(1991) Rule of Thumb")
plt.xlabel("ρ")
plt.ylabel("ERP (kWh•h")
plt.legend()
plt.savefig("optimizer_vs_ruleofthumb.pdf")
plt.show()