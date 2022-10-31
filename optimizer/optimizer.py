import joblib
from numpy import average, var
import pandas as pd
from math import factorial
from scipy.optimize import differential_evolution

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
    X_to_predict_cnn_scaled = scaler_cnn.transform(X_to_predict_cnn)
    X_to_predict_resnet_scaled = scaler_resnet.transform(X_to_predict_resnet)
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
    mu_val = 1/(sum(service_times)/len(service_times))
    return mu_val

def calculate_energy_usage_per_experiment(X_cnn, X_resnet, X_cnn_unscaled, X_resnet_unscaled, rf_cnn_cpu, rf_resnet_cpu):

    # Skylake, Broadwell, Haswell, AMD EPYC Rome, and AMD EPYC Milan
    # 6700: 65/4, 5775c: 65/8, 4770:  84/4, EPYC 7352: 155/24, EPYC 7443: 200/24
    tdp_per_core = sum([65/4, 65/8, 84/4, 155/24, 200/24]) / 5

    energy_usages = []
    for (X, X_unscaled, rf) in [(X_cnn, X_cnn_unscaled, rf_cnn_cpu), (X_resnet, X_resnet_unscaled, rf_resnet_cpu)]:
        for i, exp in enumerate(X):
            unscaled_exp = X_unscaled.iloc[i]
            cores = unscaled_exp['CPU']
            parallel = unscaled_exp["Paralell"]
            total_cores = cores*parallel
            cpu_util = rf.predict([exp])[0]
            energy_usage = tdp_per_core * cpu_util * total_cores
            energy_usages.append(energy_usage)
    return energy_usages



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

def calculate_ERP_for_k_servers(k_servers):
    k_servers = int(k_servers)
    X_cnn_unscaled, X_resnet_unscaled, X_combined_unscaled = import_data()
    X_cnn, X_resnet, X_combined = scale_data(X_cnn_unscaled, X_resnet_unscaled, X_combined_unscaled)
    rf_combined_servicetime, rf_cnn_cpu, rf_resnet_cpu = import_regression_models()
    service_times = predict_service_times(X_combined, rf_combined_servicetime)
    mu_value = calculate_mu(service_times)
    if  not(lambda_value < mu_value): # Stability condition
        print("Lambda is not smaller than mu. The system is not stable")
        exit()
    avg_response_time = EWmgk(service_times, k_servers, lambda_value, mu_value) + 1/mu_value
    energy_usages = calculate_energy_usage_per_experiment(X_cnn, X_resnet, X_cnn_unscaled, X_resnet_unscaled, rf_cnn_cpu, rf_resnet_cpu)
    avg_energy_usage = average(energy_usages)
    ERP = avg_energy_usage * avg_response_time
    return ERP

lambda_value = 1/(0.5*3600000) # Set


bounds = [(1,100)]
result = differential_evolution(calculate_ERP_for_k_servers, bounds)
solution = result['x']
minERP = calculate_ERP_for_k_servers(solution)
print("Optimal n servers:", int(solution))
print("Watts * ms:", minERP)
print("KiloWatts * h:", minERP/1000/3600000)

# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(1, 100)
# y = [calculate_ERP_for_k_servers(x)/3600000 for x in x]

# solution = y.index(min(y))
# print(solution)
# print(calculate_ERP_for_k_servers(solution))
# # print(calculate_ERP_for_k_servers(18))

# # plt.plot(x, y)
# plt.show()