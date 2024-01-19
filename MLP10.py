#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ktrain
from ktrain import tabular
import os
import glob
from tqdm import *
import pickle
import matplotlib.pyplot as plt

def read_and_combine(mode="training",drop=""):
    if mode=="training":
        all_files = glob.glob(os.path.join("./training_data", "*.csv"))
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        cols = [x for x in df.columns if x not in ["Avg_inverted_error","observed_prob_25","observed_prob_75"]]
        return df.loc[:,cols]
    elif mode=="testing":
        all_files = glob.glob(os.path.join("./testing_data", "*.csv"))
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        cols = [x for x in df.columns if x not in ["Avg_inverted_error","observed_prob_25","observed_prob_75"]]
        return df.loc[:,cols]
    
def read_hardware(drop=None):
    all_files = glob.glob(os.path.join("./real_circuits_hardware/", "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    cols = [x for x in df.columns if x not in ["Avg_inverted_error","observed_prob_25","observed_prob_75"]]
    return df.loc[:,cols]

def read_program(program_name,backend,drop=None):
    all_files = glob.glob(os.path.join("./real_circuits_hardware/", "*.csv"))
    for file in all_files:
        if program_name in file and backend.name in file:
            df = pd.read_csv(file)
            cols = [x for x in df.columns if x not in ["Avg_inverted_error","observed_prob_25","observed_prob_75"]]            
            return df.loc[:,cols]
        
def HellingerDistance(p, q, qraft=False):
    p = p/100
    q = q/100
    if not qraft:
        p[p<=0] = 0
        q[q<=0] = 0
        if q.sum()>1:
            q[q>0] = q[q>0]-((q.sum()-p.sum())/(len(q)-len(q[q==0])))
            q[q<0] = 0
    
    n = len(p)
    sum_ = 0.0
    for i in range(n):
        sum_ += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_)
    return result

def get_program_dict():
    program = {}
    for file in os.listdir("./real_circuits"):
        name = file.split("_")[0]
        program[name] = {"name":name,"mlp":0}
    return program

from qiskit_ibm_provider import IBMProvider
with open("API_KEY.txt","r") as file:
    key = file.read()
provider = IBMProvider(token=key)
def filters(x):
    if "simulator" not in x.name:
        return x
backends = provider.backends(filters=filters)
backends = sorted(backends,key=lambda x: x.name)
backends


# In[2]:


features = ['Avg_inverted_error_25', 'Avg_inverted_error_50',
       'Avg_inverted_error_75', 'Avg_odds_ratio', 'Num_1Q_Gates',
       'Num_2Q_Gates', 'circuit_depth', 'circuit_width', 'observed_prob_50',
       'state_weight']


mlp_predictor = ktrain.load_predictor("mlp")
backend_dict = {}
for backend in backends:
    
    RealPrograms = get_program_dict()
    
    for program in RealPrograms.keys():
        try:
            data = read_program(RealPrograms[program]['name'],backend)
            preds = mlp_predictor.predict(data)
            RealPrograms[program]["mlp"] = np.round(HellingerDistance(data["target"].values,preds)[0],2)
        except:
            continue

    backend_dict[backend.name] = RealPrograms

reform = {(outerKey, innerKey): values for outerKey, innerDict in backend_dict.items() for innerKey, values in innerDict.items()}
df = pd.DataFrame.from_dict(reform,orient='index')

base_value = {}
for index in df.index:
    base_value[index] = []


# In[3]:


for rep in tqdm(range(10)):
    mlp_training_data = read_and_combine()
    mlp_test_data = read_and_combine(mode="testing")
    mlp_training_data = mlp_training_data.astype('float')
    mlp_test_data = mlp_test_data.astype('float')

    trn, val, preproc = tabular.tabular_from_df(mlp_training_data, is_regression=True, 
                                             label_columns='target', val_df=mlp_test_data,verbose=0)
    mlp = tabular.tabular_regression_model('mlp', trn, hidden_layers=[128,1000,128],
                                       hidden_dropouts=[0,0.5,0], metrics=['mse'], bn=False,verbose=0)

    learner = ktrain.get_learner(mlp, train_data=trn, val_data=val, batch_size=128)
    learner.autofit(lr=1e-4,verbose=0)
    mlp_predictor_new = ktrain.get_predictor(learner.model, preproc)
    backend_dict = {}
    for backend in backends:

        RealPrograms = get_program_dict()

        for program in RealPrograms.keys():
            try:
                data = read_program(RealPrograms[program]['name'],backend)
                preds = mlp_predictor_new.predict(data)
                RealPrograms[program]["mlp"] = np.round(HellingerDistance(data["target"].values,preds)[0],2)
            except:
                continue

        backend_dict[backend.name] = RealPrograms

    reform = {(outerKey, innerKey): values for outerKey, innerDict in backend_dict.items() for innerKey, values in innerDict.items()}
    df = pd.DataFrame.from_dict(reform,orient='index')

    for index in df.index:
        base_value[index].append(df.loc[index]["mlp"])


# In[4]:


with open("mlp_10_runs","wb") as file:
    pickle.dump(base_value,file)


# In[5]:


with open("mlp_10_runs","rb") as file:
    base_value = pickle.load(file)


# In[6]:


base_value


# In[ ]:




