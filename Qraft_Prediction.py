#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import os
import glob
import pickle
import math
import matlab
import matlab.engine
from tqdm import *
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


# In[13]:


def read_program_qraft(program_name,backend):
    all_files = glob.glob(os.path.join("./real_circuits_qraft/", "*.csv"))
    for file in all_files:
        if program_name in file and backend.name in file:
            df = pd.read_csv(file)
            return df
        
def read_program_qraft_hardware(program_name,backend):
    all_files = glob.glob(os.path.join("./real_circuits_hardware_qraft/", "*.csv"))
    for file in all_files:
        if program_name in file and backend.name in file:
            df = pd.read_csv(file)
            return df
        
def get_program_dict():
    program = {}
    for file in os.listdir("./real_circuits"):
        name = file.split("_")[0]
        program[name] = {"name":name,"sqraft":0,"hqraft":0}
    return program

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


# In[3]:


# load models
eng = matlab.engine.start_matlab()


# In[14]:


backend_dict = {}
for backend in tqdm(backends):
    
    RealPrograms = get_program_dict()
    
    for program in RealPrograms.keys():

        Qraft_data = read_program_qraft(RealPrograms[program]['name'],backend)
        
        Qraft_data = Qraft_data[['CircuitWidth','CircuitDepth','CircuitNumU1Gates',
                                 'CircuitNumU2Gates','CircuitNumU3Gates','CircuitNumCXGates',
                                 'TotalUpDnErr25','TotalUpDnErr50','TotalUpDnErr75','StateHammingWeight',
                                 'StateUpProb25','StateUpProb50','StateUpProb75','StateUpDnErr25',
                                 'StateUpDnErr50','StateUpDnErr75','StateRealProb']].copy()
        

        mat = matlab.double(Qraft_data.values.tolist())
        pred = eng.prediction(mat)
        qraft_predictions = [x[0] for x in pred]       
        RealPrograms[program]["sqraft"] = np.round(HellingerDistance(Qraft_data["StateRealProb"].values,np.array(qraft_predictions),qraft=True),2)

        Qraft_data = read_program_qraft_hardware(RealPrograms[program]['name'],backend)
        
        Qraft_data = Qraft_data[['CircuitWidth','CircuitDepth','CircuitNumU1Gates',
                                 'CircuitNumU2Gates','CircuitNumU3Gates','CircuitNumCXGates',
                                 'TotalUpDnErr25','TotalUpDnErr50','TotalUpDnErr75','StateHammingWeight',
                                 'StateUpProb25','StateUpProb50','StateUpProb75','StateUpDnErr25',
                                 'StateUpDnErr50','StateUpDnErr75','StateRealProb']].copy()
        

        mat = matlab.double(Qraft_data.values.tolist())
        pred = eng.prediction(mat)
        qraft_predictions = [x[0] for x in pred]       
        RealPrograms[program]["hqraft"] = np.round(HellingerDistance(Qraft_data["StateRealProb"].values,np.array(qraft_predictions),qraft=True),2)

    backend_dict[backend.name] = RealPrograms


# In[16]:


with open("QRAFT.pkl","wb") as file:
    pickle.dump(backend_dict,file)


# In[ ]:




