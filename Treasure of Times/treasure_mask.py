import numpy as np
import pandas as pd

def ItemEncoder(item_df,set1,set2):
    """Cretes a column encoding each item in tnto the dataframe"""
    df = item_df
    for index,row in df.iterrows():
        if row['set'] == set1 and row['part'] == 'outfit':
            df.loc[index,'onehot'] = 1
        elif row['set'] == set1 and row['part'] == 'hair':
            df.loc[index,'onehot'] = 2
        elif row['set'] == set2 and row['part'] == 'outfit':
            df.loc[index,'onehot'] = 3
        elif row['set'] == set2 and row['part'] == 'hair':
            df.loc[index,'onehot'] = 4
        else:
            df.loc[index,'onehot'] = 0
    return df

def all_mask(array,result=0):
    """Mask array to catch only successful draws
       array: Arrays containing result of draws
       result: Specify what type of result to return
        - 0: Return masked array
        - 1: Return mean of each simulation
        - 2: Return mean of means (For bootstraping)"""
    temp = np.where(array > 0,1,0)
    if result == 1:
        temp = np.mean(temp,axis=0)
    if result == 2:
        temp = np.mean(np.mean(temp,axis=0))
        
    return temp

def outfit_mask(array,result=0):
    """Mask array to catch only outfit draws
       array: Arrays containing result of draws
       result: Specify what type of result to return
        - 0: Return masked array
        - 1: Return mean of each simulation
        - 2: Return mean of means (For bootstraping)"""
    temp = np.isin(array,[1,3])
    if result == 1:
        temp = np.mean(temp,axis=0)
    if result == 2:
        temp = np.mean(np.mean(temp,axis=0))
    return temp

def hair_mask(array,result=0):
    """Mask array to catch only hair draws
       array: Arrays containing result of draws
       result: Specify what type of result to return
        - 0: Return masked array
        - 1: Return mean of each simulation
        - 2: Return mean of means (For bootstraping)"""
    temp = np.isin(array,[2,4])
    if result == 1:
        temp = np.mean(temp,axis=0)
    if result == 2:
        temp = np.mean(np.mean(temp,axis=0))
    return temp

def individual_mask(array,target,result=0):
    """Mask array to catch only specific item draws
       array: Arrays containing result of draws
       target: Number of item in interest from ItemEncoder
       result: Specify what type of result to return
        - 0: Return masked array
        - 1: Return mean of each simulation
        - 2: Return mean of means (For bootstraping)"""
    temp = np.where(array == target,1,0)
    if result == 1:
        temp = np.mean(temp,axis=0)
    if result == 2:
        temp = np.mean(np.mean(temp,axis=0))
    return temp

def multi_mask(array,target,result=0):
    """Mask the array to catch only fashion set in interst
    array: Arrays containing result of draws
    target: List of item interested
    result: Specify what type of result to return
        - 0: Return masked array
        - 1: Return mean of each simulation
        - 2: Return mean of means (For bootstraping)"""
    temp = np.isin(array,target)
    if result == 1:
        temp = np.mean(temp,axis=0)
    if result == 2:
        temp = np.mean(np.mean(temp,axis=0))
    return temp    

def draw_tot(array,size=1000,sample=100):
    """Draw treasure of times
       array: target array containing sample results
       size: Number of simulations
       sample: Number of draw per simulation"""     
    temp = np.random.choice(array,(sample,size),replace=True)
    return temp

def bootstrap_tot(array,num_iter,func=all_mask,size=1000,sample=100,target=0):
    temp_array = np.empty(num_iter)
    if func == individual_mask or func == multi_mask:
        for i in range(num_iter):
            bypass_array = draw_tot(array,size,sample)
            temp_array[i] = func(bypass_array,result=2,target=target)
    else:
        for i in range(num_iter):
            bypass_array = draw_tot(array,size,sample)
            temp_array[i] = func(bypass_array,result=2)
    return temp_array
    
