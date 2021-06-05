import pandas as pd
import numpy as np
import dataModel

data = dataModel.Data('data/preprocessed_data_training.csv', 'data/preprocessed_data_validation.csv', 'data/activities.txt')
#print(f"{data.training}\n{data.validation}\n{len(data.activityList)}")

def euclideanDistance(a,b):
    return