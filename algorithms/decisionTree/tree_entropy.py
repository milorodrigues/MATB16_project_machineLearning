import math
import numpy as np
import pandas as pd


def entropy(dataset, classLabel, classes):
    e = 0
    count = dataset[classLabel].value_counts()
    if count.shape[0] == 1:
        return 0
    for l in classes:
        if l in count.index:
            p = count[l] / dataset.shape[0]
            e += -p * math.log(p, count.shape[0])
    return e

def entropy2(count, classes):
    e = 0
    nonZeroCount = 0
    for c in count:
        if count[c] > 0:
            nonZeroCount += 1

    length = sum(count.values())
    if nonZeroCount <= 0:
        return 0

    for l in classes:
        if count[l] > 0:
            p = count[l] / length
            e += -p * math.log(p, nonZeroCount)
    return e

def infoGain(dataset, attribute, classLabel, classes):
    #print(attribute)
    gain = entropy(dataset, classLabel, classes)

    values = [True, False]
    for v in values:
        #newDataset = []
        count = {
            'Amazing': 0,
            'Good': 0,
            'Normal': 0,
            'Bad': 0,
            'Awful': 0
        }
        for i in np.arange(dataset.shape[0]):
            if (attribute in dataset.iloc[i]['activities']) == v:
                count[dataset.iloc[i]['mood']] += 1
        #dataset2 = pd.DataFrame(newDataset, columns=['weekday', 'activities', 'mood'])
        gain += -((sum(count.values())/dataset.shape[0]) * entropy2(count, classes))
    return gain
