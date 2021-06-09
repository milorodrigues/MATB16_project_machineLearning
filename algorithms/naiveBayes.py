import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataModel

sns.set()

data = dataModel.Data('../data/preprocessed_data_training.csv', '../data/preprocessed_data_validation.csv',
                      '../data/activities.txt')
data.convertActivitiesToList()


def generateProbabilityDF(data):
    activities = data.activitySet
    cols = ['activity', 'value']
    cols.extend(data.labels)

    dfList = []
    for a in activities:
        dfList.append([a, True, -1, -1, -1, -1, -1])
        dfList.append([a, False, -1, -1, -1, -1, -1])

    df = pd.DataFrame(dfList, columns=cols)
    return df


def baseProbability(dataset):
    amountAll = dataset.shape[0]
    amounts = {
        'Amazing': 0,
        'Good': 0,
        'Normal': 0,
        'Bad': 0,
        'Awful': 0
    }
    count = dataset['mood'].value_counts()

    for a in amounts:
        amounts[a] = count[a] / amountAll

    return amounts


def conditionalProbability(dataset, a, aCol, b, bCol):
    datasetWhereB = dataset[dataset[bCol] == b]
    amountWhereB = datasetWhereB.shape[0]
    amountWhereAandB = 0

    for i in np.arange(amountWhereB):
        instance = datasetWhereB.iloc[i]
        if (aCol in instance['activities']) == a:
            amountWhereAandB += 1

    return amountWhereAandB / amountWhereB


def naiveBayes(instance, dataset, baseProb, probDF, activitySet):
    probabilities = {
        'Amazing': 0,
        'Good': 0,
        'Normal': 0,
        'Bad': 0,
        'Awful': 0
    }

    for label in probabilities:
        probabilities[label] = baseProb[label]

        activities = instance['activities']
        for a in activitySet:
            value = a in activities
            p = probDF[(probDF['activity'] == a) & (probDF['value'] == value)].iloc[0][label]

            if p == -1:
                p = conditionalProbability(dataset, value, a, label, 'mood')
                probDF.loc[(probDF['activity'] == a) & (probDF['value'] == value), label] = p
            probabilities[label] *= p

    normFactor = sum(probabilities.values())

    for label in probabilities:
        probabilities[label] = probabilities[label] / normFactor

    chosen = max(probabilities, key=lambda key: probabilities[key])
    expected = instance['mood']

    return {'chosen': chosen, 'expected': expected}


baseProb = baseProbability(data.training)
prob = generateProbabilityDF(data)

resultList = []
dChosen = {
        'Amazing': 0,
        'Good': 0,
        'Normal': 0,
        'Bad': 0,
        'Awful': 0
    }
dExpected = {
        'Amazing': 0,
        'Good': 0,
        'Normal': 0,
        'Bad': 0,
        'Awful': 0
    }
accuracy = 0

for i in np.arange(data.validation.shape[0]):
    print(f"====> i = {i}")
    instance = data.validation.iloc[i]
    r = naiveBayes(instance, data.training, baseProb, prob, data.activitySet)
    resultList.append(r)
    dChosen[r['chosen']] += 1
    dExpected[r['expected']] += 1

    if r['chosen'] == r['expected']:
        print("correct")
        accuracy +=1
    else:
        print("wrong")

prob.to_csv('../data/naiveBayes_probabilities.csv')
print(f"accuracy = {accuracy}")
print(f"dChosen = {dChosen}")
print(f"dExpected = {dExpected}")
