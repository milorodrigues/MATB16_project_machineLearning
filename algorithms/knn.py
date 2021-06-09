import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataModel


class KNN_similarity:
    def __init__(self, k):
        self.k = k

        self.data = dataModel.Data('../data/preprocessed_data_training.csv', '../data/preprocessed_data_validation.csv',
                                   '../data/activities.txt')

    def similarity(self, a, b):
        similarity = 0
        activitiesA = self.data.getActivitySet(a)
        activitiesB = self.data.getActivitySet(b)

        if a['weekday'] == b['weekday']:
            similarity += 1

        if len(activitiesA) > len(activitiesB):
            for i in activitiesB:
                if i in activitiesA:
                    similarity += 1
        else:
            for i in activitiesA:
                if i in activitiesB:
                    similarity += 1

        similarity *= min((len(activitiesA) + 1 / len(activitiesB) + 1), (len(activitiesB) + 1 / len(activitiesA) + 1))
        return similarity

    def knn(self, instance):
        similarities = np.zeros(self.data.training.shape[0])
        for i in np.arange(self.data.training.shape[0]):
            similarities[i] = self.similarity(instance, self.data.training.iloc[i])
        return np.argsort(similarities)[::-1][:self.k]

    def run(self, instance):
        neighbors = self.knn(instance)
        count = {
            'Amazing': 0,
            'Good': 0,
            'Normal': 0,
            'Bad': 0,
            'Awful': 0
        }

        for n in neighbors:
            count[self.data.training.iloc[n]['mood']] += 1

        result = max(count, key=lambda key: count[key])
        return result

    def getActivitySet(self, instance):
        return set(instance['activities'].split("|"))



sns.set()
knn = KNN_similarity(0)


def runKNNmultiK(k):
    r = []
    for i in np.arange(1, k):
        print(f"===== Starting k = {i} out of {k}")
        knn.k = i
        accuracy = 0
        for j in np.arange(knn.data.validation.shape[0]):
            print(f"== j={j+1}/{knn.data.validation.shape[0]} (k={i}/{k})")
            expected = knn.data.validation.iloc[j]['mood']
            response = knn.run(knn.data.validation.iloc[j])
            if expected == response:
                accuracy += 1
        result = [i, accuracy]
        r.append(result)

    results = pd.DataFrame(r, columns=['k', 'accuracy'])
    print(results)
    plot = sns.relplot(x='k', y='accuracy', kind='line', data=results)
    plot.set(ylim=(0, 100))
    plt.show()
    return


def runKNNsingle(k):
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
    knn.k = k
    accuracy = 100
    for j in np.arange(knn.data.validation.shape[0]):
        expected = knn.data.validation.iloc[j]['mood']
        print(f"== j={j+1}/{knn.data.validation.shape[0]}")
        response = knn.run(knn.data.validation.iloc[j])
        if response != expected:
            accuracy -= 1
        dChosen[response] += 1
        dExpected[expected] += 1
    print(f"accuracy = {accuracy}")
    print(f"dChosen = {dChosen}")
    print(f"dExpected = {dExpected}")


runKNNsingle(16)
