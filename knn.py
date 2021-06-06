import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dataModel


sns.set()


class KNN_similarity:
    def __init__(self, k):
        self.k = k

        self.data = dataModel.Data('data/preprocessed_data_training.csv', 'data/preprocessed_data_validation.csv',
                                   'data/activities.txt')

    def similarity(self, a, b):
        similarity = 0
        activitiesA = self.getActivitySet(a)
        activitiesB = self.getActivitySet(b)

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
    """
    def run(self, instance):
        response = {
            'result': 0,
            'expected': self.data.classMap[instance['mood']]
        }
        neighbors = self.knn(instance)
        for n in neighbors:
            response['result'] += self.data.classMap[self.data.training.iloc[n]['mood']]
        response['result'] /= self.k
        #if round(response['result']) == response['expected']:
        #    response['result'] = response['expected']
        response['result'] = round(response['result'])
        return response
    """

    def run(self, instance):
        expected = instance['mood']
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
        if expected == result:
            return 0
        else:
            return 1

    def getActivitySet(self, instance):
        return set(instance['activities'].split("|"))


knn = KNN_similarity(0)
r = []
for i in np.arange(1, 21):
    print(f"===== Starting k = {i} out of 20")
    knn.k = i
    error = 0
    accuracy = 100
    for j in np.arange(knn.data.validation.shape[0]):
        print(f"== j={j}/{knn.data.validation.shape[0]} (k={i}/20)")
        response = knn.run(knn.data.validation.iloc[j])
        error += pow(response, 2)
        accuracy -= response
    error /= knn.data.validation.shape[0]
    result = [i, error, accuracy]
    r.append(result)

results = pd.DataFrame(r, columns=['k', 'error', 'accuracy'])
print(results)
sns.relplot(x='k', y='error', kind='line', data=results)
plt.show()
