import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dataModel
import perceptron_node

sns.set()


class Network:
    def __init__(self, eta, epochs):
        self.data = dataModel.Data('../../data/preprocessed_data_training.csv',
                                   '../../data/preprocessed_data_validation.csv',
                                   '../../data/activities.txt')
        self.eta = eta
        self.epochs = epochs
        self.perceptrons = [
            perceptron_node.Perceptron(self.data, eta, 1),  # Checks if >= Bad
            perceptron_node.Perceptron(self.data, eta, 2),  # Checks if >= Normal
            perceptron_node.Perceptron(self.data, eta, 3),  # Checks if >= Good
            perceptron_node.Perceptron(self.data, eta, 4)  # Checks if >= Amazing
        ]

    def train(self, epoch):
        for i in np.arange(self.data.training.shape[0]):
            print(f"[[5C]] Training instance {i+1}/{self.data.training.shape[0]} => Epoch {epoch}/{self.epochs}")
            instance = self.data.training.iloc[i]

            rList = [0, 0, 0, 0]
            for p in self.perceptrons:
                rList[p.threshold - 1] = p.trainInstance(instance)['result']
        return
    
    def validate(self, epoch):
        accuracy = 0
        for i in np.arange(self.data.validation.shape[0]):
            print(f"[[5C]] Training instance {i+1}/{self.data.validation.shape[0]} => Epoch {epoch}/{self.epochs}")
            instance = self.data.validation.iloc[i]
            expected = self.data.classMap[instance['mood']]

            rList = [0, 0, 0, 0]
            for p in self.perceptrons:
                rList[p.threshold - 1] = p.validateInstance(instance)

            result = 0
            for r in rList:
                if r > 0:
                    result += 1
                else:
                    break

            if result == expected:
                accuracy += 1
        return accuracy

    def analyze(self):
        analysis = []
        for e in np.arange(1, self.epochs + 1):
            print(f"=>>>>>> [[5C]] Epoch {e}/{self.epochs}")
            self.train(e)
            a = self.validate(e)
            analysis.append([e, a])

        analysisDF = pd.DataFrame(analysis, columns=['epoch', 'classified correctly'])
        return analysisDF
