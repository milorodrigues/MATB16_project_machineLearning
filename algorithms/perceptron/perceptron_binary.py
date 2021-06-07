import numpy as np
import pandas as pd
import perceptron_node
import dataModel


class Network:
    def __init__(self, eta, epochs, threshold):
        self.data = dataModel.Data('../../data/preprocessed_data_training.csv',
                                   '../../data/preprocessed_data_validation.csv',
                                   '../../data/activities.txt')
        self.eta = eta
        self.epochs = epochs
        self.perceptron = perceptron_node.Perceptron(self.data, eta, threshold)  # Checks if >= Normal

    def analyze(self):
        analysis = []
        for e in np.arange(1, self.epochs + 1):
            print(f"=>>>>>> [[BINARY]] Epoch {e}/{self.epochs}")
            for i in np.arange(self.data.training.shape[0]):
                print(f"[[BINARY]] Training instance {i + 1}/{self.data.training.shape[0]} => Epoch {e}/{self.epochs}")
                instance = self.data.training.iloc[i]
                self.perceptron.trainInstance(instance)

            accuracy = 0
            for i in np.arange(self.data.validation.shape[0]):
                print(f"[[BINARY]] Validating instance {i + 1}/{self.data.validation.shape[0]} => Epoch {e}/{self.epochs}")
                instance = self.data.validation.iloc[i]
                expected = self.data.classMap[instance['mood']]
                if expected >= self.perceptron.threshold:
                    expected = 1
                else:
                    expected = -1

                result = self.perceptron.validateInstance(instance)

                if result == expected:
                    accuracy += 1
            analysis.append([e, accuracy])

        analysisDF = pd.DataFrame(analysis, columns=['epoch', 'classified correctly'])
        return analysisDF
