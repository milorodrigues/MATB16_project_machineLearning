import numpy as np
import pandas as pd


class Data:
    def __init__(self, trainingFile, validationFile, activityList):
        self.training = pd.read_csv(trainingFile)
        self.validation = pd.read_csv(validationFile)

        with open(activityList, 'r') as file:
            string = file.read()
            self.activitySet = set(string.split("|"))

        self.classLabel = self.training.columns[-1]
        self.classMap = {
            'Amazing': 4,
            'Good': 3,
            'Normal': 2,
            'Bad': 1,
            'Awful': 0
        }

    def convertActivitiesToList(self):
        for i in np.arange(self.training.shape[0]):
            self.training.iloc[i]['activities'] = list(self.getActivitySet(self.training.iloc[i]))
        for i in np.arange(self.validation.shape[0]):
            self.validation.iloc[i]['activities'] = list(self.getActivitySet(self.validation.iloc[i]))

    def getActivitySet(self, instance):
        return frozenset(instance['activities'].split("|"))
