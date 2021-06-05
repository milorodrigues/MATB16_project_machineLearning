import pandas as pd


class Data:
    def __init__(self, trainingFile, validationFile, activityList):
        self.training = pd.read_csv(trainingFile)
        self.validation = pd.read_csv(validationFile)

        with open(activityList, 'r') as file:
            string = file.read()
            self.activityList = string.split("|")

        self.classLabel = self.training.columns[-1]
        self.classMap = {
            'Amazing': 5,
            'Good': 4,
            'Normal': 3,
            'Bad': 2,
            'Awful': 1
        }

