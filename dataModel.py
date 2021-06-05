import pandas as pd


class Data:
    def __init__(self, file):
        self.all = pd.read_csv(file)

        self.classLabel = self.all.columns[-1]

