import pandas as pd


# import numpy as np


class Data:
    def __init__(self, file):
        self.raw = pd.read_csv(file)
        print(f"{self.raw}")

        self.classLabel = self.raw.columns[-1]

        self.all = None
        self.attributes = None

        self.preprocess()

    def preprocess(self):
        rawCols = self.raw.columns
        print(f"{rawCols}")

        #Removendo colunas que não têm a ver com o humor
        self.all = self.raw.drop(['full_date', 'date', 'time'], axis=1)

        print(f"{self.raw[self.classLabel].unique()}")

