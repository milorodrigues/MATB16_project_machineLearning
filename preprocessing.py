import pandas as pd
import numpy as np
import random

# dataset source:
# https://www.kaggle.com/kingabzpro/daylio-mood-tracker

# Leitura do csv
raw = pd.read_csv('data/Daylio_Abid.csv')
# Remoção das colunas que não serão usadas
raw.drop(['full_date', 'date', 'time', 'sub_mood'], axis=1, inplace=True)
# Remoção das linhas com informação vazia
raw.dropna(subset=['weekday', 'activities', 'mood'], inplace=True)

# Determinação da lista de atividades possíveis + reformatação da coluna activities
activities = set()
for i in np.arange(raw.shape[0]):
    a = raw.iloc[i]['activities'].split("|")
    for j in np.arange(len(a)):
        a[j] = a[j].strip()
        if a[j] == "Write dairy":
            a[j] = "write diary"
        activities.add(a[j])

    newA = ""
    for j in a:
        newA += j + "|"
    raw.iloc[i]['activities'] = newA[:-1]
activities = list(activities)

raw.to_csv('data/preprocessed_data_all.csv', index=False)

# Separando em grupos de treinamento e teste
rand = sorted(random.sample(np.arange(raw.shape[0]).tolist(), k=100))
training = raw.iloc[~raw.index.isin(rand)]
training.to_csv('data/preprocessed_data_training.csv', index=False)
testing = raw.iloc[rand]
testing.to_csv('data/preprocessed_data_testing.csv', index=False)

