import pandas as pd
import numpy as np
import seaborn as sns
import random

sns.set()


# dataset source:
# https://www.kaggle.com/kingabzpro/daylio-mood-tracker


def writeActivities(activities):
    with open("data/activities.txt", 'w') as output:
        activitiesString = ""
        for a in activities:
            activitiesString += a + "|"
        activitiesString = activitiesString[:-1]
        output.write(activitiesString)


def boolean_df(item_lists, unique_items):
    # https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173
    bool_dict = {}
    for i, item in enumerate(unique_items):
        bool_dict[item] = item_lists.apply(lambda x: item in x)
    return pd.DataFrame(bool_dict)


# Leitura do csv
raw = pd.read_csv('data/Daylio_Abid.csv')
# Remoção das colunas que não serão usadas
raw.drop(['full_date', 'date', 'time', 'sub_mood'], axis=1, inplace=True)
# Remoção das linhas com informação vazia
raw.dropna(subset=['weekday', 'activities', 'mood'], inplace=True)

# Determinação da lista de atividades possíveis + reformatação da coluna activities
activities = set()
indexesToDrop = []
for i in np.arange(raw.shape[0]):
    a = raw.iloc[i]['activities'].split("|")
    indexesToDrop = []
    for j in np.arange(len(a)):
        a[j] = a[j].strip()
        if a[j] == "Write dairy":
            a[j] = "write diary"

        if a[j] == "coding" or a[j] == "cooking" or a[j] == "Tutorial":
            a[j] = "coding/cooking/tutorial"

        if a[j] == "Dota 2":
            a[j] = "gaming"
        if a[j] == "hiking":
            a[j] = "walk"
        if a[j] == "weight log" or a[j] == "keto" or a[j] == "trimming":
            a[j] = "diet"
        if a[j] == "language learning":
            a[j] = "learning"
        if a[j] == "Quran":
            a[j] = "prayer"

        if a[j] == "youtube" or a[j] == "streaming":
            indexesToDrop.append(j)
            continue

        activities.add(a[j])

    for j in sorted(indexesToDrop, reverse=True):
        del a[j]
    raw.iloc[i]['activities'] = a

writeActivities(activities)

activitiesDF = boolean_df(raw['activities'], activities)
activitiesCorr = activitiesDF.corr(method="pearson")
activitiesCorr.to_csv('data/activities_correlation.csv')



classCorrelation = activitiesDF.copy(deep=True)
classesColumnList = []
for i in np.arange(raw.shape[0]):
    instance = raw.iloc[i]['mood']
    newInstance = [False, False, False, False, False]
    if instance == 'Amazing':
        newInstance[0] = True
    elif instance == 'Good':
        newInstance[1] = True
    elif instance == 'Normal':
        newInstance[2] = True
    elif instance == 'Bad':
        newInstance[3] = True
    elif instance == 'Awful':
        newInstance[4] = True
    classesColumnList.append(newInstance)
classesColumnDF = pd.DataFrame(classesColumnList, columns=['Amazing', 'Good', 'Normal', 'Bad', 'Awful'])
classCorrelation.reset_index(drop=True, inplace=True)
classCorrelation = pd.concat([classCorrelation, classesColumnDF], axis=1)
classCorrelationPearson = classCorrelation.corr(method="pearson")

intendedCols = ['Amazing', 'Good', 'Normal', 'Bad', 'Awful']
classCorrelationPearson.drop(columns=[col for col in classCorrelationPearson if col not in intendedCols], inplace=True)
classCorrelationPearson.to_csv('data/activitiesToClass_correlation.csv')

countingActivities = []
for a in activities:
    count = 0
    for i in np.arange(raw.shape[0]):
        #print(f"a = {a} i = {i}/{raw.shape[0]}")
        actvs = raw.iloc[i]['activities']
        # actvs = set(actvs.split("|"))
        actvs = set(actvs)
        if a in actvs:
            count += 1
    countingActivities.append([count, a])

df = pd.DataFrame(countingActivities, columns=['occurrences', 'activity'])
df = df.sort_values(by=['occurrences'])
df.to_csv('data/activityoccurrences.csv', index=False)

for i in np.arange(raw.shape[0]):
    a = set(raw.iloc[i]['activities'])
    newA = ""
    for j in a:
        newA += j + "|"
    raw.iloc[i]['activities'] = newA[:-1]

raw.drop(raw.loc[raw['activities']==""].index, inplace=True)
raw.to_csv('data/preprocessed_data_all.csv', index=False)

# Separando em grupos de treinamento e teste
rand = sorted(random.sample(np.arange(raw.shape[0]).tolist(), k=100))
training = raw.iloc[~raw.index.isin(rand)]
training.to_csv('data/preprocessed_data_training.csv', index=False)
testing = raw.iloc[rand]
testing.to_csv('data/preprocessed_data_validation.csv', index=False)
