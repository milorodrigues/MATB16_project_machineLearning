import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set()
results = [
    [1, 18, 82],
    [2, 32, 68],
    [3, 39, 61],
    [4, 38, 62],
    [5, 33, 67],
    [6, 37, 63],
    [7, 40, 60],
    [8, 41, 59],
    [9, 36, 64],
    [10, 36, 64]
]

df = pd.DataFrame(results, columns=['maxHeight', 'accuracy %', 'incorrect'])
plot = sns.relplot(x='maxHeight', y='accuracy %', kind='line', data=df)
plot.set(ylim=(0, 100))
plt.show()
