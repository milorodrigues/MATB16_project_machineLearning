import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import perceptron_binary as pBin
import perceptron_5classes as p5

sns.set()
eta = 0.3
epochs = 500

networkBin = pBin.Network(eta, epochs, 2)
analysisBin = networkBin.analyze()
analysisBin['algorithm'] = 'binary'

network5 = p5.Network(eta, epochs)
analysis5 = network5.analyze()
analysis5['algorithm'] = '5 categories'

df = pd.concat([analysisBin, analysis5]).reset_index(drop=True)

plot = sns.relplot(x='epoch', y='classified correctly', hue='algorithm', palette=['red', 'blue'], kind='line', data=df)
plot.set(ylim=(0, 100))
plt.show()