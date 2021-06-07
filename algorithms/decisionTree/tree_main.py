import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dataModel
import tree_model
import tree_entropy

sns.set()
model = tree_model.Model(3)
model.root.printRecursive()


def decisionTree(instance, node):
    while node.isLeaf is False:
        currentAttribute = node.label
        currentValue = False
        if currentAttribute in instance['activities']:
            currentValue = True
        try:
            node = (next(child for child in node.children if child['edge'] == currentValue))['child']
        except:
            return 'Failure'
    return node.label


analysis = {
    'correct': 0,
    'incorrect': 0,
    'failed': 0
}
results = []
for i in np.arange(model.data.validation.shape[0]):
    print(f"Validating {i+1} out of {model.data.validation.shape[0]}")
    instance = model.data.validation.iloc[i]
    expected = instance['mood']
    result = decisionTree(instance, model.root)
    results.append([result, expected])
    if result == "Failure":
        analysis['failed'] += 1
    elif expected == result:
        analysis['correct'] += 1
    else:
        analysis['incorrect'] += 1
print(analysis)
print(results)



# e = tree_entropy.entropy2({'Amazing': 116, 'Good': 303, 'Normal': 110, 'Bad': 31, 'Awful': 35}, ['Amazing', 'Good', 'Normal', 'Bad', 'Awful'])
# print(e)

# print(model.data.training.iloc[84])
# test = model.data.training.iloc[84]
# print(test['activities'])
# print('youtube' in test['activities'])
# print('prayer' in test['activities'])

# print(model.data.validation.iloc[15])
# test = model.data.validation.iloc[15]
# print(test['activities'])
# print('youtube' in test['activities'])
# print('language learning' in test['activities'])
