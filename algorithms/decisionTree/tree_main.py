import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dataModel
import tree_model
import tree_entropy

sns.set()
model = tree_model.Model(8)
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
dChosen = {
            'Amazing': 0,
            'Good': 0,
            'Normal': 0,
            'Bad': 0,
            'Awful': 0
        }
dExpected = {
        'Amazing': 0,
        'Good': 0,
        'Normal': 0,
        'Bad': 0,
        'Awful': 0
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
    dChosen[result] += 1
    dExpected[expected] += 1
print(analysis)
print(results)
print(f"dChosen = {dChosen}")
print(f"dExpected = {dExpected}")
