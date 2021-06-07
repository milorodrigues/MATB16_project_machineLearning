import dataModel
import pandas as pd
import numpy as np
from algorithms.decisionTree import tree_entropy


class Model:
    def __init__(self, maxDepth):
        self.data = dataModel.Data('../../data/preprocessed_data_training.csv',
                                   '../../data/preprocessed_data_validation.csv',
                                   '../../data/activities.txt')
        self.data.convertActivitiesToList()

        self.maxDepth = maxDepth

        self.classLabel = 'mood'
        self.classes = ['Amazing', 'Good', 'Normal', 'Bad', 'Awful']
        self.classCounts = self.data.training[self.classLabel].value_counts()
        self.nodeCount = 0

        self.root = self.createRoot()

    def createRoot(self):
        activities = self.data.activitySet
        print(f"[createRoot]")

        infoGains = {}
        #infoGains['hiking'] = tree_entropy.infoGain(self.data.training, 'hiking', self.classLabel, self.classes)
        for a in activities:
            infoGains[a] = tree_entropy.infoGain(self.data.training, a, self.classLabel, self.classes)
        chosen = max(infoGains, key=lambda key: infoGains[key])

        treeRoot = Node(None, chosen, self.nodeCount, 0)
        self.nodeCount += 1

        cols = self.data.activitySet.copy()
        cols.remove(chosen)

        self.createModel(self.data.training, treeRoot, cols)
        return treeRoot

    def createModel(self, dataset, parent, cols):
        attribute = parent.label
        currentDepth = parent.depth
        print(f"[createModel] parent: {parent.label} {parent.id} depth: {currentDepth}")

        if currentDepth == self.maxDepth:
            print(f"forcing leaf")
            values = dataset[self.classLabel].value_counts().to_dict()
            adjustedValues = {
                'Amazing': (values.setdefault('Amazing', 0) / len(values)) / (self.classCounts['Amazing'] / self.data.training.shape[0]),
                'Good': (values.setdefault('Good', 0) / len(values)) / (self.classCounts['Good'] / self.data.training.shape[0]),
                'Normal': (values.setdefault('Normal', 0) / len(values)) / (self.classCounts['Normal'] / self.data.training.shape[0]),
                'Bad': (values.setdefault('Bad', 0) / len(values)) / (self.classCounts['Bad'] / self.data.training.shape[0]),
                'Awful': (values.setdefault('Awful', 0) / len(values)) / (self.classCounts['Awful'] / self.data.training.shape[0])
            }
            chosenValue = max(adjustedValues, key=lambda key: adjustedValues[key])
            node = Node(parent, chosenValue, self.nodeCount, currentDepth+1, True)
            self.nodeCount += 1
            parent.addChild(node, True)
            parent.addChild(node, False)
            return

        values = [True, False]
        for v in values:
            newDataset = []
            for i in np.arange(dataset.shape[0]):
                if (attribute in dataset.iloc[i]['activities']) == v:
                    newDataset.append([
                        dataset.iloc[i]['weekday'],
                        dataset.iloc[i]['activities'],
                        dataset.iloc[i]['mood'],
                    ])
            dataset2 = pd.DataFrame(newDataset, columns=['weekday', 'activities', 'mood'])

            count = dataset2[self.classLabel].value_counts()
            #print(f"{v} {count}")
            if count.shape[0] == 1:
                print(f"[createModel] creating leaf from {parent.label} {parent.id} => {v} => {count.index[0]}")
                node = Node(parent, count.index[0], self.nodeCount, currentDepth+1, True)
                self.nodeCount += 1
                parent.addChild(node, v)
                continue

            if len(cols) == 0:
                return

            infoGains = {}
            for c in cols:
                #print(f"[createModel] {parent.label} {parent.id} c = {c}")
                infoGains[c] = tree_entropy.infoGain(self.data.training, c, self.classLabel, self.classes)
            chosen = max(infoGains, key=lambda key: infoGains[key])

            newCols = cols.copy()
            newCols.remove(chosen)

            node = Node(parent, chosen, self.nodeCount, currentDepth+1)
            self.nodeCount += 1
            parent.addChild(node, v)

            self.createModel(dataset2, node, newCols)
        return


class Node:
    def __init__(self, parent, label, id, depth, isLeaf=False):
        self.parent = parent
        self.label = label
        self.children = []
        self.id = id
        self.depth = depth
        self.isLeaf = isLeaf

    def addChild(self, child, edge):
        self.children.append({
            'child': child,
            'edge': edge
        })

    def print(self):
        if self.parent is None:
            print(f"{self.label} {self.id} (root) {self.childrenString()}")
        else:
            print(f"{self.label} {self.id} ({self.parent.label} {self.parent.id}) {self.childrenString()}")

    def childrenString(self):
        string = ""
        if self.isLeaf:
            return string
        for c in self.children:
            string += f"[{c['edge']}=>{c['child'].label} {c['child'].id}] "
        return string

    def printRecursive(self):
        self.print()
        for c in self.children:
            c['child'].printRecursive()