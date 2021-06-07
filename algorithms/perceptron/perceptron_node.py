import random


class Perceptron:
    def __init__(self, data, eta, threshold):
        self.data = data
        self.eta = eta
        self.threshold = threshold

        self.bias = random.uniform(-0.1, 0.1)
        self.weights = {}
        for a in self.data.activitySet:
            self.weights[a] = random.uniform(-0.1, 0.1)

    def activate(self, instance):
        u = 0
        for a in instance:
            u += instance[a] * self.weights[a]
        u += self.bias
        if u >= self.threshold:
            return 1
        else:
            return -1

    def adjustParams(self, instance, result, expected):
        for w in self.weights:
            self.weights[w] += self.eta * (expected - result) * instance[w]
        self.bias += self.eta * (expected - result)
        return

    def trainInstance(self, rawInstance):
        instance = self.toDict(rawInstance)

        result = self.activate(instance)
        if self.data.classMap[rawInstance['mood']] >= self.threshold:
            expected = 1
        else:
            expected = -1

        if expected != result:
            self.adjustParams(instance, result, expected)
        return {'result': result, 'expected': expected}

    def validateInstance(self, rawInstance):
        instance = self.toDict(rawInstance)

        result = self.activate(instance)
        return result

    def toDict(self, instance):
        d = {}
        activities = self.data.getActivitySet(instance)

        for a in self.data.activitySet:
            if a in activities:
                d[a] = 1
            else:
                d[a] = 0
        return d