import numpy as np
from scipy.special import expit


class NeuralNetwork:
    def __init__(self, nodes):
        self.nodes = nodes
        self.classes = len(nodes)
        self.weight = [
            np.random.normal(
                0.0, self.nodes[i]**(-1/2), (self.nodes[i+1], self.nodes[i])
            ) for i in range(self.classes-1)
        ]
        self.activate_function = lambda x: expit(x)
        self.training_data = []

    def query(self, input_data):
        outputs = []
        outputs.append(
            np.array([input_data]).T
        )
        for i in range(self.classes-1):
            outputs.append(
                self.activate_function(self.weight[i]@outputs[i])
            )
        return outputs

    def train(self, epoch=5, learning_rate = 0.01):
        for _ in range(epoch):
            for data in self.training_data:
                errors = []
                outputs = self.query(data[1])
                target = np.array([data[0]]).T
                for i in range(self.classes-1):
                    if i is 0:
                        errors.insert(0, target-outputs[-1])
                    else:
                        errors.insert(0, self.weight[-i].T@errors[0])
                self.weight[-i-1] += -learning_rate*(
                    (errors[0]*self.activate_function(outputs[-i-1])*(1-self.activate_function(outputs[-i-1])))@outputs[-i-2].T
                )
        return 1


