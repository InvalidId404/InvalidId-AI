import neuralnetwork
import os

os.chdir(os.getcwd()+'/test_data')
test_file = open('mnist_train_100.csv')

file = test_file.readlines()
training_data = []

for record in file:
    record = record.split(',')
    training_data.append([int(record[0]), [int(i)/256 + 0.01 for i in record[1:]]])
    training_data[-1][0] = [int(training_data[-1][0] is i)*0.98+0.01 for i in range(10)]

nodes = [
    len(training_data[0][1]),
    100,
    10
]
AI = neuralnetwork.FeedFowardNeuralNetwork(nodes)
AI.dataset = training_data
AI.weight = AI.train(
    epoch=1,
    learning_rate=0.0001,
)
print(training_data[0][0])
print(AI.query(training_data[0][1])[-1])