import pdb
import os
from collections import deque
import numpy as np

class Counter:
    def __init__(self):
        # start at weakly taken
        self.state = 3

    def predict(self):
        # if in a not taken state, return not taken
        if self.state < 3:
            return 0
        # return taken
        else:
            return 1

    def update(self, actual):
        # if branch was taken, increment state
        if actual == 1:
            if self.state != 4:
                self.state += 1
        # branch not taken, decrement state
        else:
            if self.state != 1:
                self.state -= 1

def saturatingCounter(trace, l):
    # variable for the counter for each branch
    counter_list = {}
    # variable for counting correct predictions
    correct = 0

    # for all the branches
    for branch in trace:
        # if the branch does not have a counter, add one to the list
        if branch[0] not in counter_list:
            counter_list[branch[0]] = Counter()
        
        # make prediction
        prediction = counter_list[branch[0]].predict()
        # update the counter using whether or not the branch was actually taken or not
        counter_list[branch[0]].update(branch[1])

        # if we match, count as correct prediction
        if prediction == branch[1]:
            correct += 1

    return correct

class Perceptron:
    def __init__(self, N):
        # number of inputs
        self.N = N
        self.bias = 0
        # threshold that helps dictates when to update
        self.threshold = 2 * N + 14
        # initialization of weights for the inputs
        self.weights = [0] * N      

    def predict(self, global_branch_history):
        output = self.bias

        # sum up output of the neuron
        for i in range(0, self.N):
            output += global_branch_history[i] * self.weights[i]
        
        # if less than 0, predict not taken
        if output < 0:
            prediction = -1 
        # predict taken
        else:
            prediction = 1
        
        return prediction, output

    def update(self, prediction, actual, global_branch_history, output):
        if (prediction != actual) or (abs(output) < self.threshold):   
            self.bias += actual
            
            for i in range(0, self.N):
                self.weights[i] += actual * global_branch_history[i]

def perceptronPredictor(trace, l):
    # list for branch history
    global_branch_history = deque([])
    global_branch_history.extend([0]*l)

    # list of perceptrons for each branch
    perceptron_list = {}
    # variable for counting correct predictions
    correct = 0

    # for all the branches
    for branch in trace:
        # if no previous branch from this memory location
        if branch[0] not in perceptron_list:  
            perceptron_list[branch[0]] = Perceptron(l)
        
        # predict taken/not taken and get output of the perceptron
        prediction, output = perceptron_list[branch[0]].predict(global_branch_history)
        
        # get whether taken or not taken
        if branch[1] == 1:
            actual = 1
        else:
            actual = -1
        
        # update the perceptron
        perceptron_list[branch[0]].update(prediction, actual, global_branch_history, output)
        # add the result to the branch history
        global_branch_history.appendleft(actual)
        # delete the oldest result
        global_branch_history.pop()

        # increment if correct
        if prediction == actual:
            correct += 1

    return correct

class hybridCounterPerceptron:
    def __init__(self, N):
        self.p = Perceptron(N)
        self.state = 3

    def predict(self, global_branch_history):
        if self.state == 2:
            prediction, output = self.p.predict(global_branch_history)

            return prediction, output
        elif self.state < 2:
            return -1, 0
        else:
            return 1, 0
    
    def update(self, actual, prediction, global_branch_history, output):
        if self.state == 2:
            self.p.update(prediction, actual, global_branch_history, output)
        # self.p.update(prediction, actual, global_branch_history, output)

        if actual == 1:
            if self.state != 3:
                self.state += 1
        else:
            if self.state != 1:
                self.state -= 1

def hybridPredictor(trace, l):
    global_branch_history = deque([])
    global_branch_history.extend([0]*l)

    hybrid_list = {}
    correct = 0
    
    for branch in trace:
        if branch[0] not in hybrid_list:
            hybrid_list[branch[0]] = hybridCounterPerceptron(l)

        prediction, output = hybrid_list[branch[0]].predict(global_branch_history)
        
        if branch[1] == 1:
            actual = 1
        else:
            actual = -1

        if hybrid_list[branch[0]].state == 3:
            global_branch_history.appendleft(actual)

        # global_branch_history.appendleft(actual)
        hybrid_list[branch[0]].update(actual, prediction, global_branch_history, output)

        if prediction == actual:
            correct += 1

    return correct

def simulation(predictor, file, **kwargs):
    trace = {}
    branches = []

    with open(file, 'r') as file_in:
        for line in file_in:
            register = line[2:8]
            result = int(line[9])
            trace.setdefault(register, []).append(result)
            branches.append([register, result])

    correct = predictor(branches, l=kwargs['l'])
    total = sum(len(r) for r in trace.values())

    return correct / total

gcc = 'gcc_branch.out'
mcf = 'mcf_branch.out'
print("|Predictor|         |gcc accuracy|         |mcf accuracy|")

nn_gcc = simulation(saturatingCounter, file=gcc, l=16)
nn_mcf = simulation(saturatingCounter, file=mcf, l=16)
print("Saturating counter     %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=1)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=1)
print("Perceptron (depth 1)   %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=2)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=2)
print("Perceptron (depth 2)   %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=4)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=4)
print("Perceptron (depth 4)   %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=8)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=8)
print("Perceptron (depth 8)   %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=16)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=16)
print("Perceptron (depth 16)  %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(perceptronPredictor, file=gcc, l=32)
nn_mcf = simulation(perceptronPredictor, file=mcf, l=32)
print("Perceptron (depth 32)  %.5f                %.5f" % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=1)
nn_mcf = simulation(hybridPredictor, file=mcf, l=1)
print('Hybrid (depth 1)       %.5f                %.5f' % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=2)
nn_mcf = simulation(hybridPredictor, file=mcf, l=2)
print('Hybrid (depth 2)       %.5f                %.5f' % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=4)
nn_mcf = simulation(hybridPredictor, file=mcf, l=4)
print('Hybrid (depth 4)       %.5f                %.5f' % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=8)
nn_mcf = simulation(hybridPredictor, file=mcf, l=8)
print('Hybrid (depth 8)       %.5f                %.5f' % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=16)
nn_mcf = simulation(hybridPredictor, file=mcf, l=16)
print('Hybrid (depth 16)      %.5f                %.5f' % (nn_gcc, nn_mcf))

nn_gcc = simulation(hybridPredictor, file=gcc, l=32)
nn_mcf = simulation(hybridPredictor, file=mcf, l=32)
print('Hybrid (depth 32)      %.5f                %.5f' % (nn_gcc, nn_mcf))