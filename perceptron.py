# PSYC489 Final Project Professor Hummel, J.
# Author: Yuzuki Ishikawa

import math
import random
import pandas as pd
import numpy as np
from random import randrange

# hyperparameters
MOMENTUM = 0.90
LRATE = 0.1
SETTLING_CRITERION = 0.01
INPUT_FIELD = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# Training data
# [[Data1, Data2, Data3,...], [Data1, Data2, Data3,...],...] 
theta_mini = pd.DataFrame(pd.read_excel(r'Eyes-closed EEG.xlsx', sheet_name=r'minimal_theta')).to_numpy()[:,2:16]
theta_mode = pd.DataFrame(pd.read_excel(r'Eyes-closed EEG.xlsx', sheet_name=r'moderate_theta')).to_numpy()[:,2:16]

# apply population coding
def population_coding(data):
    output = []
    for i in range(len(data)):
        output.append([])
        for j in range(len(data[i])):
            tempList = [0,0,0,0,0,0,0]
            closest = min(INPUT_FIELD, key=lambda x:abs(x-data[i][j]))
            if data[i][j] - closest < 0:
                lower = abs(1-(data[i][j]-(closest-1)))
                higher = 1-lower
                tempList[INPUT_FIELD.index(closest-1)] = lower 
                tempList[INPUT_FIELD.index(closest)] = higher
            else:
                lower = abs(1-(data[i][j]-closest))
                higher = 1-lower
                tempList[INPUT_FIELD.index(closest)] = lower 
                tempList[INPUT_FIELD.index(closest+1)] = higher
            
            output[i] += tempList
            
    return np.array(output)

# final form of inputs
mini = population_coding(theta_mini)
mode = population_coding(theta_mode)

# Major object classes start here
class Connection(object):
    def __init__(self, sender, recipient, range = 2.0):
        self.sender         = sender
        self.recipient      = recipient
        self.weight         = range * (random.random() - 0.5) # random weight , -1.0 ~ 1.0
        self.delta_weight   = 0.0
            
class Node(object):
    def __init__(self, owner, isa_bias_node = False):
        self.net_input          = 0.0
        self.activation         = 0.0
        self.desired_activation = 0.0 # output nodes only
        self.abs_error          = 0.0 # error before derivative correction
        self.error              = 0.0 # error for learning: includes derivative correction
        self.isa_bias_node      = isa_bias_node # If I'm bias node then my act always = 1.0
        self.layer              = owner
        self.incoming_connections = []
        self.outgoing_connections = []
        
    def add_connection(self,sender):
        new_conn = Connection(sender,self)
        self.incoming_connections.append(new_conn)
        sender.outgoing_connections.append(new_conn)
        
    def update_input(self):
        self.net_input = 0
        for conn in self.incoming_connections:
            self.net_input += conn.weight * conn.sender.activation
            
    def update_activation(self):
        # Sigmoid
        if self.net_input >= 0:
          self.activation = 1/(1+math.exp(-self.net_input))
        else:
          self.activation = math.exp(self.net_input)/(1+math.exp(self.net_input))
    
    def update_error(self):
        if self.layer.is_output_layer:
            self.abs_error = self.desired_activation - self.activation
        else:
            self.abs_error = 0.0
            for conn in self.outgoing_connections:
                self.abs_error += conn.weight * conn.recipient.error
        
        # Sigmoid
        self.error = self.abs_error * (1.0 - self.activation) * self.activation

class Layer(object):
    def __init__(self, num_nodes, is_output_layer):
        self.nodes           = []
        self.is_output_layer = is_output_layer
        
        self.nodes = [Node(self) for i in range(num_nodes)]
        # append a bias node
        if is_output_layer is False:
          self.nodes.append(Node(self, isa_bias_node=True))

class Network(object):
    def __init__(self, layer_list):
        self.layers = []
        self.global_error = 0.0
        
        # construct the layers of nodes based on the values in layer_list
        for i in range(len(layer_list)):
            if i == (len(layer_list) - 1):
                # this is the last layer
                is_output_layer = True
            else:
                is_output_layer = False
            self.layers.append(Layer(layer_list[i], is_output_layer)) 
        
        # create the connections between the layers
        for lower_layer_index in range(len(layer_list)-1):
            upper_layer_index = lower_layer_index + 1
            for lower_node in self.layers[lower_layer_index].nodes:
                for upper_node in self.layers[upper_layer_index].nodes:
                    if not upper_node.isa_bias_node:
                        upper_node.add_connection(lower_node)
    
    def forward_prop(self, input_pattern):
        for i in range(len(input_pattern)):
            self.layers[0].nodes[i].activation = input_pattern[i]
        self.layers[0].nodes[-1].activation = 1.0
        
        for i in range(1,len(self.layers)):
            for node in self.layers[i].nodes:
                if node.isa_bias_node:
                    node.activation = 1.0
                else:
                    node.update_input()
                    node.update_activation()
        
        output_activation = [node.activation for node in self.layers[-1].nodes]
        return output_activation
                    
    def backward_prop(self, desired_output):
        for i in range(len(desired_output)):
            self.layers[-1].nodes[i].desired_activation = desired_output[i]
        
        layer_index = len(self.layers)-1    
        while layer_index > 0:
            for node in self.layers[layer_index].nodes:
                if node.isa_bias_node:
                    node.abs_error = node.error = 0.0
                else:
                    node.update_error()
                    for conn in node.incoming_connections:
                        this_weight_change = LRATE * node.error * conn.sender.activation
                        conn.delta_weight += this_weight_change
            layer_index -= 1
        
        # update global error
        pattern_error = 0.0               
        for node in self.layers[-1].nodes:
            pattern_error += node.abs_error**2
        pattern_error /= len(self.layers[-1].nodes)
        return pattern_error
        
    def update_weights(self):
        for layer_index in range(1, len(self.layers)):
            for node in self.layers[layer_index].nodes:
                if not node.isa_bias_node:
                    for conn in node.incoming_connections:
                        conn.weight += conn.delta_weight
                        conn.delta_weight *= MOMENTUM
    
    def train(self, training_set):
        # train the whole network until it settles
        # self.global_error < settling_criterion or num_epochs > 1000
        self.global_error = 100000  # just to pass while condition   
        epoch_index = 0
        
        # fp = open("energy_0.1_0.001.txt","a")
        i = 0
        while self.global_error > SETTLING_CRITERION and epoch_index < 1000:
            epoch_index += 1
            self.global_error = 0
            for input_pattern in training_set:
                self.forward_prop(input_pattern[0])
                self.global_error += self.backward_prop(input_pattern[1])
                
            # back_propagate method also updates self.global_error
            self.global_error /= len(training_set)
            # fp.write(str(self.global_error) + "\n")
            self.update_weights()
            i += 1
            print(str(i) + " iterations: " + str(self.global_error))
          
        # fp.close()
        return self, epoch_index
            

# some functions needed for the actural run           
def create_network(layer_list, train):
    network = Network(layer_list)
    network, epochs = network.train(train)
    
    return network, epochs

def do_test(network, test):
    results = []
    for i in range(len(test)):
        results.append([])
        result = network.forward_prop(test[i])
        results[i] = [round(value,2) for value in result]
        results[i] = tuple(results[i])
        
    return tuple(results)

def make_training_input(folds, label):
    inputs = []
    for i in range(len(folds)):
      inputs.append(())
      if label == 0:
          inputs[i] = folds[i],[1,0]
      else:
          inputs[i] = folds[i],[0,1]
    return inputs

# data are divided into training and validation sets
# using the stratified 5-fold cross validation
# n = 30 (minimal) / 28 (moderate) 

mini3 = np.concatenate((mini[12:30],mini[0:6]), axis = 0)
mini4 = np.concatenate((mini[18:30],mini[0:12]), axis = 0)
mini5 = np.concatenate((mini[24:30],mini[0:18]), axis = 0)
mode3 = np.concatenate((mode[12:28],mode[0:6]), axis = 0)
mode4 = np.concatenate((mode[18:28],mode[0:12]), axis = 0)
mode5 = np.concatenate((mode[23:28],mode[0:18]), axis = 0)

mini_input1 = make_training_input(mini[0:24], label=0)
mode_input1 = make_training_input(mode[0:23], label=1)
mini_input2 = make_training_input(mini[6:30], label=0)
mode_input2 = make_training_input(mode[6:28], label=1)                             
mini_input3 = make_training_input(mini3, label=0)
mode_input3 = make_training_input(mode3, label=1)
mini_input4 = make_training_input(mini4, label=0)
mode_input4 = make_training_input(mode4, label=1)
mini_input5 = make_training_input(mini5, label=0)
mode_input5 = make_training_input(mode5, label=1)

input1 = mini_input1 + mode_input1
input2 = mini_input2 + mode_input2
input3 = mini_input3 + mode_input3
input4 = mini_input4 + mode_input4
input5 = mini_input5 + mode_input5
 
# network training: 4 folds / test: 1 fold
network, epochs = create_network([98,50,2], input1)

print(do_test(network, mini[24:30]))
print(do_test(network, mode[23:28]))