# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:09:01 2020

@author: joshu
"""
import numpy
import csv
import scipy.special
import imageio
import glob
import matplotlib.pyplot
import pickle


f = open("mnist_train.csv","r")
csv_reader = csv.reader(f)
train_inputs_list=[]
a=0
for row in csv_reader:
    train_inputs_list.append(row)
    train_inputs_list[a].pop(0)
    a=a+1

f.close()
for i in list(range(0,len(train_inputs_list))):
    for j in list(range(0,len(train_inputs_list[0]))):
        train_inputs_list[i][j]=float(train_inputs_list[i][j])
        train_inputs_list[i][j]=((train_inputs_list[i][j]/255)*0.99) + 0.01

f=open("mnist_train_targets_formatted.csv","r")
csv_reader2=csv.reader(f)
train_targets_list = []
for row in csv_reader2:
    train_targets_list.append(row)
f.close()    
for i in list(range(0,len(train_targets_list))):
    for j in list(range(0,len(train_targets_list[0]))):
        train_targets_list[i][j]=float(train_targets_list[i][j])
        train_targets_list[i][j]=(train_targets_list[i][j]*0.98)+0.01
        
        
class NeuralNetwork():
    
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learningrate):
        self.inodes=input_nodes
        self.hnodes=hidden_nodes
        self.onodes=output_nodes
        
        self.activation_function = lambda x: scipy.special.expit(x)
        
        self.lr=learningrate
        
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
    
        

    
    def train(self, inputs_list,targets_list):
        #Convert inputs and targets into a 1D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #Calculate the output from an individual data
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        
        #Backpropagate the error
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),numpy.transpose(inputs))
        
        
        
        
    
    def query(self,inputs_list):
        #convert input into a 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #Send signal through NN to calculate output
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    
    def reversepropagation(self,outputs_list):
        outputlayer_inputs=scipy.special.logit(outputs_list)
        hidden_outputs=numpy.dot(((self.who**(-1))),outputlayer_inputs)
        hidden_inputs=scipy.special.logit(hidden_outputs)
        inputlayer_outputs=numpy.dot(self.wih**(-1),hidden_inputs)
        return inputlayer_outputs
        
    
    
    
    


f = open("mnist_test.csv","r")
csv_reader = csv.reader(f)
test_inputs_list=[]
a=0
for row in csv_reader:
    test_inputs_list.append(row)
    test_inputs_list[a].pop(0)
    a=a+1

f.close()
for i in list(range(0,len(test_inputs_list))):
    for j in list(range(0,len(test_inputs_list[0]))):
        test_inputs_list[i][j]=float(test_inputs_list[i][j])
        test_inputs_list[i][j]=((test_inputs_list[i][j]/255)*0.99)+0.01

f=open("mnist_test_targets_formatted.csv","r")
csv_reader2=csv.reader(f)
test_targets_list = []
for row in csv_reader2:
    test_targets_list.append(row)
f.close()    
for i in list(range(0,len(test_targets_list))):
    for j in list(range(0,len(test_targets_list[0]))):
        test_targets_list[i][j]=float(test_targets_list[i][j])
        test_targets_list[i][j]=(test_targets_list[i][j]*0.98)+0.01
        
        
        
        
input_nodes=len(train_inputs_list[1])
hidden_nodes=500
output_nodes=len(train_targets_list[1])
learning_rate=0.3
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

for i in list(range(0,len(train_inputs_list))):
    print(i/len(train_inputs_list))
    n.train(train_inputs_list[i],train_targets_list[i])
    





#Check success rate
def convert_output(NN_output):
    for i in list(range(0,len(NN_output))):
        if NN_output[i]==max(NN_output):
            return i
        
        
a=0
for i in list(range(0,len(test_inputs_list))):
    x=convert_output(n.query(test_inputs_list[i]))
    if x==convert_output(test_targets_list[i]):
        a=a+1
success=a/len(test_inputs_list)
print(success)





# our own image test data set
our_own_dataset = []
for image_file_name in glob.glob('3.png'):
    print ("loading ... ", image_file_name)
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    # load image data from png files into an array
    img_array = imageio.imread(image_file_name, as_gray=True)
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    pass
img_data = (img_data/255)*0.98+0.01
matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')
img_data = list(img_data)
convert_output(n.query(img_data))






"""
TrialReverse=numpy.array([[0.01],[0.01],[0.99],[0.01],[0.01],[0.01],[0.01],[0.01],[0.01],[0.01]])
outputlayer_inputs=scipy.special.logit(TrialReverse)
hiddenlayer_outputs=numpy.dot(n.who.T,outputlayer_inputs)
hiddenlayer_outputs -= numpy.min(hiddenlayer_outputs)
hiddenlayer_outputs /= numpy.max(hiddenlayer_outputs)
hiddenlayer_outputs *= 0.98
hiddenlayer_outputs += 0.01
hiddenlayer_inputs=scipy.special.logit(hiddenlayer_outputs) 
inputlayer_outputs=numpy.dot(n.wih.T,hiddenlayer_inputs)
inputlayer_outputs -= numpy.min(inputlayer_outputs)
inputlayer_outputs /= numpy.max(inputlayer_outputs)
inputlayer_outputs *= 0.98
inputlayer_outputs += 0.01
matplotlib.pyplot.imshow(inputlayer_outputs.reshape(28,28), cmap='Greys', interpolation='None')
"""

"""
for i in list(range(0,100)):
    if convert_output(n.query(test_inputs_list[i]))!=convert_output(test_targets_list[i]):
        print("index {} produces: code: {} and actual:  {}".format(i, convert_output(n.query(test_inputs_list[i])),convert_output(test_targets_list[i])))
        matplotlib.pyplot.imshow(numpy.array(test_inputs_list[i]).reshape(28,28), cmap='Greys', interpolation='None')
"""