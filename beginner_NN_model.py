from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import 
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from random import randint

df = pd.read_csv('nlp_code_dataset.csv')
tokens = {'':0,}
sen = sent_tokenize(df['Text'])
for line in sen:
    words = word_tokenize(line)
    for word in words:
        word = word.lower()
        if word == ',' or word == '.' or word == '!' or word == '?':
            continue
        if word not in tokens:
            tokens[word] = len(tokens+1)#offset by one since 0 will represent null word

class node:
    def __init__(self):
        self.__points_x = np.array([],dtype='int32')
        self.__points_y = np.array([],dtype='int32')
        self.lr = LinearRegression()

    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self,input):
        return relu(self.lr.predict(input))
    
    def train(self):
        self.lr.fit(self.points_x,self.points_y)

    def __append__(self,x,y):
        #x is input
        #y is the output to the next node
        try:
            self.__points_x = np.append(self.__points_x,x)
            self.__points_y = np.append(self.__points_y,y)
        except:
            print(Exception)

class training_node:#used when train a model.difference to regular node is that there is only a single x and y value.use int for simplicity and faster on CPU
    def __init__(self,input):
        self.__input = input
        self.__output = randint(1,2^8)
        self.lr = LinearRegression()
    
    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self):
        return relu(self.lr.predict(self.__input))
    
    def train(self):
        self.lr.fit(self.__input,self.__output)

    def change(self,value):
        if self.__output > 1:
            self.__output += int(value)

class text_model:
    def __init__(self,dimension,token,learning_rate):
        self.__learning_rate = int(learning_rate)
        self.__dim = tuple(dimension)#defines the archetexture of the model
        self.__layers = [None]#stores the layers that contain nodes
        self.__token = token#used to encode text
        self.__token_reverse = ['']#used to encode text
        for text,token_num in self.__token:#index represent the token
            self.__token_reverse.append(text)
        #initalize model
        for num in self.__dim:
            layer = (node()) * num
            self.__layers.append(layer)
        self.__layers = tuple(self.__layers)

    def encode(self,input):#converts text to int
        encoded_text = []
        for word in range(input):
            word = word.lower()
            word = word.strip(",.?!:-_")
            encoded_text.append(self.__token[word])
        return encoded_text
    def decode(self,tokens):
        for t,n in tokens:
            self.__token_reverse.append(t)
    
    def train(self,input,output):
        if isinstance(input,str) == False:raise TypeError("Input is not type: str")
        elif len(input) > self.__dim[0]: raise ValueError("input is too larger")
        elif isinstance(output,str) == False:raise TypeError("output is not type: str")
        else:
            #encode input
            input = word_tokenize(input)
            output = word_tokenize(output)
            input = self.encode(input)
            output = self.encode(output)

            #pad input so that it is same size as input layer
            if len(input) < self.__dim[0]:
                padding = ((0)*(self.__dim[0] - len(input)))
            data = self.encode(input)
            data.extend(padding)
            data_forward = 0#sum of all wieghts
            data_layer = []#stores the the individual data of the prev layer
            training_layers = []

            #initiat with first layer
            layer = []
            for n in range(self.__dim[0]):
                node = training_node(data[n])
                node.train()
                data_forward += node.output()
                layer.append(node)
            training_layers.append(layer)

            #create hidden layer and forward pass data
            for layer_num in range(1,len(self.__dim)-1):
                layer = []
                data_forward_next = 0
                for n in range(self.__dim[layer_num]):
                    node = training_node(data_forward)#forward data to the input
                    node.train()
                    data_forward_next += node.output()
                    layer.append(node)
                training_layers.append(layer)
                data_forward = data_forward_next
            if len(data_forward) != len(self.__dim[-1]): raise Warning(f"size of output is not the same zise as output layer\noutput size-{len(data_forward)}\noutput layer size-{len(self.__dim[-1])}")

            #output layer and forward pass
            for n in range(self.__dim[-1]):
                node = training_node(data_forward)#forward data to the input
                node.train()
                data_layer.append(node.output())
                layer.append(node)
            training_layers.append(layer)

            #calc loss function
            avg_loss = 0
            for n in range(len(self.__dim[-1])):
                avg_loss += output[n] - data_layer[n]
            avg_loss = avg_loss / len(self.__dim[-1])

            #forward pass training
            data_forward_training = 0
            cache = 0#stores the sum of previous layer
            direction = 1#if positive node is increased and visa versa
            for layer_num in range(1,len(self.__dim)):
                for node_num in range(len(self.__dim[layer_num])):
                    change_amount = avg_loss * self.__learning_rate * direction
                    node = training_layers[layer_num][node_num]
                    node.change(change_amount)
                    #calc change
                    for n in range(len(self.__dim[0])):
                        training_layers[0][n].train()
                        cache += training_layers[0][n].output()
                    for l in range(layer_num,len(self.__dim)-1):
                        data_forward_next = 0
                        for n in range(len(self.__dim[l])):
                            training_layers[l][n].train()
                            data_forward_next += training_layers[l][n].output()
                        data_forward_training = data_forward_next
                    data_layer_training = []
                    for n in range(len(self.__dim[-1])):
                        training_layers[-1][n].train()
                        data_layer_training.append(training_layers[-1][n].output())
                    avg_loss_next = 0
                    for n in range(len(self.__dim[-1])):
                        avg_loss_training += output[n] - data_layer_training[n]
                    avg_loss_next = avg_loss_next / len(self.__dim[-1])
                    if avg_loss > avg_loss_next:
                        direction = 1
                    else:direction = -1
                        



    def store_model():
        with open('custom_nn.py','w') as model:
            #store model as an python script that is executed via terminal.
            #the script take a string arg as an input
            #encode the string inpt arg
            #store the wieghts and biases
            #decode output to string
            #all output values of nodes must pass through relu
            pass

def relu(input):
    if input <= 0:
        return 0
    else: return input