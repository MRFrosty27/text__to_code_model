from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize

df = pd.read_csv('nlp_code_dataset.csv')
tokens = {}
sen = sent_tokenize(df['Text'])
for line in sen:
    words = word_tokenize(line)
    for word in words:
        word.lower()
        if word not in tokens:
            tokens[word] = len(tokens)

class node:
    def __init__(self):
        self.__points_x = np.array([])
        self.__points_y = np.array([])
        self.lr = LinearRegression()

    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self,input):
        return self.lr.predict(input)
    
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

class training_node:#used when train a model.difference to regular node is that there is only a single x and y value.
    def __init__(self,input):
        self.__input = input
        self.__output = 1
        self.lr = LinearRegression()
    
    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self,input):
        return self.lr.predict(input)
    
    def train(self):
        self.lr.fit(self.points_x,self.points_y)

class text_model:
    def __init__(self,dimension,token):
        self.__dim = tuple(dimension)
        self.__layers = []
        self.__token = token
        for num in self.__dim:
            layer = (node()) * num
            self.__layers.append(layer)
        self.__layers = tuple(self.__layers)

    def encode(self,input):
        encoded_text = []
        for word in range(input):
            encoded_text.append(tokens[word])
        return encoded_text
    
    def train_forward(self,input,output):
        if isinstance(input,tuple) == False:raise TypeError("Input is not type: tuple")
        elif len(input) > self.__dim[0]: raise ValueError("input is too larger")
        elif isinstance(output,tuple) == False:raise TypeError("output is not type: tuple")
        else:
            if len(input) < self.__dim[0]:
                padding = ((0)*(self.__dim[0] - len(input)))
            data = input
            data.extend(padding)
            data_forward = []
            for node_num in range(self.__dim[0]):
                node = self.__layers[0][node_num]
                node.append(data[node_num],1)
                node.train()
                data_forward.append(node.output)

            for layer_num in range(1,len(self.__layers)):
                for node in self.__layers[layer_num]:
                    node_value = node.output()
                    for node in self.__layers[layer_num-1]:
                        node_value += 

    def store_model():
        with open('custom_nn.py','w') as model:
            model.write()