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
        self.points_x = np.array([])
        self.points_y = np.array([])
        self.lr = LinearRegression()

    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self,input):
        return self.lr.predict(input)
    
    def train(self):
        self.lr.fit(self.points_x,self.points_y)

    def __append__(self,x,y):
        try:
            self.points_x = np.append(self.points_x,x)
            self.points_y = np.append(self.points_y,y)
        except:
            print(Exception)

class text_model:
    def __init__(self,dimension,token):
        self.__dim = tuple(dimension)
        self.__layers = []
        self.__token = token
        for num in self.__dim:
            layer = []
            for _ in range(num):
                layer.append(node())
            self.__layers.append(layer)
        self.__layers = tuple(self.__layers)

    def decode(self,input):
        decoded_text = []
        for word in range(input):
            decoded_text.append(tokens[word])
        return decoded_text



    def train(self,input,output):
        if isinstance(input,tuple) == False:raise TypeError("Input is not type: tuple")
        elif isinstance(output,tuple) == False:raise TypeError("output is not type: tuple")
        else:
            for layer_num in range(len(self.__layers)-1):
                for node in self.__layers[layer_num]:
                    for node in self.__layers[layer_num+1]:
                    

    def store_model():
        with open('custom_nn.py','w') as model:
            model.write()