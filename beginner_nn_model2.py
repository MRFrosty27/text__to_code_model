from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import 
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from random import randint

df = pd.read_csv('ML\dataset.csv',sep=';',engine='python',dtype=str)
df['Text'] = df['Text'].astype(str)
df['Code'] = df['Code'].astype(str)
def tokenize(token_dict,dataframe):
    if isinstance(token_dict,dict) == False:raise TypeError('First arg is not type dict')
    elif isinstance(dataframe,pd.DataFrame) == False: TypeError('second arg is not type dataframe')
    elif 'Text' not in dataframe.columns: raise NameError('column Text was not found')
    elif len(token_dict) == 0:
        token_dict = {'':0,'(':1,')':2,'    ':3,':':4,'[':5,']':6}
    for t in df['Text']:
        sen = sent_tokenize(t)
        for line in sen:
            words = word_tokenize(line)
            for word in words:
                word = word.lower()
                if word == ',' or word == '.' or word == '!' or word == '?':
                    continue
                if word not in token_dict:
                    token_dict[word] = len(token_dict)
    for t in df['Code']:
        sen = sent_tokenize(t)
        for line in sen:
            words = word_tokenize(line)
            for word in words:
                word = word.lower()
                if word == ',' or word == '.' or word == '!' or word == '?':
                    continue
                if word not in token_dict:
                    token_dict[word] = len(token_dict)
    return token_dict

class node:
    def __init__(self):
        self.__wieght = np.array([],dtype='int32')
        self.__bias = np.array([],dtype='int32')

    def get_state(self):
        return self.__wieght,self.__bias
    
    def output(self,input):
        return relu(self.__wieght*input+self.__bias)
    
    def train(self):
        self.lr.fit(self.points_x,self.points_y)

    def change_wieght(self,value):
        self.__wieght = np.add(self.__wieght,value)

    def change_bias(self,value):
        self.__bias = np.add(self.__bias,value)

class text_model:
    def __init__(self,dimension,token,learning_rate,epoch):
        self.__learning_rate = float(learning_rate)
        self.__epoch = epoch
        self.__dim = tuple(dimension)#defines the archetexture of the model
        self.__layers = [None]#stores the layers that contain nodes
        self.__token = token#used to encode text
        self.__token_reverse = ['']#used to decode text
        for key, value in self.__token.items():#index represent the token
            self.__token_reverse.append(key)
        #initalize model
        for num in self.__dim:
            #use tuple for immutable and reduced memory size
            layer = [node()] * num
            self.__layers.append(tuple(layer))
        self.__layers = tuple(self.__layers) 

    def encode(self,input):#converts text to int
        encoded_text = []
        for word in input:
            word = word.lower()
            word = word.strip(",.?!:-_")
            encoded_text.append(self.__token[word])
        return encoded_text
    def decode(self,tokens):
        for t,n in tokens:
            self.__token_reverse.append(t)
    
    def __append__(self,input,output):
        if isinstance(input,str) == False:raise TypeError("Input is not type: str")
        elif len(input) > self.__dim[0]: raise ValueError(f"input is too large\ninput size:{len(input)}\ninput layer size:{self.__dim[0]}")
        elif isinstance(output,str) == False:raise TypeError("output is not type: str")
        else:
            #encode input
            input = word_tokenize(input)
            output = word_tokenize(output)
            input = self.encode(input)
            output = self.encode(output)

            #pad input so that it is same size as input layer
            if len(input) < self.__dim[0]:
                padding = [0]*(self.__dim[0] - len(input))
            elif len(input) > self.__dim[0]:raise Warning('input size is larger than context window')
            input.extend(padding)
            #pad output so that it is same size as input layer
            if len(output) < self.__dim[0]:
                padding = [0]*(self.__dim[0] - len(output))
            elif len(output) > self.__dim[0]:raise Warning('input size is larger than context window')
            output.extend(padding)

    def train(self,input,output):
        pass
    def save(self):         
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

token_words = {}
token_words = tokenize(token_words,df)
model = text_model((50,10,10,50),token_words,10,100)
for text, code in df[['Text', 'Code']].values:
    model.train(text, code)