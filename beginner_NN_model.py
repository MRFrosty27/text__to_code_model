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
        self.__input = np.array(input).reshape(1,-1)
        self.__output = np.array(randint(1,2^16)).reshape(1,-1)
        self.lr = LinearRegression()
        self.lr.fit(self.__input,self.__output)
    
    def get_state(self):
        return self.lr.coef_[0],self.lr.intercept_
    
    def output(self):
        return relu(self.lr.predict(self.__input))

    def train(self,input):
        self.__input = np.array(input).reshape(1,-1)
        self.lr.fit(self.__input,self.__output)

    def change(self,value):
        value = np.array(value).reshape(1,-1)
        self.__output = np.add(self.__output,value)

class text_model:
    def __init__(self,dimension,token,learning_rate):
        self.__learning_rate = float(learning_rate)
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
    
    def train(self,input,output):
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
            data_forward = 0#sum of all wieghts
            data_layer = []#stores the the individual data of the output layer for calculating loss
            training_layers = []

            #initiat with first layer
            layer = []
            for n in range(self.__dim[0]):
                node = training_node(input[n])
                data_forward += node.output()
                layer.append(node)
            training_layers.append(layer)

            #create hidden layer and forward pass data
            for layer_num in range(1,len(self.__dim)-1):
                layer = []
                data_forward_next = 0
                for n in range(self.__dim[layer_num]):
                    node = training_node(data_forward)#forward data to the input
                    data_forward_next += node.output()
                    layer.append(node)
                training_layers.append(layer)
                data_forward = data_forward_next

            #output layer and forward pass
            for n in range(self.__dim[-1]):
                node = training_node(data_forward)#forward data to the input
                data_layer.append(node.output())
                layer.append(node)
            training_layers.append(layer)
            if len(data_layer) != self.__dim[-1]: raise Warning(f"size of output is not the same size as output layer\noutput size:{len(data_layer)}\noutput layer size:{self.__dim[-1]}")
            elif len(output) != self.__dim[-1]: raise Warning(f"size of output is not the same size as output layer\noutput size:{len(output)}\noutput layer size:{self.__dim[-1]}")

            #calc loss function
            avg_loss = 0
            for n in range(self.__dim[-1]):
                avg_loss += output[n] - data_layer[n]
            avg_loss = avg_loss / self.__dim[-1]
            print(f"initial loss:{avg_loss}")
            loss_list = [avg_loss]
            avg_loss_next = 0

            #forward pass training
            data_forward = 0
            data_forward_training = 0
            direction = 1#if positive node output value is increased and visa versa
            loss = 0

            #input layer
            for node_num in range(self.__dim[0]):
                for _ in range(10):
                    change_amount = avg_loss * self.__learning_rate * direction
                    node = training_layers[0][node_num]
                    node.change(change_amount)

                    #forward pass input after change
                    for n in range(self.__dim[0]):
                        training_layers[0][n].train(input[n])
                        data_forward_training += training_layers[0][n].output()
                    for l in range(layer_num,len(self.__dim)-1):
                        data_forward_training_next = 0
                        for n in range(self.__dim[l]):
                            training_layers[l][n].train(data_forward_training)
                            data_forward_training_next += training_layers[l][n].output()
                        data_forward_training = data_forward_training_next
                    data_forward_training_next = 0
                    data_layer = []
                    for n in range(self.__dim[-1]):
                        training_layers[-1][n].train(data_forward_training)
                        data_layer.append(training_layers[l][n].output())
                        data_forward_training_next += training_layers[-1][n].output()

                    for n in range(self.__dim[-1]):#calc new loss
                            loss += output[n] - data_layer[n]
                            if loss < 0:loss*-1#convert to positive value
                            avg_loss_next += loss
                    avg_loss_next = avg_loss_next / self.__dim[-1]
                    if 0 <= avg_loss-avg_loss_next <= 0.1: break#stop training when gradient is small
                    elif avg_loss > avg_loss_next:direction = 1
                    else: direction = -1
                    avg_loss = avg_loss_next
                    avg_loss_next = 0
                avg_loss = avg_loss_next
                loss_list.append(avg_loss)
                print(f"New loss:{avg_loss}")
                avg_loss_next = 0
                
                data_forward += node.output()
            #hidden layer and output
            for layer_num in range(1,len(self.__dim)):
                for node_num in range(self.__dim[layer_num]):
                    for _ in range(10):
                        change_amount = avg_loss * self.__learning_rate * direction
                        node = training_layers[layer_num][node_num]
                        node.change(change_amount)
                        #calc change
                        for l in range(layer_num,len(self.__dim)-1):
                            data_forward_training_next = 0
                            for n in range(self.__dim[l]):
                                training_layers[l][n].train(data_forward_training)
                                data_forward_training_next += training_layers[l][n].output()
                        data_layer = []
                        for n in range(self.__dim[-1]):
                            training_layers[-1][n].train(data_forward_training)
                            data_layer.append(training_layers[l][n].output())
                            data_forward_training_next += training_layers[-1][n].output()
                        for n in range(self.__dim[-1]):#calc new loss
                            loss += output[n] - data_layer[n]
                            if loss < 0:loss*-1#convert to positive value
                            avg_loss_next += loss
                        avg_loss_next = avg_loss_next / self.__dim[-1]
                        if 0 <= avg_loss-avg_loss_next <= 0.1: break#stop training when gradient is small
                        elif avg_loss > avg_loss_next:direction = 1
                        else: direction = -1
                        avg_loss = avg_loss_next
                        avg_loss_next = 0
                    avg_loss = avg_loss_next
                    loss_list.append(avg_loss)
                    print(f"New loss:{avg_loss}")
                    avg_loss_next = 0
                    data_forward_next += node.output()
                data_forward = data_forward_next

                mpl.figure()
                mpl.plot(loss_list)
                mpl.title('loss graph')
                mpl.xlabel(layer)
                mpl.ylabel('loss')
                mpl.show()
                
                #append points to 
                for layer_num in self.__dim:
                    for n in self.__dim[layer_num]:

                
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
model = text_model((50,10,10,50),token_words,0.1)
for text, code in df[['Text', 'Code']].values:
    model.train(text, code)