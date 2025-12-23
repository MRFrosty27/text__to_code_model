from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import 
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from random import randint
from textblob import Word
import array

df = pd.read_csv('ML\dataset.csv',sep=';',engine='python',dtype=str)
df['Text'] = df['Text'].astype(str)
df['Code'] = df['Code'].astype(str)

def tokenize(token_dict,dataframe):
    if isinstance(token_dict,dict) == False:raise TypeError('First arg is not type dict')
    elif isinstance(dataframe,pd.DataFrame) == False: TypeError('second arg is not type dataframe')
    elif 'Text' not in dataframe.columns: raise NameError('column Text was not found')
    elif len(token_dict) == 0:
        #'!unkown!' is used when a word is not stored in the token dict
        token_dict = {'':0,'(':1,')':2,'    ':3,':':4,'[':5,']':6,'!unkown!':7}
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
        self.__wieght = randint(1,2^16)
        self.__bias = randint(1,2^16)

    def get_state(self):
        return self.__wieght,self.__bias
    
    def output(self,input):
        return relu(self.__wieght*input+self.__bias)

    def change_wieght(self,value):
        self.__wieght += value

    def change_bias(self,value):
        self.__bias += value

    def __str__(self):
        return f"Weight:{self.__wieght}, Bias:{self.__bias}"

class text_model:
    def __init__(self,dimension,token,learning_rate,epoch):
        self.__learning_rate = float(learning_rate)
        self.__epoch = epoch
        self.__dim = tuple(dimension)#defines the archetexture of the model
        self.__layers = []#stores the layers that contain nodes
        self.__token = token#used to encode text
        self.__token_reverse = tuple(self.__token.keys())#used to decode text
        self.__inputs = []#stores the tokenizes inputs
        self.__outputs = []#stores the tokenizes outputs
        self.__loss_list = []
        self.__avg_loss = None
        #initalize model
        for num in self.__dim:
            layer = [node() for _ in range(num)]
            self.__layers.append(tuple(layer))
        self.__layers = tuple(self.__layers) 

    def encode(self,input):#converts text to int
        encoded_text = []
        if isinstance(input,str) == False:raise TypeError("Input is not type: str")
        sen = sent_tokenize(input)
        for line in sen:
            words = word_tokenize(line)
            for word in words:
                word = word.lower()
                word = word.strip(",.?!:-_")
                try:
                    encoded_text.append(int(self.__token[word]))
                except KeyError:
                    print(f"'{word}' is not stored as a token")
                    try:
                        corrected_word = Word(word)
                        corrected_word = Word.correct(corrected_word)
                        encoded_text.append(int(self.__token[corrected_word]))
                        print(f"'{word}' was corrected to '{corrected_word}'")
                    except KeyError:
                        print(f"Both '{word}' and '{corrected_word}' are not stored in token")
                        encoded_text.append(7)#7 is the encoded value for !unkown!
                except:
                    print(Exception)
        return encoded_text
    
    def decode(self,tokens):#convert array of int values to text
        output = ''
        for token in tokens:
            try:
                output += f"{self.__token_reverse[token]} "
            except IndexError:
                output += f"{self.__token_reverse[-1]} "
        return str(output)

    def add_data(self,input,output):
        if isinstance(input,str) == False:raise TypeError("Input is not type: str")
        elif len(input) > self.__dim[0]: raise ValueError(f"input is too large\ninput size:{len(input)}\ninput layer size:{self.__dim[0]}")
        elif isinstance(output,str) == False:raise TypeError("output is not type: str")
        else:
            #encode data
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

            self.__inputs.append(tuple(input))
            self.__outputs.append(tuple(output))

    def train(self):
        #calc loss
        self.forward_pass_all_data()

        #change wieghts and bias
        direction_wieght = 1
        direction_bias = 1
        improvement_after_wieght = 0#the change in loss, positive value = decrease in loss
        improvement_after_bias = 0#the change in loss, positive value = decrease in loss
        for layer in self.__layers:
            for node in layer:
                for _ in range(self.__epoch):
                    #calc value to change by
                    change_amount_wieght = self.__learning_rate * self.__avg_loss * direction_wieght
                    change_amount_bias = self.__learning_rate * self.__avg_loss * direction_bias

                    #change node wieght or bias
                    #wieght
                    node.change_wieght(change_amount_wieght)
                    #calc new loss after wieght change, forward pass
                    improvement_after_wieght = self.forward_pass_all_data()
                    if improvement_after_wieght < 0: 
                        node.change_wieght(-change_amount_wieght)#revert change
                        direction_wieght *= -1#flip direction

                    #bias
                    node.change_bias(change_amount_bias)
                    #calc new loss after bias change, forward pass
                    improvement_after_bias = self.forward_pass_all_data()
                    if improvement_after_bias < 0: 
                        node.change_bias(-change_amount_bias)#revert change
                        direction_bias *= -1#flip direction

                    if (improvement_after_wieght + improvement_after_bias)/2 < 0.001: break
            print(self.__avg_loss)

    def prod_save(self,path):         
        with open('custom_nn.py','w') as model:
            #store model as an python script that is executed via terminal.
            #the script take a string arg as an input
            #encode the string input arg
            #store the wieghts and biases
            #decode output to string
            #all output values of nodes must pass through relu
            #only save minimal feature for lowest resource requirements
            with open(f'{path}','w') as new_model:
                new_model.write("import numpy as np")
                new_model.write("parameters_wieghts = np.array(")
                new_model.write("from sys import argv")
                wieght_array = []
                for layer in self.__layers:
                    temp_array = []
                    for node in layer:
                        temp_array.append(node.get_state[0])
                    wieght_array.append(temp_array)
                new_model.write(f"{wieght_array}")
                new_model.write(",int32)")
                new_model.write("parameters_bias = np.array(")
                bias_array = []
                for layer in self.__layers:
                    temp_array = []
                    for node in layer:
                        temp_array.append(node.get_state[1])
                    bias_array.append(temp_array)
                new_model.write(f"{bias_array}")
                new_model.write(",int32)")
                new_model.write(')')
                new_model.write("if __name__ == '__Main__'")
                new_model.write("if isinstance(argv[1],str) == False: raise Warning('')")

    def dev_save(self,path): 
        #save all features for later
        #for example self.__inputs and self.__outputs so that it does not have to retrain
        pass
    def forward_pass_all_data(self):
        data_forward = 0
        data_forward_next = 0
        loss = 0
        new_loss_list = []
        new_avg_loss = 0

        for nth_data in range(len(self.__inputs)):
            for n_input in range(len(self.__layers[0])):#pass input into input layer
                node = self.__layers[0][n_input]
                input_data = self.__inputs[nth_data][n_input]
                data_forward_next += node.output(input_data) 
            data_forward = data_forward_next
            data_forward_next = 0

            for layer_num in range(1,len(self.__dim)-1):#hidden layer
                for node in self.__layers[layer_num]:
                    data_forward_next += node.output(data_forward) 
                data_forward = data_forward_next
                data_forward_next = 0

            for n_output in range(len(self.__layers[-1])):#output layer
                node = self.__layers[-1][n_output]
                loss += node.output(data_forward) - self.__outputs[nth_data][n_output]

            #calc loss for an individual input
            data_forward = data_forward_next
            data_forward_next = 0
            loss = loss/len(self.__layers[-1])
            new_loss_list.append(loss)
        
        #calc avg of loss for all inputs
        for l in new_loss_list:
            new_avg_loss += l
        new_avg_loss = new_avg_loss/len(new_loss_list)

        if self.__avg_loss == None: 
            self.__avg_loss = new_avg_loss
            return None#only used for when calculating the loss for the first time
        else:
            delta_loss = self.__avg_loss - new_avg_loss
            self.__loss_list = new_loss_list
            self.__avg_loss = new_avg_loss
            return delta_loss
        
    def output(self,input):
        if isinstance(input,str) == False:raise TypeError("Input is not type: str")
        data = self.encode(input)
        #pad input so that it is same size as input layer
        if len(data) < self.__dim[0]:
            padding = [0]*(self.__dim[0] - len(data))
        elif len(data) > self.__dim[0]:raise Warning('input size is larger than context window')
        data.extend(padding)

        data_forward = 0
        data_forward_next = 0
        data_output = array.array('L', [])

        for n in range(len(self.__layers[0])):#pass input into input layer
            node = self.__layers[0][n]
            data_forward_next += node.output(data[n]) 
        data_forward = data_forward_next
        data_forward_next = 0

        for layer_num in range(1,len(self.__dim)-1):#hidden layer
            for node in self.__layers[layer_num]:
                data_forward_next += node.output(data_forward) 
            data_forward = data_forward_next
            data_forward_next = 0

        for n in range(len(self.__layers[-1])):#output layer
            node = self.__layers[-1][n]
            data_output.append(round(node.output(data_forward)))
        
        data_output = self.decode(data_output)
        return data_output
    
    def __str__(self):
        string = []
        for layer in self.__layers:
            for node in layer:
                string.append(str(node))
        return str(string)
        
def relu(input):
    if input <= 0:
        return 0
    else: return input

token_words = {}
token_words = tokenize(token_words,df)
model = text_model((100,100,100,100),token_words,10,100)
for text, code in df[['Text', 'Code']].values:
    model.add_data(text, code)
model.train()
print(f"model output: {model.output('print ')}")