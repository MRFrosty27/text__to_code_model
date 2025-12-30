from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from nltk.tokenize import word_tokenize,sent_tokenize
from random import randint
from textblob import Word
from math import log2
#from sklearn.metrics import 
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import array,os

df = pd.read_csv('ML\dataset.csv',sep=';',engine='python',dtype=str)
df['Text'] = df['Text'].astype(str)
df['Code'] = df['Code'].astype(str)

def tokenize(token_dict,dataset):
    if not isinstance(token_dict, dict):
        raise TypeError("First argument 'token_dict' must be a dict")
    
    # Initialize default tokens if dictionary is empty
    if len(token_dict) == 0:
        token_dict = {
            '': 0,        # padding or empty
            '(': 1,
            ')': 2,
            '    ': 3,    # indentation (4 spaces)
            ':': 4,
            '[': 5,
            ']': 6,
            '!unknown!': 7  # unknown token
        }
    
    # Case 1: Dataset is a pandas DataFrame
    if isinstance(dataset, pd.DataFrame):
        if dataset.empty:
            raise ValueError("DataFrame is empty")
        
        # Process all columns that contain strings/objects
        text_columns = dataset.select_dtypes(include=['object', 'string']).columns
        if len(text_columns) == 0:
            raise ValueError("DataFrame has no columns with text data (object/string dtype)")
        
        for col in text_columns:
            for value in dataset[col].dropna():  # Skip NaN
                if not isinstance(value, str):
                    continue
                value = str(value)  # Ensure it's string
                sentences = sent_tokenize(value)
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    for word in words:
                        word = word.lower()
                        # Skip common punctuation
                        if word in {',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}'}:
                            continue
                        if word not in token_dict:
                            token_dict[word] = len(token_dict)
    
    # Case 2: Dataset is a string path to a text file
    elif isinstance(dataset, str):
        if not os.path.isfile(dataset):
            raise FileNotFoundError(f"Text file not found: {dataset}")
        
        with open(dataset, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                word = word.lower()
                if word in {',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}'}:
                    continue
                if word not in token_dict:
                    token_dict[word] = len(token_dict)
    
    else:
        raise TypeError("Second argument 'dataset' must be a pandas.DataFrame or a file path (str)")
    
    return token_dict

class node:
    def __init__(self):
        self.__wieght = randint(1,2^16)
        self.__bias = randint(1,2^4)

    def get_state(self):
        return self.__wieght,self.__bias
    
    def output(self,input):
        return self.__wieght*input+self.__bias

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
        self.__training_inputs = []#stores the tokenizes inputs
        self.__training_outputs = []#stores the tokenizes outputs
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

    def add_training_data(self,input,output):
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

            self.__training_inputs.append(input)
            self.__training_outputs.append(output)

    def train(self):
        #calc loss
        self.forward_pass_all_data()

        for layer in self.__layers:
            for node in layer:
                change_amount_wieght = self.__learning_rate
                change_amount_bias = self.__learning_rate
                direction_wieght = 1
                direction_bias = 1
                improvement_after_weight_change = 0#the change in loss, positive value = decrease in loss
                improvement_after_bias_change = 0#the change in loss, positive value = decrease in loss

                for e in range(self.__epoch):
                    #change node wieght or bias
                    #wieght
                    node.change_wieght(change_amount_wieght)
                    #calc new loss after wieght change, forward pass
                    improvement_after_weight_change = self.forward_pass_all_data()
                    if improvement_after_weight_change < 0: 
                        node.change_wieght(-change_amount_wieght)#revert change
                        direction_wieght *= -1#flip direction
                        print(f'fliped weight direction after {e} epoch as improvement was {improvement_after_weight_change}')

                    #bias
                    node.change_bias(change_amount_bias)
                    #calc new loss after bias change, forward pass
                    improvement_after_bias_change = self.forward_pass_all_data()
                    if improvement_after_bias_change < 0: 
                        node.change_bias(-change_amount_bias)#revert change
                        direction_bias *= -1#flip direction
                        print(f'fliped bias direction after {e} epoch as improvement was {improvement_after_bias_change}')

                    if (improvement_after_weight_change + improvement_after_bias_change)/2 < 0.001 and e > self.__epoch//10: 
                        print(f'reached min improvement threshold after {e} epoch')
                        break
                    elif isinstance(self.__avg_loss,(float,int)) == False: raise TypeError('loss value')

                    #calc value to change by
                    change_amount_wieght = self.__learning_rate * improvement_after_weight_change * direction_wieght
                    change_amount_bias = self.__learning_rate * improvement_after_bias_change * direction_bias
                    print(f'loss:{self.__avg_loss}')
            print(f'loss:{self.__avg_loss}')

    def forward_pass_all_data(self):
        if len(self.__training_inputs) != len(self.__training_outputs): raise ValueError(f'the size of training inputs and outputs are not the same\nsize-\ntraining input:{len(self.__training_inputs)}\ntraining output:{len(self.__training_outputs)}')

        data_forward = 0
        data_forward_next = 0
        sum_loss = 0#stores the sum of all loss values of output nodes for an individual input
        total_avg_loss = 0
        list_of_average_loss = []#stores the

        for nth_data in range(len(self.__training_inputs)):
            sum_loss = 0
            for n_input in range(len(self.__layers[0])):#pass input into input layer
                node = self.__layers[0][n_input]
                input_data = self.__training_inputs[nth_data][n_input]
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
                sum_loss += abs(node.output(data_forward) - self.__training_outputs[nth_data][n_output])

            #calc loss for an individual input
            data_forward = data_forward_next
            data_forward_next = 0
            list_of_average_loss.append(sum_loss/len(self.__layers[-1]))
        
        #calc avg of loss for all inputs
        for average_loss in list_of_average_loss:
            total_avg_loss += average_loss
        total_avg_loss = total_avg_loss/len(list_of_average_loss)

        if self.__avg_loss == None: 
            self.__avg_loss = total_avg_loss
            print('self.__avg_loss was None')
            return None#only used for when calculating the loss for the first time
        elif isinstance(total_avg_loss,float) == True:
            delta_loss = self.__avg_loss - total_avg_loss#positive value is improvement
            self.__avg_loss = total_avg_loss
            return delta_loss
        else: raise TypeError('self.__avg_loss was not float')
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
    
    def load_as_array(self):
        model_weight_in_array_format = []
        model_bias_in_array_format = []
        model_layer_weights = []
        model_layer_bias = []

        for layer in self.__layers:
            for node in layer:
                state = node.get_state()
                model_layer_weights.append(state[0])
                model_layer_bias.append(state[1])
            model_weight_in_array_format.append(model_layer_weights)
            model_bias_in_array_format.append(model_layer_bias)

        model_weight_in_array_format = np.array(model_bias_in_array_format)
        model_bias_in_array_format = np.array(model_bias_in_array_format)

        if model_weight_in_array_format.ndim() != 2:
            raise TypeError('Failed to convert model weights to a 2D numpy array')
        elif model_bias_in_array_format.ndim() != 2:
            raise TypeError('Failed to convert model biases to a 2D numpy array')
        return model_weight_in_array_format,model_bias_in_array_format

class identify_model:
    def __init__(self, input_size, first_training_input, first_training_output):
        self.__input_size = int(input_size)
        self.__mlp_classifier = None
        
        # --- Handle first_training_input: must be 2D (n_samples, input_size) ---
        if not isinstance(first_training_input, np.ndarray):
            if not isinstance(first_training_input, (list, tuple)):
                raise TypeError('first_training_input must be a numpy array, list of lists, or tuple of tuples')
            # Convert list of lists / tuple of tuples to numpy array
            input_arr = np.array(first_training_input, dtype=int)
        else:
            input_arr = first_training_input
            if not np.issubdtype(input_arr.dtype, np.integer):
                input_arr = input_arr.astype(int)
        
        # Must be exactly 2D
        if input_arr.ndim != 2:
            raise ValueError('first_training_input must be 2D (n_samples rows x input_size columns)')
        
        # Check number of features matches input_size
        if input_arr.shape[1] != self.__input_size:
            raise ValueError(
                f'Each input sample must have exactly {self.__input_size} features, '
                f'but got {input_arr.shape[1]}'
            )
        
        n_samples = input_arr.shape[0]
        if n_samples == 0:
            raise ValueError('first_training_input must contain at least one sample')
        
        self.__training_inputs = input_arr  # Shape: (n_samples, input_size), dtype: int
        
        # --- Handle first_training_output: must have n_samples integer scalars ---
        if not isinstance(first_training_output, np.ndarray):
            if not isinstance(first_training_output, (list, tuple)):
                raise TypeError('first_training_output must be a list, tuple, or numpy array of integers')
            output_arr = np.array(first_training_output, dtype=int)
        else:
            output_arr = first_training_output
            if not np.issubdtype(output_arr.dtype, np.integer):
                output_arr = output_arr.astype(int)
        
        # Allow 1D or 2D (e.g., column vector), but not higher
        if output_arr.ndim == 1:
            pass  # Good: shape (n_samples,)
        elif output_arr.ndim == 2:
            if output_arr.shape[1] != 1:
                raise ValueError('If first_training_output is 2D, it must be a column vector (n_samples, 1)')
            output_arr = output_arr.ravel()  # Flatten to 1D
        else:
            raise ValueError('first_training_output must be 1D or 2D (column vector)')
        
        # Must match number of samples
        if len(output_arr) != n_samples:
            raise ValueError(
                f'Number of outputs ({len(output_arr)}) must match number of input samples ({n_samples})'
            )
        
        # Store as 1D array for simplicity (easy to extend later)
        self.__training_outputs = output_arr  # Shape: (n_samples,), dtype: int

    def add_training_data(self, new_inputs, new_outputs):
        """
        Add more training samples to the model.
        
        Parameters:
            new_inputs: list of lists, tuple, or np.ndarray of shape (n_new_samples, input_size)
            new_outputs: list, tuple, or np.ndarray of integer scalars (length n_new_samples)
        """
        # --- Validate and convert new_inputs ---
        if not isinstance(new_inputs, (np.ndarray,list,tuple)):
             new_input_arr = np.array(new_inputs, dtype=int)
        else:
            new_input_arr = new_inputs.astype(int) if not np.issubdtype(new_inputs.dtype, np.integer) else new_inputs
        
        if new_input_arr.ndim != 2:
            raise ValueError('new_inputs must be 2D (n_samples x input_size)')
        if new_input_arr.shape[1] != self.__input_size:
            raise ValueError(f'Each new sample must have {self.__input_size} features, got {new_input_arr.shape[1]}')
        if new_input_arr.shape[0] == 0:
            raise ValueError('Cannot add zero new samples')
        
        # --- Validate and convert new_outputs ---
        if not isinstance(new_outputs, np.ndarray):
            new_output_arr = np.array(new_outputs, dtype=int)
        else:
            new_output_arr = new_outputs.astype(int) if not np.issubdtype(new_outputs.dtype, np.integer) else new_outputs
        
        if new_output_arr.ndim == 2:
            if new_output_arr.shape[1] != 1:
                raise ValueError('If 2D, new_outputs must be a column vector')
            new_output_arr = new_output_arr.ravel()
        elif new_output_arr.ndim != 1:
            raise ValueError('new_outputs must be 1D or 2D column vector')
        
        if len(new_output_arr) != new_input_arr.shape[0]:
            raise ValueError('Number of new outputs must match number of new input samples')
        
        # --- Concatenate to existing data ---
        self.__training_inputs = np.vstack([self.__training_inputs, new_input_arr])
        self.__training_outputs = np.concatenate([self.__training_outputs, new_output_arr])
        
        print(f"Added {new_input_arr.shape[0]} new samples. Total samples: {self.__training_inputs.shape[0]}")

    def train(self):
        if self.__input_size == 1: hidden_layer_depth = round(log2(len(self.__training_inputs)))
        else: 
            number_of_hidden_layers = round(log2(self.__input_size))
            hidden_layer_len = round(log2(len(self.__training_inputs)))
            hidden_layer_depth = [[hidden_layer_len] * number_of_hidden_layers]
        self.__mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_depth, max_iter=1000, random_state=42, solver='adam', activation='relu')
        self.__mlp_classifier.fit(self.__training_inputs,self.__training_outputs)

    def output(self,input):
        # --- Validate and convert new_inputs ---
        if not isinstance(input, (np.ndarray,list,tuple)):
            input = np.array(input, dtype=int)
        else:
            input = input.astype(int) if not np.issubdtype(input.dtype, np.integer) else input

        if len(input) != 1: raise ValueError('only a single input is permitted')
        elif input.shape[1] <= self.__input_size: raise ValueError('size of input is too small')
        elif input.shape[1] >= self.__input_size: raise ValueError('size of input is too large')

        return self.__mlp_classifier.predict(input)

def relu(input):
    if input <= 0:
        return 0
    else: return input

def prod_save(self,path,models):         
    with open('custom_nn.py','w') as model:
        #store model as an python script that is executed via terminal.
        #the script take a string arg as an input
        #encode the string input arg
        #store the wieghts and biases
        #decode output to string
        #all output values of nodes must pass through relu
        #only save minimal feature for lowest resource requirements
        if isinstance(models,(tuple,list)) == True:
            for model in models:
                if isinstance(model,text_model) == False and isinstance(model,identify_model) == False: raise TypeError('arg models contains a non-model class datatype')
        else: raise TypeError('arg models is neither type list or tuple')
                
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

token_words = {}
token_words = tokenize(token_words,df)
model = text_model((100,10,100),token_words,1,1000)
for text, code in df[['Text', 'Code']].values:
    model.add_training_data(text, code)
model.train()
print(f"model output: {model.output('print ')}")
print(f"model output: {model.output('print ')}")