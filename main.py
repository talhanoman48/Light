from pandas.tests.extension.decimal import DecimalArray
from tensorflow.python.ops.gen_batch_ops import batch
from text_processing_functions import *
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import streamlit as st
from PIL import Image
import time
 

#initialize list of subjects and make select box in sidebar
subjects = ["CS", "Statics", "LA", "Islamiat"]
subject = st.sidebar.selectbox("Subject", subjects)
#setting the maximum length of the sequences
max_words = 20
#intialize array to store exectuion times
execution_time = []
num_execution = 0
#generating all the relavent dictionaries
patterns_counter, patterns, classes, responses,exec_time = gen_dictionaries(subject)
execution_time.append(exec_time)
labels = generate_labels(file_name=subject, classes=classes)
num_words = len(patterns_counter)
#initialize list of funtion names used to tell the execution time
name = ["gen_dictionaries", "load_my_model", "inference_function"]
#initialize and fit the tokenizer or load existing
try:
    if subject != "":
        with open(subject + 'tokenizer.json') as f:
            data = json.load(f)
            tokenizer = Tokenizer.from_json(data)
except:
    if subject != "":
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(patterns)
        #generate the sequences from the pattern of question previously extracted
        pattern_sequences = tokenizer.texts_to_sequences(patterns)
        #padding the sequences to ensure all sequences are of the same length
        pattern_sequences_padded = pad_sequences(
            pattern_sequences, maxlen=max_words, padding="post", truncating="post"
            )
        tokenizer_json = tokenizer.to_json()
        with io.open(subject + 'tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    else:
        pass


def run_time(specific_times = False):
    '''
    Returns the exectution times for each function and total exectution time
    '''
    if specific_times == True:
        for exec in range(num_execution+1):
            st.write(f"The model {name[exec]} took {execution_time[exec]} to run!")
        total_runtime = np.sum(execution_time)
        st.write(f"The scipt took {total_runtime} to run")
    else:
        total_runtime = np.sum(execution_time)
        st.write(f"The scipt took {total_runtime} to run")


tic = time.time()
@st.cache(allow_output_mutation=True) 
def load_my_model(subject):
    '''
    Inputs:
    subject(type string) the name of the model save file minus the file extension
    Outputs:
    model infrastrucure and weights 
    '''
    model = load_model(subject + ".hd5")
    model.summary()
    return model
toc = time.time()
execution_time.append(toc - tic)
num_execution += 1

def TrainModel():
        #initializing the model
        model = Sequential()
        #adding layers
        model.add(Embedding(num_words, 32, input_length=max_words))
        model.add(LSTM(64, dropout=0.3))
        model.add(Dense(labels.shape[1], activation="softmax"))
        #selecting the optimizer (Adam with default parameters i.e, learning_rate = 0.001, beta1 (momentum)= 0.9, beta2(RMSprop)= 0.999, epsilon= 1e-7)
        optimizer = Adam()
        #compiling the model
        model.compile(
            optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"]
            )
        #print the summary of the model to make sure every dimension is accounted for
        model.summary()
        #fit the model (number epochs intentionally kept low to prevent overfitting of the model)
        model.fit(x=pattern_sequences_padded, y=labels, batch_size=32, epochs=600)
        #save the model
        model.save(subject + ".hd5")
        return model



if subject != "":
    model = load_my_model(subject)


tic = time.time()
input_1 = []
x = st.text_input(label="Enter Your query")
input_1.append(x)
seq = tokenizer.texts_to_sequences(input_1)
padded = pad_sequences(
    seq, maxlen = max_words, padding="post", truncating="post"
)
prediction = model.predict(padded)
st.write(responses[np.squeeze(np.argmax(prediction))])
toc = time.time()
execution_time.append(toc - tic)
num_execution += 1



run_time()

