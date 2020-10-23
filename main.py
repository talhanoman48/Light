from text_processing_functions import *
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import streamlit as st
import time
 

#initialize list of subjects and make select box in sidebar
subjects = ["CS", "Statics", "LA", "Islamiat", "train_new"]
subject = st.sidebar.selectbox("Subject", subjects)
load_existing = st.sidebar.checkbox("do you want to load existing model".title(), value = True)
#setup the title and opening statement
st.title('Light')
st.write("Your learning companion".title())
#setting the maximum length of the sequences
max_words = 20
#intialize array to store exectuion times and num_execution to count functions executed
execution_time = []
num_execution = 0
#generating all the relavent dictionaries
patterns_counter, patterns, classes, responses,exec_time = gen_dictionaries(subject)
execution_time.append(exec_time)
labels = generate_labels(file_name=subject, classes=classes)
num_words = len(patterns_counter)
#initialize list of funtion names used to tell the execution time
name = ["gen_dictionaries", "load_my_model", "inference_function"]


def fit_tokenizer(load_exising = False):
    '''
    Returns the tokenizer and if it doesn't exist will create it

    load_existing (Type: boolean): used to indicate whether to load existing tolenizer 
                                    or create a new one... useful when training data updated
    '''
    if load_exising:
        if subject != "":
            with open(subject + 'tokenizer.json') as f:
                data = json.load(f)
                tokenizer = tokenizer_from_json(data)
            return tokenizer
    else:
        if subject != "":
            tokenizer = Tokenizer(num_words=num_words, oov_token='UNK')
            tokenizer.fit_on_texts(patterns)
            tokenizer_json = tokenizer.to_json()
            with io.open(subject + 'tokenizer.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))
            return tokenizer


#initialize and fit the tokenizer or load existing
tokenizer = fit_tokenizer(load_existing)
#generate the sequences from the pattern of question previously extracted
pattern_sequences = tokenizer.texts_to_sequences(patterns)
#padding the sequences to ensure all sequences are of the same length
pattern_sequences_padded = pad_sequences(
    pattern_sequences, maxlen=max_words, padding="post", truncating="post"
    )

   
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


def createModel():
    '''
    Creates the model and returns it
    '''
    #initializing the model
    inputs = Input(shape = (max_words,))
    #Configure Layers of the model
    X = Embedding(num_words, 100, input_length=max_words, name= "embedding")(inputs)
    X = LSTM(64, dropout=0.3, name= "LSTM")(X)
    outputs = Dense(labels.shape[1], activation="softmax", name="outputs")(X)
    #selecting the optimizer (Adam with default parameters i.e, learning_rate = 0.001, beta1 (momentum)= 0.9, beta2(RMSprop)= 0.999, epsilon= 1e-7)
    optimizer = Adam()
    #compiling the model
    model = Model(inputs=inputs, outputs=outputs, name="Light")
    model.compile(
    optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if load_existing:
    #if subject is define load the corresponding model
    model = load_my_model(subject)
else:
    #create model
    model = createModel()
    #print the summary of the model to make sure every dimension is accounted for
    model.summary()
    #fit the model (number epochs intentionally kept low to prevent overfitting of the model)
    model.fit(x=pattern_sequences_padded, y=labels, batch_size=32, epochs=600)
    #save the model
    model.save(subject + ".hd5")    

def predict(input, num_execution, ERR_Threshold):
    '''
    Takes a string and Error Threshold as input and outputs an appropriate respose to the question

    input (Type: string): user input string
    num_execution (Type: int): number of functions executed to keep track of time taken by the script to run
    ERR_Threshhold (Type: int)
    '''
    tic = time.time()
    input_1 = []
    input_1.append(input)
    seq = tokenizer.texts_to_sequences(input_1)
    st.write(seq)
    padded = pad_sequences(
        seq, maxlen = max_words, padding="post", truncating="post"
    )
    prediction = model.predict(padded)
    if np.max(prediction) > ERR_Threshold:
        st.write(responses[np.squeeze(np.argmax(prediction))])
    toc = time.time()
    execution_time.append(toc - tic)
    num_execution += 1


x = st.text_input(label=f"Enter Your query regarding {subject} below".title())
predict(x, num_execution, 0.6)
run_time()

