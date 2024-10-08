# Import necessary libraries
from keras.layers import Input, Reshape, LSTM, Embedding, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Define the model class
class model:
    def __init__(self, parameters):
        """
        Initialize the model with the given parameters.
        
        Parameters:
        - parameters (dict): A dictionary containing the model's hyperparameters and data.
        """
        # LSTM units
        self.nA = parameters['nA']
        self.nC = parameters['nA']  # Cell state size, set to the same as nA
        self.Tx = parameters['input_length']  # Sequence length
        self.lr = parameters['learning_rate']  # Learning rate
        self.beta1 = parameters['beta_1']  # Adam optimizer beta_1
        self.beta2 = parameters['beta_2']  # Adam optimizer beta_2
        self.decay = parameters['decay']  # Learning rate decay
        self.nValues = parameters['nFeatures']  # Input feature size (vocabulary size)

        # Layers
        self.reshapor = Reshape((1, 64))  # Reshape the input
        self.LSTM_cell = LSTM(64, return_state=True)  # LSTM cell with 64 units
        self.embedding = Embedding(input_dim=parameters['nFeatures'], output_dim=64, input_length=self.Tx)  # Embedding layer
        self.densor = Dense(parameters['Y'].shape[1], activation='softmax')  # Output layer with softmax activation

        # Optimizer settings
        self.optimizer = parameters['optimizer']

        # Training data
        self.X = parameters['X']  # Input sequences
        self.Y = parameters['Y']  # Output labels

        # Initial states for LSTM
        self.a0 = np.zeros((449, self.nA))  # Initial hidden state (set to zeros)
        self.c0 = np.zeros((449, self.nA))  # Initial cell state (set to zeros)

    def modelInit(self):
        """
        Initialize the LSTM-based model architecture.
        """
        # Input layers
        X = Input(shape=(self.Tx,))  # Input sequence
        a0 = Input(shape=(self.nA,), name='a0')  # Initial hidden state
        c0 = Input(shape=(self.nC,), name='c0')  # Initial cell state

        # Initial states for the LSTM
        a = a0
        c = c0

        outputs = []  # Collect outputs

        # Iterate over the sequence length
        for t in range(self.Tx):
            e = self.embedding(X)  # Embed the input sequence
            x = Lambda(lambda z: z[:, t, :])(e)  # Slice the input at time step t
            x = self.reshapor(x)  # Reshape the input
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])  # LSTM cell update at time t

        # Final output through a dense layer
        outputs.append(self.densor(a))

        # Create the Keras model
        self.model = Model(inputs=[X, a0, c0], outputs=outputs)

    def modelOptimizer(self):
        """
        Set up the optimizer for model training using the Adam optimizer.
        """
        # Adam optimizer with given parameters
        self.opt = Adam(lr=self.lr, beta_1=self.beta1, beta_2=self.beta2, decay=self.decay)

    def modelCompile(self):
        """
        Compile the model with the Adam optimizer and categorical crossentropy loss.
        """
        # Compile the model using the optimizer, loss function, and accuracy metric
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def fitModel(self):
        """
        Train the model on the input data.
        """
        # Train the model on input data (X, a0, c0) and labels (Y)
        self.model.fit([self.X, self.a0, self.c0], self.Y, epochs=100, batch_size=3)

    def predict(self, input):
        """
        Make predictions using the trained model.
        
        Parameters:
        - input: Input data to make predictions on.
        
        Returns:
        - Predicted output.
        """
        # Generate predictions based on the input
        return self.model.predict(input)

# Tokenizer and preprocessing
tokenizer = Tokenizer(num_words=num_words)  # Initialize the tokenizer with the number of words
tokenizer.fit_on_texts(patterns)  # Fit the tokenizer on the input text patterns
st.write(tokenizer)  # Output tokenizer

# Generate sequences from the text patterns
pattern_sequences = tokenizer.texts_to_sequences(patterns)

# Pad the sequences to ensure they have the same length
pattern_sequences_padded = pad_sequences(
    pattern_sequences, maxlen=max_words, padding="post", truncating="post"
)

# Define model parameters
parameters = {
    'nA': 64,  # Number of LSTM units
    'input_length': 20,  # Length of input sequences
    'nFeatures': num_words,  # Vocabulary size
    'optimizer': 'Adam',  # Optimizer type
    'learning_rate': 0.01,  # Learning rate
    'beta_1': 0.9,  # Adam beta1
    'beta_2': 0.99,  # Adam beta2
    'decay': 0.01,  # Learning rate decay
    'X': pattern_sequences_padded,  # Input sequences
    'Y': labels  # Output labels
}

# Print data shapes for verification
labels.shape
pattern_sequences_padded.shape

# Initialize, compile, and train the model
m1 = model(parameters)  # Instantiate the model
m1.modelInit()  # Initialize the model architecture
m1.modelOptimizer()  # Set up the optimizer
m1.modelCompile()  # Compile the model
m1.fitModel()  # Train the model
