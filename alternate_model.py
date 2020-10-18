class model:
    def __init__(self, parameters):
        self.nA = parameters['nA']
        self.nC = parameters['nA']
        self.Tx = parameters['input_length']
        self.lr = parameters['learning_rate']
        self.beta1 = parameters['beta_1']
        self.beta2 = parameters['beta_2']
        self.decay = parameters['decay']
        self.nValues = parameters['nFeatures']
        self.reshapor = Reshape((1, 64))
        self.LSTM_cell = LSTM(64, return_state= True)
        self.embedding = Embedding(input_dim=parameters['nFeatures'], output_dim=64, input_length=self.Tx)
        self.densor = Dense(parameters['Y'].shape[1], activation='softmax')
        self.optimizer = parameters['optimizer']
        self.X = parameters['X']
        self.Y = parameters['Y']
        self.a0 = np.zeros((449, self.nA))
        self.c0 = np.zeros((449, self.nA))

    def modelInit(self):
        X = Input(shape=(self.Tx))
        a0 = Input(shape=(self.nA,), name = 'a0')
        c0 = Input(shape=(self.nC,), name = 'c0')
        a = a0
        c = c0
        outputs = []
        for t in range(self.Tx):
            e = self.embedding(X)
            x = Lambda(lambda z: z[:, t, :])(e)
            x = self.reshapor(x)
            a, _ , c = self.LSTM_cell(inputs = x, initial_state=[a, c])
        outputs.append(self.densor(a))

        self.model = Model(inputs=[X, a0, c0], outputs=outputs)

    def modelOptimizer(self):
        self.opt = Adam(lr= self.lr, beta_1=self.beta1, beta_2=self.beta2, decay=self.decay)


    def modelCompile(self):
        self.model.compile(optimizer = self.opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def fitModel(self):
        self.model.fit([self.X, self.a0, self.c0], self.Y, epochs=100, batch_size=3)

    def predict(self, input):
        self.model.predict(input)


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(patterns)
st.write(tokenizer)
#generate the sequences from the pattern of question previously extracted
pattern_sequences = tokenizer.texts_to_sequences(patterns)
#padding the sequences to ensure all sequences are of the same length
pattern_sequences_padded = pad_sequences(
    pattern_sequences, maxlen=max_words, padding="post", truncating="post"
    )

parameters = {
    'nA': 64,
    'input_length': 20,
    'nFeatures': num_words,
    'optimizer': 'Adam',
    'learning_rate': 0.01,
    'beta_1': 0.9,
    'beta_2': 0.99,
    'decay': 0.01, 
    'X': pattern_sequences_padded,
    'Y': labels
}

labels.shape
pattern_sequences_padded.shape

m1 = model(parameters)
m1.modelInit()
m1.modelOptimizer()
m1.modelCompile()
m1.fitModel()
