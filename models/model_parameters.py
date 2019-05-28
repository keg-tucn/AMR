class ModelParameters:
    def __init__(self,
                 embeddings_dim=200,
                 train_epochs=50,
                 hidden_layer_size=1024,
                 learning_rate=0.005,  # original value: 0.01
                 dropout=0.25,
                 recurrent_dropout=0.25,
                 ):
        self.embeddings_dim = embeddings_dim
        self.train_epochs = train_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
