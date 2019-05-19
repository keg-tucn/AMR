class ModelParameters:
    def __init__(self,
                 max_len=50,
                 embeddings_dim=200,
                 train_epochs=50,
                 hidden_layer_size=1024,
                 learning_rate=0.005,  # original value: 0.01
                 dropout=0.25,
                 recurrent_dropout=0.25,
                 with_enhanced_dep_info=False,
                 with_target_semantic_labels=False,
                 with_reattach=False
                 ):
        self.max_len = max_len
        self.embeddings_dim = embeddings_dim
        self.train_epochs = train_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.with_enhanced_dep_info = with_enhanced_dep_info
        self.with_target_semantic_labels = with_target_semantic_labels
        self.with_reattach = with_reattach
