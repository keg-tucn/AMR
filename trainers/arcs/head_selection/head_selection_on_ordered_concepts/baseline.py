from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import SGD_Trainer

NO_EPOCHS = 20
MLP_DROPOUT = 0.5
MAX_PARENT_VECTORS = 6
REENTRANCY_THRESHOLD = 0.6
PREPROCESSING = True
TRAINABLE_EMB_SIZE = 50
GLOVE_EMB_SIZE = 100
LSTM_OUT_DIM = 50
MLP_DIM = 32
NO_LSTM_LAYERS = 1
CHAR_CNN_CUTOFF = 5
USE_VERB_FLAG = True
TRAINER = SGD_Trainer