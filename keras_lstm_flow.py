import TrainingDataExtractor as tde
from deep_dynet import support
from os import listdir
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Flatten, TimeDistributed
from keras.models import Model
import sklearn.preprocessing

from os import path
import sys
sys.path.append(path.abspath('./stanford_parser'))

from stanford_parser.parser import Parser

# Dep parsing test
parser = Parser()

dependencies = parser.parseToStanfordDependencies("Most death sentences are for drug @-@ related offenses .")

tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
print tupleResult

# Load data
def read_data(type, dataset=None):
    data = []
    mypath = 'resources/alignments/split/' + type
    print(mypath)
    if dataset is None:
        for f in listdir(mypath):
            mypath_f = mypath + "/" + f
            print(mypath_f)
            data += tde.generate_training_data(mypath_f, False)
    else:
        mypath_f = mypath + "/" + dataset
        print(mypath_f)
        data = tde.generate_training_data(mypath_f, verbose=False, withDependencies=True)
    return data

data = read_data('dev', 'deft-p2-amr-r1-alignments-dev-bolt.txt')
#data = read_data('training', 'deft-p2-amr-r1-alignments-training-dfa.txt')

print "Data size %s" % len(data)

sentences = [d[0] for d in data]
amrs = [d[2] for d in data]

vocab_acts = support.Vocab.from_list(['SH', 'RL', 'RR', 'DN', 'SW'])
#action_sequence = support.oracle_actions_to_action_index(data[20][1], vocab_acts)

actions = [support.oracle_actions_to_action_index(d[1], vocab_acts) for d in data]

action_indices = [[a.index for a in actions_list] for actions_list in actions]
action_labels = [[a.label for a in actions_list] for actions_list in actions]

dependencies = [d[3] for d in data]

tokenizer = Tokenizer(filters="", lower=True, split=" ")
tokenizer.fit_on_texts(sentences)
sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

word_index = tokenizer.word_index
print len(word_index)

# Shuffle data
indices = np.arange(sequences.shape[0])
np.random.shuffle(indices)
sequences = sequences[indices]

actions = np.asarray(action_indices)[indices]
dependencies = [dependencies[i] for i in indices]

amrs = np.asanyarray(amrs)[indices]

num_train_samples = int(0.95 * sequences.shape[0])

x_train = sequences[:num_train_samples]
y_train = actions[:num_train_samples]
amrs_train = amrs[:num_train_samples]
dependencies_train = dependencies[:num_train_samples]

x_test = sequences[num_train_samples:]
y_test = actions[num_train_samples:]
amrs_test = amrs[num_train_samples:]
dependencies_test = dependencies[num_train_samples:]

print x_train.shape
print y_train.shape
print amrs_train.shape
print len(dependencies_train)

embeddings_index = {}
f = open('./resources/glove/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 2, EMBEDDING_DIM))
not_found = []
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        not_found.append(word)

print 'Not found: {}'.format(not_found)

# Prepare the proper data set:
# Input: Buffer top, First three elements on the stack, previous action index, stack[0] deps on stack[1], stack[1] deps on stack[0], stack[0] deps on buffer[0], buffer[0] deps on stack[1], stack[0] deps on stack[2], stack[2] deps on stack[0].
# If the current action is shift, the next action will have the next token in the buffer and updated stack elements.
# Else, the same element on the buffer is fed and the elements from the stack are updated
# Do not consider instances with more than 30 actions for the moment.

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(5))

SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NONE = 5

max_len = 30
no_word_index = (len(word_index)) + 1


def generate_dataset(x, y, dependencies):
    x_full = np.zeros((len(x), max_len, 15), dtype=np.int32)
    y_full = np.full((len(y), max_len), dtype=np.int32, fill_value=NONE)

    lengths = []
    filtered_count = 0

    for action_sequence, tokens_sequence, deps, i in zip(y, x, dependencies, range(len(y))):
        next_action_token = tokens_sequence[0]
        next_action_stack = [no_word_index, no_word_index, no_word_index, no_word_index]
        next_action_prev_action = NONE
        tokens_sequence_index = 0
        features_matrix = []
        lengths.append(len(action_sequence))
        if len(action_sequence) > 30:
            filtered_count += 1
            continue
        for action, j in zip(action_sequence, range(len(action_sequence))):
            if next_action_prev_action != NONE:
                next_action_prev_action_ohe = label_binarizer.transform([next_action_prev_action])[0, :]
            else:
                next_action_prev_action_ohe = [0, 0, 0, 0, 0]

            dep_0_on_1 = 0
            dep_1_on_0 = 0
            dep_0_on_2 = 0
            dep_2_on_0 = 0
            dep_0_on_b = 0
            dep_b_on_0 = 0
            if next_action_stack[0] in deps.keys() and deps[next_action_stack[0]][0] == next_action_stack[1]:
                dep_0_on_1 = 1
            if next_action_stack[1] in deps.keys() and deps[next_action_stack[1]][0] == next_action_stack[0]:
                dep_1_on_0 = 1
            if next_action_stack[0] in deps.keys() and deps[next_action_stack[0]][0] == next_action_stack[2]:
                dep_0_on_2 = 1
            if next_action_stack[2] in deps.keys() and deps[next_action_stack[2]][0] == next_action_stack[0]:
                dep_2_on_0 = 1
            if next_action_stack[0] in deps.keys() and deps[next_action_stack[0]][0] == next_action_token:
                dep_0_on_b = 1
            if next_action_token in deps.keys() and deps[next_action_token][0] == next_action_stack[0]:
                dep_b_on_0 = 1
            features = np.concatenate((np.asarray([next_action_token, next_action_stack[0],
                                                   next_action_stack[1], next_action_stack[2]]),
                                       next_action_prev_action_ohe,
                                       np.asarray(
                                           [dep_0_on_1, dep_1_on_0, dep_0_on_2, dep_2_on_0, dep_0_on_b, dep_b_on_0])))
            if action == SH:
                tokens_sequence_index += 1
                next_action_stack = [next_action_token] + next_action_stack
                if tokens_sequence_index < len(tokens_sequence):
                    next_action_token = tokens_sequence[tokens_sequence_index]
                else:
                    next_action_token = no_word_index
            if action == RL:
                next_action_stack = [next_action_stack[0]] + next_action_stack[2:]
            if action == RR:
                next_action_stack = [next_action_stack[1]] + next_action_stack[2:]
            if action == DN:
                tokens_sequence_index += 1
                if tokens_sequence_index < len(tokens_sequence):
                    next_action_token = tokens_sequence[tokens_sequence_index]
                else:
                    next_action_token = no_word_index
            if action == SW:
                next_action_stack = [next_action_stack[0], next_action_stack[2],
                                     next_action_stack[1]] + next_action_stack[3:]
            next_action_prev_action = action
            features_matrix.append(features)
        if tokens_sequence_index != len(tokens_sequence):
            raise Exception("There was a problem at training instance " + str(i) + "\n")

        features_matrix = np.concatenate((np.asarray(features_matrix),
                                          np.zeros((max_len - len(features_matrix), 15), dtype=np.int32)))
        actions = np.concatenate((np.asarray(action_sequence),
                                  np.full((max_len - len(action_sequence)), dtype=np.int32, fill_value=NONE)))
        x_full[i, :, :] = features_matrix
        y_full[i, :] = actions
    return x_full, y_full, lengths, filtered_count


(x_train_full, y_train_full, lengths_train, filtered_count_tr) = generate_dataset(x_train, y_train, dependencies_train)
(x_test_full, y_test_full, lengths_test, filtered_count_test) = generate_dataset(x_test, y_test, dependencies_test)

print "Mean length %s " % np.asarray(lengths_train).mean()
print (filtered_count_tr)
print (x_train_full.shape)
print (y_train_full.shape)

y_train_ohe = np.zeros((y_train.shape[0], max_len, 5), dtype='int32')
for row, i in zip(y_train_full[:, :], range(y_train_full.shape[0])):
    y_train_instance_matrix = label_binarizer.transform(row)
    y_train_ohe[i, :, :] = y_train_instance_matrix

buffer_input = Input(shape=(max_len,), dtype='int32')
stack_input_0 = Input(shape=(max_len,), dtype='int32')
stack_input_1 = Input(shape=(max_len,), dtype='int32')
stack_input_2 = Input(shape=(max_len,), dtype='int32')
prev_action_input = Input(shape=(max_len, 5), dtype='float32')
dep_info_input = Input(shape=(max_len, 6), dtype='float32')

embedding = Embedding(len(word_index) + 2,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)

buffer_emb = embedding(buffer_input)
stack_emb_0 = embedding(stack_input_0)
stack_emb_1 = embedding(stack_input_1)
stack_emb_2 = embedding(stack_input_2)

x = concatenate([buffer_emb, stack_emb_0, stack_emb_1, stack_emb_2, prev_action_input, dep_info_input])

lstm_output = LSTM(1024, return_sequences=True)(x)

dense = TimeDistributed(Dense(5, activation="softmax"))(lstm_output)

model = Model([buffer_input, stack_input_0, stack_input_1, stack_input_2, prev_action_input, dep_info_input], dense)


rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#plot_model(model, to_file='model.png')

print model.summary()

#model_path = 'models/proxy_model_with_deps'
model_path = 'models/dfa_model_with_deps'

model.fit([x_train_full[:, :, 0], x_train_full[:, :, 1], x_train_full[:, :, 2], x_train_full[:, :, 3], x_train_full[:, :, 4:9], x_train_full[:, :, 9:]],
         y_train_ohe,
         epochs=10, batch_size=16,
         validation_split=0.2,
         callbacks=[ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])


model.load_weights('models/proxy_model_with_deps', by_name=False)
index_to_word_map = {v: k for k, v in tokenizer.word_index.iteritems()}


def get_predictions_from_distr(predictions_distr):
    predictions = [np.argmax(p) for p in predictions_distr]
    return predictions


def pretty_print_predictions(predictions):
    actions = ['SH', 'RL', 'RR', 'DN', 'SW']
    for i in range(len(predictions)):
        print actions[predictions[i]], ;
    print '\n'


def pretty_print_sentence(tokens, index_to_word_map):
    for i in range(len(tokens)):
        print index_to_word_map[tokens[i]], ;
    print '\n'


def make_prediction(model, x_test, y_test):
    tokens_buffer = x_test
    tokens_stack = []
    current_step = 0
    buffer_token = np.zeros((1, max_len))
    stack_token0 = np.zeros((1, max_len))
    stack_token1 = np.zeros((1, max_len))
    stack_token2 = np.zeros((1, max_len))
    prev_action = np.zeros((1, max_len, 5))

    buffer_token[0][current_step] = tokens_buffer[0]
    stack_token0[0][current_step] = no_word_index
    stack_token1[0][current_step] = no_word_index
    stack_token2[0][current_step] = no_word_index
    prev_action[0][current_step] = [0, 0, 0, 0, 0]

    final_prediction = []
    while (len(tokens_buffer) != 0 or len(tokens_stack) != 1) and current_step < max_len - 1:
        prediction = model.predict([buffer_token, stack_token0, stack_token1, stack_token2, prev_action])
        current_actions_distr_ordered = np.argsort(prediction[0][current_step])[::-1]
        current_inspected_action_index = 0
        current_action = current_actions_distr_ordered[current_inspected_action_index]
        invalid = True
        while invalid:
            invalid = False
            current_action = current_actions_distr_ordered[current_inspected_action_index]
            current_inspected_action_index += 1
            if current_action == SH:
                if len(tokens_buffer) == 0:
                    invalid = True
                    continue
                tokens_stack = [tokens_buffer[0]] + tokens_stack
                tokens_buffer = tokens_buffer[1:]
            if current_action == RL:
                if len(tokens_stack) < 2:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[0]] + tokens_stack[2:]
            if current_action == RR:
                if len(tokens_stack) < 2:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[1]] + tokens_stack[2:]
            if current_action == DN:
                if len(tokens_buffer) == 0:
                    invalid = True
                    continue
                tokens_buffer = tokens_buffer[1:]
            if current_action == SW:
                if len(tokens_stack) < 3:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[0], tokens_stack[2], tokens_stack[1]] + tokens_stack[3:]
        final_prediction.append(current_action)
        current_step += 1
        if len(tokens_buffer) > 0:
            buffer_token[0][current_step] = tokens_buffer[0]
        else:
            buffer_token[0][current_step] = no_word_index

        if len(tokens_stack) > 3:
            stack_token0[0][current_step] = tokens_stack[0]
            stack_token1[0][current_step] = tokens_stack[1]
            stack_token2[0][current_step] = tokens_stack[2]
        else:
            if len(tokens_stack) > 2:
                stack_token0[0][current_step] = tokens_stack[0]
                stack_token1[0][current_step] = tokens_stack[1]
                stack_token2[0][current_step] = no_word_index
            else:
                if len(tokens_stack) > 1:
                    stack_token0[0][current_step] = tokens_stack[0]
                    stack_token1[0][current_step] = no_word_index
                    stack_token2[0][current_step] = no_word_index
                else:
                    stack_token0[0][current_step] = no_word_index
                    stack_token1[0][current_step] = no_word_index
                    stack_token2[0][current_step] = no_word_index
        prev_action[0][current_step] = label_binarizer.transform([current_action])[0, :]
    print 'Buffer and stack at end of prediction'
    print tokens_buffer
    print tokens_stack
    return final_prediction


for i in range(10):
    prediction = make_prediction(model, x_test[i], y_test[i])
    print 'Predicted'
    pretty_print_predictions(prediction)
    print 'Actual'
    pretty_print_predictions(y_test[i])
    print 'Sentence'
    pretty_print_sentence(x_test[i], index_to_word_map)
    print 'Amr'
    print amrs_test[i]