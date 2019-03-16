import pickle
import sys
from os import path
import numpy as np
import re
import sklearn.preprocessing
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from Baseline import reentrancy_restoring
import logging

from amr_util.KerasPlotter import plot_history
from postprocessing import ActionSequenceReconstruction as asr
from smatch import smatch_amr
from smatch import smatch_util
import models.Actions as act
from amr_reader import read_data as ra

sys.path.append(path.abspath('./stanford_parser'))

SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NONE = 5
coref_handling = False
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(5))


def read_sentence(type):
    return [d[0] for d in read_data(type)]


def read_sentence_test(type):
    return [d[0] for d in read_test_data(type)]


def read_test_data(type, dataset=None):
    # make sure testing done on same data subset
    return ra(type, True, dataset)


# Load data
def read_data(type, dataset=None, cache=False):
    return ra(type, cache, dataset)


def get_predictions_from_distr(predictions_distr):
    predictions = [np.argmax(p) for p in predictions_distr]
    return predictions


def pretty_print_actions(acts_i):
    print actions_to_string(acts_i)


def actions_to_string(acts_i):
    str = ""
    for a in acts_i:
        str += act.acts[a] + " "
    str += "\n"
    return str


def pretty_print_sentence(tokens, index_to_word_map):
    print tokens_to_sentence(tokens, index_to_word_map)


def tokens_to_sentence(tokens, index_to_word_map):
    str = ""
    for t in tokens:
        str += index_to_word_map[t] + " "
    str += "\n"
    return str


def make_prediction(model, x_test, deps, no_word_index, max_len):
    tokens_buffer = x_test
    tokens_stack = []
    current_step = 0
    buffer_token = np.zeros((1, max_len))
    stack_token0 = np.zeros((1, max_len))
    stack_token1 = np.zeros((1, max_len))
    stack_token2 = np.zeros((1, max_len))
    prev_action = np.zeros((1, max_len, 5))
    dep_info = np.zeros((1, max_len, 6))

    buffer_token[0][current_step] = tokens_buffer[0]
    stack_token0[0][current_step] = no_word_index
    stack_token1[0][current_step] = no_word_index
    stack_token2[0][current_step] = no_word_index
    prev_action[0][current_step] = [0, 0, 0, 0, 0]
    dep_info[0][current_step] = [0, 0, 0, 0, 0, 0]

    final_prediction = []
    while (len(tokens_buffer) != 0 or len(tokens_stack) != 1) and current_step < max_len - 1:
        prediction = model.predict([buffer_token, stack_token0, stack_token1, stack_token2, prev_action, dep_info])
        current_actions_distr_ordered = np.argsort(prediction[0][current_step])[::-1]
        current_inspected_action_index = 0
        current_action = current_actions_distr_ordered[current_inspected_action_index]
        invalid = True
        while invalid:
            if current_inspected_action_index == 5:
                return []
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

        dep_0_on_1 = 0
        dep_1_on_0 = 0
        dep_0_on_2 = 0
        dep_2_on_0 = 0
        dep_0_on_b = 0
        dep_b_on_0 = 0

        if stack_token0[0][current_step] in deps.keys() and deps[stack_token0[0][current_step]][0] == stack_token1[0][
            current_step]:
            dep_0_on_1 = 1
        if stack_token1[0][current_step] in deps.keys() and deps[stack_token1[0][current_step]][0] == stack_token0[0][
            current_step]:
            dep_1_on_0 = 1
        if stack_token0[0][current_step] in deps.keys() and deps[stack_token0[0][current_step]][0] == stack_token2[0][
            current_step]:
            dep_0_on_2 = 1
        if stack_token2[0][current_step] in deps.keys() and deps[stack_token2[0][current_step]][0] == stack_token0[0][
            current_step]:
            dep_2_on_0 = 1
        if stack_token0[0][current_step] in deps.keys() and deps[stack_token0[0][current_step]][0] == buffer_token[0][
            current_step]:
            dep_0_on_b = 1
        if buffer_token[0][current_step] in deps.keys() and deps[buffer_token[0][current_step]][0] == stack_token0[0][
            current_step]:
            dep_b_on_0 = 1
        dep_info[0][current_step] = [dep_0_on_1, dep_1_on_0, dep_0_on_2, dep_2_on_0, dep_0_on_b, dep_b_on_0]
    print 'Buffer and stack at end of prediction'
    print tokens_buffer
    print tokens_stack
    return final_prediction


def generate_dataset(x, y, dependencies, no_word_index, max_len, amr_ids, index_to_word_map):
    lengths = []
    filtered_count = 0
    exception_count = 0
    for action_sequence in y:
        lengths.append(len(action_sequence))
        if len(action_sequence) > max_len:
            filtered_count += 1
            continue

    x_full = np.zeros((len(x) - filtered_count, max_len, 15), dtype=np.int32)
    y_full = np.full((len(y) - filtered_count, max_len), dtype=np.int32, fill_value=NONE)
    i = 0

    for action_sequence, tokens_sequence, deps, amr_id in zip(y, x, dependencies, amr_ids):
        next_action_token = tokens_sequence[0]
        next_action_stack = [no_word_index, no_word_index, no_word_index, no_word_index]
        next_action_prev_action = NONE
        tokens_sequence_index = 0
        features_matrix = []

        if len(action_sequence) > max_len:
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
            logging.warn("There was a problem at training instance %d at %s. Actions %s. Tokens %s", i, amr_id,
                         actions_to_string(action_sequence), tokens_to_sentence(tokens_sequence, index_to_word_map))
            exception_count += 1
            continue
            # raise Exception("There was a problem at training instance " + str(i) + " at " + amr_id + "\n")

        features_matrix = np.concatenate((np.asarray(features_matrix),
                                          np.zeros((max_len - len(features_matrix), 15), dtype=np.int32)))
        actions = np.concatenate((np.asarray(action_sequence),
                                  np.full((max_len - len(action_sequence)), dtype=np.int32, fill_value=NONE)))
        x_full[i, :, :] = features_matrix
        y_full[i, :] = actions
        i += 1
    logging.warning("Exception count " + str(exception_count))
    return x_full, y_full, lengths, filtered_count


def generate_tokenizer(tokenizer_path):
    test_data = read_sentence_test('test')
    train_data = read_sentence('training')
    dev_data = read_sentence_test('dev')
    sentences = test_data + train_data + dev_data
    tokenizer = Tokenizer(filters="", lower=True, split=" ")
    tokenizer.fit_on_texts(sentences)
    pickle.dump(tokenizer, open(tokenizer_path, "wb"))


def get_optimizer():
    return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def get_embedding_matrix(word_index, embedding_dim=100):
    special_cases_re = re.compile('''^([a-z])+-(?:entity|quantity)$''')
    embeddings_index = {}
    f = open('./resources/glove/glove.6B.{}d.txt'.format(embedding_dim))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 2, embedding_dim))
    not_found = []
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            match = re.match(special_cases_re, word)
            if match:
                print 'Embedding match for {}'.format(word)
                embedding_vector = embeddings_index.get(match.group(1))
            else:
                not_found.append(word)

    print 'First 2 not found: {}'.format(not_found[2:4])
    return embedding_matrix


def get_model(word_index, max_len, embedding_dim, embedding_matrix):
    buffer_input = Input(shape=(max_len,), dtype='int32')
    stack_input_0 = Input(shape=(max_len,), dtype='int32')
    stack_input_1 = Input(shape=(max_len,), dtype='int32')
    stack_input_2 = Input(shape=(max_len,), dtype='int32')
    prev_action_input = Input(shape=(max_len, 5), dtype='float32')
    dep_info_input = Input(shape=(max_len, 6), dtype='float32')

    embedding = Embedding(len(word_index) + 2,
                          embedding_dim,
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

    optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # plot_model(model, to_file='model.png')

    print model.summary()
    return model


def train(model_name, tokenizer_path, train_data, test_data, max_len=30, train_epochs=35, embedding_dim=100):
    model_path = './trained_models/{}'.format(model_name)
    print 'Model path is:'
    print model_path

    data = train_data + test_data

    print "Data set total size %s" % len(data)

    sentences = [d[0] for d in data]
    amrs = [d[2] for d in data]

    actions = [d[1] for d in data]

    action_indices = [[a.index for a in actions_list] for actions_list in actions]
    action_labels = [[a.label for a in actions_list] for actions_list in actions]

    dependencies = [d[3] for d in data]

    named_entities = [d[4] for d in data]
    date_entities = [d[5] for d in data]
    named_entities = [[(n[3], n[2]) for n in named_entities_list] for named_entities_list in named_entities]
    date_entities = [[(d[3], d[2], d[1]) for d in date_entities_list] for date_entities_list in date_entities]
    train_amr_ids = [d[7] for d in train_data]
    test_amr_ids = [d[7] for d in test_data]

    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    word_index = tokenizer.word_index
    print 'Word index len: '
    print len(word_index)

    indices = np.arange(sequences.shape[0])
    sequences = sequences[indices]

    actions = np.asarray(action_indices)[indices]
    labels = np.asarray(action_labels)[indices]
    dependencies = [dependencies[i] for i in indices]

    amrs = np.asanyarray(amrs)[indices]

    num_test_samples = int(max(len(test_data), 0.1 * len(train_data)))
    num_train_samples = int(len(data) - num_test_samples)

    x_train = sequences[:num_train_samples]
    y_train = actions[:num_train_samples]
    amrs_train = amrs[:num_train_samples]
    dependencies_train = dependencies[:num_train_samples]

    x_test = sequences[num_train_samples:]
    y_test = actions[num_train_samples:]
    l_test = labels[num_train_samples:]
    amrs_test = amrs[num_train_samples:]
    dependencies_test = dependencies[num_train_samples:]

    print 'Training data shape: '
    print x_train.shape

    print 'Test data shape: '
    print x_test.shape

    # Prepare the proper data set:
    # Input: Buffer top, First three elements on the stack, previous action index, stack[0] deps on stack[1],
    # stack[1] deps on stack[0], stack[0] deps on buffer[0], buffer[0] deps on stack[1], stack[0] deps on stack[2],
    # stack[2] deps on stack[0].
    # If the current action is shift, the next action will have the next token in the buffer and updated stack elements.
    # Else, the same element on the buffer is fed and the elements from the stack are updated
    # Do not consider instances with more than 30 actions for the moment.
    embedding_matrix = get_embedding_matrix(word_index, embedding_dim)
    no_word_index = (len(word_index)) + 1

    index_to_word_map = {v: k for k, v in tokenizer.word_index.iteritems()}

    (x_train_full, y_train_full, lengths_train, filtered_count_tr) = generate_dataset(x_train, y_train,
                                                                                      dependencies_train, no_word_index,
                                                                                      max_len, train_amr_ids,
                                                                                      index_to_word_map)
    (x_test_full, y_test_full, lengths_test, filtered_count_test) = generate_dataset(x_test, y_test, dependencies_test,
                                                                                     no_word_index, max_len,
                                                                                     test_amr_ids, index_to_word_map)

    print "Mean length %s " % np.asarray(lengths_train).mean()
    print "Max length %s" % np.asarray(lengths_train).max()
    print "Filtered"
    print (filtered_count_tr)
    print "Final train data shape"
    print (x_train_full.shape)
    print "Final test data shape"
    print (x_test_full.shape)

    y_train_ohe = np.zeros((y_train_full.shape[0], max_len, 5), dtype='int32')
    for row, i in zip(y_train_full[:, :], range(y_train_full.shape[0])):
        y_train_instance_matrix = label_binarizer.transform(row)
        y_train_ohe[i, :, :] = y_train_instance_matrix

    y_test_ohe = np.zeros((y_test_full.shape[0], max_len, 5), dtype='int32')
    for row, i in zip(y_test_full[:, :], range(y_test_full.shape[0])):
        y_test_instance_matrix = label_binarizer.transform(row)
        y_test_ohe[i, :, :] = y_test_instance_matrix

    model = get_model(word_index, max_len, embedding_dim, embedding_matrix)

    history = model.fit([x_train_full[:, :, 0], x_train_full[:, :, 1], x_train_full[:, :, 2], x_train_full[:, :, 3],
                         x_train_full[:, :, 4:9], x_train_full[:, :, 9:]],
                        y_train_ohe,
                        epochs=train_epochs, batch_size=16,
                        validation_split=0.1,
                        callbacks=[
                            ModelCheckpoint(model_path, monitor='val_acc', verbose=0, save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto', period=1),
                            EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0,
                                          mode='auto')])

    plot_history(history, model_name)

    model.load_weights(model_path, by_name=False)

    smatch_results = smatch_util.SmatchAccumulator()
    errors = 0
    for i in range(len(x_test)):
        # Step1: input a processed test entity test
        prediction = make_prediction(model, x_test[i], dependencies_test[i], no_word_index, max_len)

        if len(prediction) > 0:
            act = asr.ActionConceptTransfer()
            act.load_from_action_and_label(y_test[i], l_test[i])
            pred_label = act.populate_new_actions(prediction)
            print 'Predictions with old labels: '
            print pred_label

            # Step2: output: Graph respecting the predicted structure
            # Step2': predict concepts
            # Step2'': predict relations
            # Step3: replace named entitities & date date_entities

            predicted_amr_str = asr.reconstruct_all_ne(pred_label, named_entities[i], date_entities[i])

            # handling coreference(postprocessing)
            if coref_handling:
                predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            # Step4: compute smatch
            original_amr = smatch_amr.AMR.parse_AMR_line(amrs_test[i])
            predicted_amr = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)
            smatch_f_score = smatch_results.compute_and_add(predicted_amr, original_amr)

            print 'Original Amr'
            print amrs_test[i]
            print 'Predicted Amr'
            print predicted_amr_str
            print 'Smatch f-score %f' % smatch_f_score
        else:
            errors += 1

    # modified to results_train_new
    file = open('./results_keras/{}_results_train_new'.format(model_name), 'w')

    file.write('------------------------------------------------------------------------------------------------\n')
    file.write('Train data shape: \n')
    file.write(str(x_train.shape) + '\n')
    file.write('Test data shape: ' + '\n')
    file.write(str(x_test.shape) + '\n')
    file.write("Final test data shape" + '\n')
    file.write(str(x_test_full.shape) + '\n')
    file.write("Final train data shape" + '\n')
    file.write(str(x_train_full.shape) + '\n')
    file.write("Mean train length %s \n" % np.asarray(lengths_train).mean())
    file.write("Max train length %s \n" % np.asarray(lengths_train).max())
    file.write("Filtered\n")
    file.write(str(filtered_count_tr) + '\n')
    file.write('Scores for model {}\n'.format(model_path))

    file.write("Min: %f\n" % np.min(smatch_results.smatch_scores))
    file.write("Max: %f\n" % np.max(smatch_results.smatch_scores))
    file.write("Arithm. mean %s\n" % (smatch_results.smatch_sum / smatch_results.n))
    file.write("Harm. mean %s\n" % (smatch_results.n / smatch_results.inv_smatch_sum))
    file.write("Global smatch f-score %s\n" % smatch_results.smatch_per_node_mean())

    accuracy = model.evaluate([x_test_full[:, :, 0], x_test_full[:, :, 1], x_test_full[:, :, 2], x_test_full[:, :, 3],
                               x_test_full[:, :, 4:9], x_test_full[:, :, 9:]], y_test_ohe)
    file.write('Model test accuracy\n')
    file.write(str(accuracy[1]) + '\n')
    file.write('Errors\n')
    file.write(str(errors) + '\n')

    file.close()


def test(model_name, tokenizer_path, test_case_name, data, max_len=30, embedding_dim=100, with_reattach=False):
    model_path = './trained_models/{}'.format(model_name)
    print 'Model path is:'
    print model_path

    sentences = [d[0] for d in data]
    amrs = [d[2] for d in data]
    test_amr_ids = [d[7] for d in data]

    actions = [d[1] for d in data]

    action_indices = [[a.index for a in actions_list] for actions_list in actions]
    action_labels = [[a.label for a in actions_list] for actions_list in actions]

    dependencies = [d[3] for d in data]

    named_entities = [d[4] for d in data]
    date_entities = [d[5] for d in data]
    named_entities = [[(n[3], n[2]) for n in named_entities_list] for named_entities_list in named_entities]
    date_entities = [[(d[3], d[2], d[1]) for d in date_entities_list] for date_entities_list in date_entities]

    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    word_index = tokenizer.word_index
    print 'Word index len: '
    print len(word_index)

    indices = np.arange(sequences.shape[0])
    sequences = sequences[indices]

    actions = np.asarray(action_indices)[indices]
    labels = np.asarray(action_labels)[indices]
    dependencies = [dependencies[i] for i in indices]

    amrs = np.asanyarray(amrs)[indices]

    x_test = sequences
    y_test = actions
    l_test = labels
    amrs_test = amrs
    dependencies_test = dependencies

    print 'Test data shape: '
    print x_test.shape
    print y_test.shape
    print amrs_test.shape
    print len(dependencies_test)

    index_to_word_map = {v: k for k, v in tokenizer.word_index.iteritems()}

    embedding_matrix = get_embedding_matrix(word_index, embedding_dim)

    no_word_index = (len(word_index)) + 1

    (x_test_full, y_test_full, lengths_test, filtered_count_test) = generate_dataset(x_test, y_test, dependencies_test,
                                                                                     no_word_index, max_len,
                                                                                     test_amr_ids, index_to_word_map)

    y_test_ohe = np.zeros((y_test_full.shape[0], max_len, 5), dtype='int32')
    for row, i in zip(y_test_full[:, :], range(y_test_full.shape[0])):
        y_test_instance_matrix = label_binarizer.transform(row)
        y_test_ohe[i, :, :] = y_test_instance_matrix

    model = get_model(word_index, max_len, embedding_dim, embedding_matrix)

    print model.summary()

    model.load_weights(model_path, by_name=False)

    smatch_results = smatch_util.SmatchAccumulator()
    predictions = []
    errors = 0
    for i in range(len(x_test)):
        prediction = make_prediction(model, x_test[i], dependencies_test[i], no_word_index, max_len)
        predictions.append(prediction)
        print 'Sentence'
        pretty_print_sentence(x_test[i], index_to_word_map)
        print 'Predicted'
        pretty_print_actions(prediction)
        print 'Actual'
        pretty_print_actions(y_test[i])

        if len(prediction) > 0:
            act = asr.ActionConceptTransfer()
            act.load_from_action_and_label(y_test[i], l_test[i])
            pred_label = act.populate_new_actions(prediction)
            print 'Predictions with old labels: '
            print pred_label
            if with_reattach is True:
                predicted_amr_str = asr.reconstruct_all_ne(pred_label, named_entities[i], date_entities[i])
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)
            else:
                predicted_amr_str = asr.reconstruct_all(pred_label)
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            original_amr = smatch_amr.AMR.parse_AMR_line(amrs_test[i])
            predicted_amr = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)
            smatch_f_score = smatch_results.compute_and_add(predicted_amr, original_amr)

            print 'Original Amr'
            print amrs_test[i]
            print 'Predicted Amr'
            print predicted_amr_str
            print 'Smatch f-score %f' % smatch_f_score
        else:
            errors += 1

    file = open('./results_keras/{}_results_test_{}'.format(model_name, test_case_name), 'w')

    file.write('------------------------------------------------------------------------------------------------\n')
    file.write('Test data shape: ' + '\n')
    file.write(str(x_test.shape) + '\n')
    file.write("Final test data shape" + '\n')
    file.write(str(x_test_full.shape) + '\n')
    file.write("Filtered\n")
    file.write(str(filtered_count_test) + '\n')
    file.write('Scores for model {}\n'.format(model_path))

    file.write("Min: %f\n" % np.min(smatch_results.smatch_scores))
    file.write("Max: %f\n" % np.max(smatch_results.smatch_scores))
    file.write("Arithm. mean %s\n" % (smatch_results.smatch_sum / smatch_results.n))
    file.write("Harm. mean %s\n" % (smatch_results.n / smatch_results.inv_smatch_sum))
    file.write("Global smatch f-score %s\n" % smatch_results.smatch_per_node_mean())

    accuracy = model.evaluate([x_test_full[:, :, 0], x_test_full[:, :, 1], x_test_full[:, :, 2], x_test_full[:, :, 3],
                               x_test_full[:, :, 4:9], x_test_full[:, :, 9:]], y_test_ohe)
    file.write('Model test accuracy\n')
    file.write(str(accuracy[1]) + '\n')
    file.write('Errors\n')
    file.write(str(errors) + '\n')

    file.close()
    return predictions


def test_without_amr(model_name, tokenizer_path, data, max_len=30, embedding_dim=100, with_reattach=False):
    model_path = './trained_models/{}'.format(model_name)
    print 'Model path is:'
    print model_path

    sentences = [d[0] for d in data]

    dependencies = [d[1] for d in data]

    named_entities = [d[2] for d in data]

    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    word_index = tokenizer.word_index
    print 'Word index len: '
    print len(word_index)

    indices = np.arange(sequences.shape[0])
    sequences = sequences[indices]

    dependencies = [dependencies[i] for i in indices]

    x_test = sequences
    dependencies_test = dependencies

    print 'Test data shape: '
    print x_test.shape
    print len(dependencies_test)

    embedding_matrix = get_embedding_matrix(word_index, embedding_dim)

    no_word_index = (len(word_index)) + 1
    model = get_model(word_index, max_len, embedding_dim, embedding_matrix)

    print model.summary()

    model.load_weights(model_path, by_name=False)
    index_to_word_map = {v: k for k, v in tokenizer.word_index.iteritems()}

    for i in range(len(x_test)):
        prediction = make_prediction(model, x_test[i], dependencies_test[i], no_word_index, max_len)
        print 'Sentence'
        pretty_print_sentence(x_test[i], index_to_word_map)
        print 'Predicted'
        pretty_print_actions(prediction)

        if len(prediction) > 0:
            act = asr.ActionConceptTransfer()
            pred_label = act.populate_new_actions(prediction)
            print 'AMR skeleton without labels: '
            print pred_label

            if with_reattach is True:
                predicted_amr_str = asr.reconstruct_all_ne(pred_label, named_entities, [])
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)
            else:
                predicted_amr_str = asr.reconstruct_all(pred_label)
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            print 'Predicted Amr'
            print predicted_amr_str

    return prediction


def train_file(model_name, tokenizer_path, train_data_path=None, test_data_path=None, max_len=30, train_epochs=35,
               embedding_dim=100):
    test_data = read_test_data('test', test_data_path)
    train_data = read_data('training', train_data_path, cache=True)

    train(model_name, tokenizer_path, train_data, test_data, max_len, train_epochs, embedding_dim)


def test_file(model_name, tokenizer_path, test_case_name, test_data_path, max_len=30, embedding_dim=100,
              test_source="test",
              with_reattach=False):
    data = read_test_data(test_source, test_data_path)
    return test(model_name, tokenizer_path, test_case_name, data, max_len, embedding_dim, with_reattach=with_reattach)


if __name__ == "__main__":
    data_sets = ['xinhua', 'bolt', 'proxy', 'dfa', 'all']
    max_lens = [30, 30, 30, 30, 30]
    embeddings_dims = [200, 200, 300, 200, 200]
    epochs = [50, 50, 50, 50, 20]
    test_source = 'dev'

    # for data_set in data_sets:
    #     for embeddings_dim in embeddings_dims:
    #         for max_len in max_lens:
    for data_set, max_len, embeddings_dim, epoch in zip(data_sets, max_lens, embeddings_dims, epochs):
        # epochs = 20
        model_name = '{}_epochs={}_maxlen={}_embeddingsdim={}'.format(data_set, epoch, max_len, embeddings_dim)
        # if data_set == "all":
        test_set_name = None
        # else:
        #     test_set_name = 'deft-p2-amr-r1-alignments-test-{}.txt'.format(data_set)
        print 'Model name is: '
        print model_name
        model_path = './trained_models/{}'.format(model_name)

        if data_set == "all":
            train_data_path = None
            test_data_path = None
        else:
            train_data_path = data_set
            test_data_path = data_set
    tokenizer_path = "./tokenizers/full_tokenizer_extended.dump"
    generate_tokenizer(tokenizer_path)
    train_file(model_name=model_name,
               tokenizer_path=tokenizer_path,
               train_data_path=train_data_path,
               test_data_path=test_data_path, max_len=30,
               train_epochs=1, embedding_dim=100)
    #     test_file(model_name, tokenizer_path="./tokenizers/full_tokenizer.dump",
    #               test_case_name= test_source,
    #               test_data_path=test_set_name, max_len=max_len,
    #               embedding_dim=embeddings_dim, test_source="dev", with_reattach=True)
