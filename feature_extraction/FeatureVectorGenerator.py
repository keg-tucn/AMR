import logging
import numpy as np
import sklearn
import pickle

from definitions import PROJECT_ROOT_DIR
from constants import __AMR_RELATIONS
import models.Actions as act

tokenizer_path = PROJECT_ROOT_DIR + "/tokenizers/full_tokenizer_extended.dump"

SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NONE = 5

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(5))

composed_label_binarizer = sklearn.preprocessing.LabelBinarizer()
composed_label_binarizer.fit(range(5 + 2 * len(__AMR_RELATIONS)))


def extract_data_components(train_data, test_data):
    data = np.concatenate((train_data, test_data))

    sentences = [d.sentence for d in data]

    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    actions = [d.action_sequence for d in data]

    dependencies = [d.dependencies for d in data]

    named_entities = [d.named_entities for d in data]
    named_entities = [[(n[3], n[2]) for n in named_entities_list] for named_entities_list in named_entities]

    date_entities = [d.date_entities for d in data]
    date_entities = [[(d[3], d[2], d[1]) for d in date_entities_list] for date_entities_list in date_entities]

    amr_ids = [d.amr_id for d in data]

    word_index = tokenizer.word_index

    num_test_samples = int(max(len(test_data), len(train_data) / 10))
    num_train_samples = int(len(data) - num_test_samples)

    x_train = sequences[:num_train_samples]
    y_train = actions[:num_train_samples]

    x_test = sequences[num_train_samples:]
    y_test = actions[num_train_samples:]

    dependencies_train = dependencies[:num_train_samples]
    dependencies_test = dependencies[num_train_samples:]

    train_amr_ids = amr_ids[:num_train_samples]
    test_amr_ids = amr_ids[num_train_samples:]

    return x_train, y_train, x_test, y_test, dependencies_train, dependencies_test,\
           train_amr_ids, test_amr_ids, named_entities, date_entities, word_index


def generate_feature_vectors(x, y, dependencies, amr_ids, max_len, index_to_word_map, no_word_index):
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
    y_temp = []
    i = 0

    for action_sequence, tokens_sequence, dependencies, amr_id in zip(y, x, dependencies, amr_ids):
        next_action_token = tokens_sequence[0]
        next_action_stack = [no_word_index, no_word_index, no_word_index, no_word_index]
        next_action_prev_action = NONE
        tokens_sequence_index = 0
        features_matrix = []

        if len(action_sequence) > max_len:
            continue

        for action, j in zip(action_sequence, range(len(action_sequence))):
            action = action.index
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
            if next_action_stack[0] in dependencies.keys() and dependencies[next_action_stack[0]][0] == next_action_stack[1]:
                dep_0_on_1 = 1
            if next_action_stack[1] in dependencies.keys() and dependencies[next_action_stack[1]][0] == next_action_stack[0]:
                dep_1_on_0 = 1
            if next_action_stack[0] in dependencies.keys() and dependencies[next_action_stack[0]][0] == next_action_stack[2]:
                dep_0_on_2 = 1
            if next_action_stack[2] in dependencies.keys() and dependencies[next_action_stack[2]][0] == next_action_stack[0]:
                dep_2_on_0 = 1
            if next_action_stack[0] in dependencies.keys() and dependencies[next_action_stack[0]][0] == next_action_token:
                dep_0_on_b = 1
            if next_action_token in dependencies.keys() and dependencies[next_action_token][0] == next_action_stack[0]:
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

        actions = action_sequence
        x_full[i, :, :] = features_matrix
        y_temp.append(actions)
        i += 1
    logging.warning("Exception count " + str(exception_count))

    #y_ohe = np.zeros((y_full.shape[0], max_len, 5 + 2 * len(__AMR_RELATIONS)), dtype='int32')
    y_ohe = np.zeros((y_full.shape[0], max_len, 5), dtype='int32')

    for row, i in zip(y_temp, range(len(y_temp))):
        y_train_instance_matrix = []
        for r in row:
            if r.index == -10:
                y_train_instance_matrix.append(
                    composed_label_binarizer.transform([5 + __AMR_RELATIONS.index(r.label)])[0, :])
            elif r.index == -20:
                y_train_instance_matrix.append(
                    composed_label_binarizer.transform([5 + len(__AMR_RELATIONS) + __AMR_RELATIONS.index(r.label)])[0,
                    :])
            else:
                #y_train_instance_matrix.append(composed_label_binarizer.transform([r.index])[0, :])
                y_train_instance_matrix.append(label_binarizer.transform([r.index])[0, :])
        for j in range(max_len - len(row)):
            #y_train_instance_matrix.append(composed_label_binarizer.transform([1000])[0, :])
            y_train_instance_matrix.append(label_binarizer.transform([1000])[0, :])
        y_ohe[i, :, :] = y_train_instance_matrix

    return x_full, y_ohe, lengths, filtered_count


def actions_to_string(acts_i):
    str = ""
    for a in acts_i:
        str += act.acts[a] + " "
    str += "\n"
    return str


def tokens_to_sentence(tokens, index_to_word_map):
    str = ""
    for t in tokens:
        str += index_to_word_map[t] + " "
    str += "\n"
    return str
