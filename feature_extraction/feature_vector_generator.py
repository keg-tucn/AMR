import logging

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from amr_util import tokenizer_util
from constants import __AMR_RELATIONS, __DEP_AMR_REL_TABLE
from .feature_extraction_exceptions import InvalidParseException
from models.actions import *
from models.parameters import *

simple_target_label_binarizer = None
composed_target_label_binarizer = None
amr_rel_binarizer = None


def extract_data_components(data):
    """
    Return the components of a list of TrainData instances as separate arrays
    :param data: array of TrainData instances
    :return: arrays for sequences of indices (corresponding to sentence words), actions, dependencies, AMRs as
            strings, AMR IDs, named entities pairs and date entities pairs
    """
    sentences = np.asarray([d.sentence for d in data])

    tokenizer = tokenizer_util.get_tokenizer()
    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    actions = np.asarray([d.action_sequence for d in data])

    dependencies = np.asarray([d.dependencies for d in data])

    named_entities = [d.named_entities for d in data]
    named_entities = [[(n[3], n[2]) for n in named_entities_list] for named_entities_list in named_entities]

    date_entities = [d.date_entities for d in data]
    date_entities = [[(d[3], d[2], d[1]) for d in date_entities_list] for date_entities_list in date_entities]

    amr_str = np.asarray([d.original_amr for d in data])

    amr_ids = np.asarray([d.amr_id for d in data])

    return sequences, actions, dependencies, amr_str, amr_ids, named_entities, date_entities


def generate_feature_vectors(x, y, dependencies, amr_ids, parser_parameters):
    """
    Return the encoded feature vectors for the trainer, for each training instance
    :param x: list of sequences of indices corresponding to the words of the sentences
    :param y: list of sequences of AMRAction objects
    :param dependencies: list of dictionaries of dependencies
    :param amr_ids: list of strings corresponding to the IDs of the AMRs in the data set
    :param parser_parameters: collection of model properties
    :return: list of encoded and padded features for the trainer
    """
    word_index_map = tokenizer_util.get_word_index_map()
    no_word_index = tokenizer_util.get_no_word_index()

    max_len = parser_parameters.max_len
    lengths = []
    filtered_count = 0
    exception_count = 0

    for action_sequence in y:
        lengths.append(len(action_sequence))
        if len(action_sequence) > parser_parameters.max_len:
            filtered_count += 1
            continue

    model_parameters = parser_parameters.model_parameters
    no_buffer_tokens = model_parameters.no_buffer_tokens
    no_stack_tokens = model_parameters.no_stack_tokens
    no_dep_features = model_parameters.no_dep_features
    action_set_size = ActionSet.action_set_size()

    input_size = no_buffer_tokens + no_stack_tokens + action_set_size
    if parser_parameters.with_enhanced_dep_info:
        input_size += no_dep_features * len(__AMR_RELATIONS)
    else:
        input_size += no_dep_features * 1

    x_full = np.zeros((len(x) - filtered_count, max_len, input_size), dtype=np.int32)

    output_size = action_set_size
    if parser_parameters.with_target_semantic_labels:
        output_size += 2 * len(__AMR_RELATIONS)

    y_full = np.zeros((len(y) - filtered_count, max_len, output_size), dtype=np.int32)

    action_sequences = []
    i = 0

    for action_sequence, tokens_sequence, dependencies, amr_id in zip(y, x, dependencies, amr_ids):
        next_action_buffer = tokens_sequence
        next_action_stack = list(np.repeat(no_word_index, no_stack_tokens))
        next_action_prev_action = AMRAction.build("NONE")

        features_matrix = []

        if len(action_sequence) > max_len:
            continue

        try:
            for action, j in zip(action_sequence, list(range(len(action_sequence)))):
                next_action_prev_action_ohe = simple_target_label_binarizer.transform([next_action_prev_action.index])[
                                              0, :]

                if len(next_action_buffer) < no_buffer_tokens:
                    buffer_features = next_action_buffer + [no_word_index] * (
                            no_buffer_tokens - len(next_action_buffer))
                else:
                    buffer_features = next_action_buffer[0:no_buffer_tokens]

                features = np.concatenate((
                    buffer_features,
                    next_action_stack[0:no_stack_tokens],
                    next_action_prev_action_ohe,
                    get_dependency_features(next_action_stack[0], next_action_stack[1], next_action_stack[2],
                                            next_action_buffer[0] if len(next_action_buffer) else no_word_index,
                                            dependencies, parser_parameters)
                ))

                if action.action == "SH":
                    if not next_action_buffer:
                        raise InvalidParseException("Error parsing sentence for AMR with ID: %s" % amr_id)
                    next_action_stack = [next_action_buffer[0]] + next_action_stack
                    next_action_buffer = next_action_buffer[1:]

                if action.action == "RL":
                    next_action_stack = [next_action_stack[0]] + next_action_stack[2:]

                if action.action == "RR":
                    next_action_stack = [next_action_stack[1]] + next_action_stack[2:]

                if action.action == "DN":
                    if not next_action_buffer:
                        raise InvalidParseException("Error parsing sentence for AMR with ID: %s" % amr_id)
                    next_action_buffer = next_action_buffer[1:]

                if action.action == "SW":
                    next_action_stack = [next_action_stack[0], next_action_stack[2],
                                         next_action_stack[1]] + next_action_stack[3:]

                if action.action == "SW_2":
                    next_action_stack = [next_action_stack[0], next_action_stack[3],
                                         next_action_stack[2], next_action_stack[1]] + next_action_stack[4:]

                if action.action == "SW_3" or action.action == "RO":
                    next_action_stack = [next_action_stack[0], next_action_stack[4],
                                         next_action_stack[2], next_action_stack[3],
                                         next_action_stack[1]] + next_action_stack[5:]

                if action.action == "BRK":
                    if not next_action_buffer:
                        raise InvalidParseException("Error parsing sentence for AMR with ID: %s" % amr_id)
                    next_action_stack = [word_index_map.get(action.label.split("-")[0], no_word_index)] + \
                                        [word_index_map.get(action.label2.split("-")[0],
                                                            no_word_index)] + next_action_stack
                    next_action_buffer = next_action_buffer[1:]

                if action.action == "SW_BK":
                    tokens_sequence.insert(1, next_action_stack[1])
                    next_action_stack = [next_action_stack[0]] + next_action_stack[2:]

                next_action_prev_action = action
                features_matrix.append(features)

        except InvalidParseException as e:
            exception_count += 1
            logging.warn(e)
            exception_count += 1
            continue

        features_matrix = np.concatenate((np.asarray(features_matrix),
                                          np.zeros((max_len - len(features_matrix), input_size), dtype=np.int32)))

        action_sequences.append(action_sequence)
        x_full[i, :, :] = features_matrix
        i += 1

    logging.warning("Exception count " + str(exception_count))

    for action_sequence, i in zip(action_sequences, list(range(len(action_sequences)))):
        y_train_instance_matrix = []
        for action in action_sequence:
            y_train_instance_matrix.append(
                oh_encode_parser_action(action, parser_parameters.with_target_semantic_labels))

        for j in range(max_len - len(action_sequence)):
            y_train_instance_matrix.append(
                oh_encode_parser_action(None, parser_parameters.with_target_semantic_labels))
        y_full[i, :, :] = y_train_instance_matrix

    return x_full, y_full, lengths, filtered_count


def init_label_binarizers():
    global simple_target_label_binarizer, composed_target_label_binarizer, amr_rel_binarizer

    simple_target_label_binarizer = LabelBinarizer()
    simple_target_label_binarizer.fit(list(range(ActionSet.action_set_size())))

    composed_target_label_binarizer = LabelBinarizer()
    composed_target_label_binarizer.fit(list(range(ActionSet.action_set_size() + 2 * len(__AMR_RELATIONS))))

    amr_rel_binarizer = LabelBinarizer()
    amr_rel_binarizer.fit(list(range(len(__AMR_RELATIONS))))


def decode_parser_action(action_index, with_target_semantic_labels):
    if with_target_semantic_labels:
        if action_index > ActionSet.action_set_size():
            return ActionSet.index_action(action_index // len(__AMR_RELATIONS)), \
                   (action_index - ActionSet.action_set_size()) % len(__AMR_RELATIONS)
        else:
            return ActionSet.index_action(action_index), -1
    else:
        return ActionSet.index_action(action_index), -1


def oh_decode_parser_action(action_ohe, with_target_semantic_labels):
    if with_target_semantic_labels:
        action_index = composed_target_label_binarizer.inverse_transform(np.array([action_ohe]))[0]
        if action_index > ActionSet.action_set_size():
            return ActionSet.index_action(action_index // len(__AMR_RELATIONS)), \
                   (action_index - ActionSet.action_set_size()) % len(__AMR_RELATIONS)
        else:
            return ActionSet.index_action(action_index), -1
    else:
        action_index = simple_target_label_binarizer.inverse_transform(np.array([action_ohe]))[0]
        return ActionSet.index_action(action_index)


def oh_encode_parser_action(action, with_target_semantic_labels):
    if with_target_semantic_labels:
        if action is not None:
            if action.action == "RL":
                return composed_target_label_binarizer.transform([5 + __AMR_RELATIONS.index(action.label)])[0, :]
            elif action.action == "RR":
                return composed_target_label_binarizer.transform([5 + len(__AMR_RELATIONS) +
                                                                  __AMR_RELATIONS.index(action.label)])[0, :]
            else:
                return composed_target_label_binarizer.transform([action.index])[0, :]
        else:
            return composed_target_label_binarizer.transform([-1])[0, :]
    else:
        if action is not None:
            return simple_target_label_binarizer.transform([action.index])[0, :]
        else:
            return simple_target_label_binarizer.transform([-1])[0, :]


def oh_encode_amr_rel(amr_rel):
    if amr_rel is not None and amr_rel != "NONE":
        amr_rel_idx = __AMR_RELATIONS.index(amr_rel)
        return amr_rel_binarizer.transform([amr_rel_idx])[0, :]
    else:
        return amr_rel_binarizer.transform([-1])[0, :]


def get_dependency_features(stack_0_idx, stack_1_idx, stack_2_idx, buffer_0_idx, dependencies, parser_parameters):
    """
    Return the dependency features (stack_0_on_stack_1, stack_0_on_stack_2, stack_0_on_buffer and vice-versa) given
    a configuration of the stack, the buffer, known sentence dependencies and model parameters;
    Depending on model parameters values, features may encode only the presence of a dependency or the expected AMR
    relation corresponding to the dependency relation
    :param stack_0_idx: first element on stack
    :param stack_1_idx: second element on stack
    :param stack_2_idx: third element on stack
    :param buffer_0_idx: first element on buffer
    :param dependencies: dependencies between sentence tokens
    :param parser_parameters: collection of model properties
    :return: encoded dependency features
    """

    if parser_parameters.model_parameters.no_dep_features == 0:
        return []

    if parser_parameters.with_enhanced_dep_info:
        dep_0_on_1 = oh_encode_amr_rel(None)
        dep_1_on_0 = oh_encode_amr_rel(None)
        dep_0_on_2 = oh_encode_amr_rel(None)
        dep_2_on_0 = oh_encode_amr_rel(None)
        dep_0_on_b = oh_encode_amr_rel(None)
        dep_b_on_0 = oh_encode_amr_rel(None)

        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == stack_1_idx:
            dep_0_on_1 = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[stack_0_idx][1]))
        if stack_1_idx in list(dependencies.keys()) and dependencies[stack_1_idx][0] == stack_0_idx:
            dep_1_on_0 = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[stack_1_idx][1]))
        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == stack_2_idx:
            dep_0_on_2 = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[stack_0_idx][1]))
        if stack_2_idx in list(dependencies.keys()) and dependencies[stack_2_idx][0] == stack_0_idx:
            dep_2_on_0 = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[stack_2_idx][1]))
        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == buffer_0_idx:
            dep_0_on_b = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[stack_0_idx][1]))
        if buffer_0_idx in list(dependencies.keys()) and dependencies[buffer_0_idx][0] == stack_0_idx:
            dep_b_on_0 = oh_encode_amr_rel(get_amr_rel_for_dep_rel(dependencies[buffer_0_idx][1]))

        return np.concatenate((dep_0_on_1, dep_1_on_0, dep_0_on_2, dep_2_on_0, dep_0_on_b, dep_b_on_0))

    else:
        [dep_0_on_1, dep_1_on_0, dep_0_on_2, dep_2_on_0, dep_0_on_b, dep_b_on_0] = np.zeros(6)

        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == stack_1_idx:
            dep_0_on_1 = 1
        if stack_1_idx in list(dependencies.keys()) and dependencies[stack_1_idx][0] == stack_0_idx:
            dep_1_on_0 = 1
        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == stack_2_idx:
            dep_0_on_2 = 1
        if stack_2_idx in list(dependencies.keys()) and dependencies[stack_2_idx][0] == stack_0_idx:
            dep_2_on_0 = 1
        if stack_0_idx in list(dependencies.keys()) and dependencies[stack_0_idx][0] == buffer_0_idx:
            dep_0_on_b = 1
        if buffer_0_idx in list(dependencies.keys()) and dependencies[buffer_0_idx][0] == stack_0_idx:
            dep_b_on_0 = 1

        return np.asanyarray([dep_0_on_1, dep_1_on_0, dep_0_on_2, dep_2_on_0, dep_0_on_b, dep_b_on_0])


def get_amr_rel_for_dep_rel(dep_rel):
    if dep_rel in __DEP_AMR_REL_TABLE:
        return __DEP_AMR_REL_TABLE[dep_rel]
    else:
        return None
