# For memory problems
"""
import dynet_config
dynet_config.set(mem=1024)
"""

import dynet as dy

from data_extraction.word_embeddings_reader import read_glove_embeddings_from_file
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor \
    import generate_concepts_training_data, ConceptsTrainingEntry
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import get_all_paths, \
    construct_concept_glove_embeddings_list
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import get_all_concepts

from deep_dynet import support as ds
from deep_dynet.support import Vocab

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# TODO:
# - Attention analysis
# - RL
# - Split into verbs and non-verbs


EOS = "<EOS>"

# TODO: move this when fully parameterized
LSTM_NB_LAYERS = 1
WORDS_EMBEDDING_SIZE = 50
WORDS_GLOVE_EMBEDDING_SIZE = 50
CONCEPTS_EMBEDDING_SIZE = 50
STATE_SIZE = 40
ATTENTION_SIZE = 40

USE_ATTENTION = True
USE_GLOVE = False

NB_EPOCHS = 50

bleu_smoothing = SmoothingFunction()


class ConceptsDynetGraph:
    def __init__(self, words_vocab, concepts_vocab, words_glove_embeddings_list, lstm_nb_layers,
                 words_embedding_size, concepts_embedding_size, words_glove_embedding_size, state_size, attention_size,
                 test_concepts_vocab):

        self.model = dy.Model()
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab

        # Temporary, just until loss is computed on dev too
        self.test_concepts_vocab: Vocab = test_concepts_vocab

        self.input_embeddings = self.model.add_lookup_parameters((words_vocab.size(), words_embedding_size))
        self.output_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), concepts_embedding_size))
        self.glove_embeddings = self.model.add_lookup_parameters((words_vocab.size(), words_glove_embedding_size))
        self.glove_embeddings.init_from_array(dy.np.array(words_glove_embeddings_list))

        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.enc_fwd_lstm = dy.LSTMBuilder(lstm_nb_layers, words_embedding_size, state_size, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(lstm_nb_layers, words_embedding_size, state_size, self.model)

        self.dec_lstm = dy.LSTMBuilder(lstm_nb_layers, state_size * 2 + concepts_embedding_size, state_size, self.model)

        self.attention_w1 = self.model.add_parameters((attention_size, state_size * 2))
        self.attention_w2 = self.model.add_parameters(
            (attention_size, state_size * lstm_nb_layers * 2))  # lstm_nb_layers of decoder lstm

        self.attention_v = self.model.add_parameters((1, attention_size))

        self.decoder_w = self.model.add_parameters((concepts_vocab.size(), state_size))
        self.decoder_b = self.model.add_parameters((concepts_vocab.size()))


def embed_sequence(concepts_dynet_graph, sequence, use_glove):
    # Sentence already comes as list of words
    sequence = list(sequence) + [EOS]

    sequence = [concepts_dynet_graph.words_vocab.w2i[word] for word in sequence]

    if use_glove:
        return [dy.lookup(concepts_dynet_graph.glove_embeddings, index, False) for index in sequence]

    return [dy.lookup(concepts_dynet_graph.input_embeddings, index) for index in sequence]


def encode_input_sequence(concepts_dynet_graph, sequence):
    sequence_reversed = list(reversed(sequence))

    fwd_init = concepts_dynet_graph.enc_fwd_lstm.initial_state()
    bwd_init = concepts_dynet_graph.enc_bwd_lstm.initial_state()

    fwd_vectors = fwd_init.transduce(sequence)
    bwd_vectors = bwd_init.transduce(sequence_reversed)
    bwd_vectors_reversed = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors_reversed)]

    return vectors


def attend(concepts_dynet_graph, input_matrix, dec_state, w1_input, sequence_len):
    w2 = dy.parameter(concepts_dynet_graph.attention_w2)
    v = dy.parameter(concepts_dynet_graph.attention_v)

    # Dimensions
    # input_matrix: (enc_state * sequence_len) - input vectors concatenated as columns
    # w1_input: (attention_dim * sequence_len) - w1 * input_matrix
    # w2_state: (attention_dim * attention_dim) - w2 * dec_state
    # att_weights: (sequence_len)
    # context: (enc_state)

    w2_state = w2 * dy.concatenate(list(dec_state.s()))
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1_input, w2_state)))
    attention_weights = dy.softmax(unnormalized)
    context = input_matrix * attention_weights

    attention_max = max(dy.transpose(attention_weights).value()[0])
    attention_min = min(dy.transpose(attention_weights).value()[0])
    attention_diff = attention_max - attention_min

    # if sequence_len % 10 == 0:
    #    print(str(attention_max) + " " + str(attention_min) + " " + str(attention_diff) + "\n")

    return context, attention_diff


def decode(concepts_dynet_graph, encoded_sequence, golden_concepts, use_attention):
    golden_concepts = list(golden_concepts) + [EOS]

    # Temporary, just until loss is computed on dev too
    embedded_golden_concepts = []
    for concept in golden_concepts:
        if concept in concepts_dynet_graph.concepts_vocab.w2i:
            embedded_golden_concepts.append(concepts_dynet_graph.concepts_vocab.w2i[concept])
        else:
            embedded_golden_concepts.append(concepts_dynet_graph.test_concepts_vocab.w2i[concept])
    # embedded_golden_concepts = [concepts_dynet_graph.concepts_vocab.w2i[concept] for concept in golden_concepts]

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    w1_input = None
    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]

    if use_attention:
        input_matrix = dy.concatenate_cols(encoded_sequence)
        dec_state = concepts_dynet_graph.dec_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))
    else:
        input_matrix = dy.transpose(dy.concatenate_cols(encoded_sequence))
        dec_state = concepts_dynet_graph.dec_lstm.initial_state()

    loss = []
    predicted_concepts = []


    sum_attention_diff = 0
    i = 0

    for concept in embedded_golden_concepts:

        if use_attention:
            w1_input = w1_input or w1 * input_matrix
            context_vector, attention_diff = attend(concepts_dynet_graph, input_matrix, dec_state, w1_input,
                                                    len(encoded_sequence))
            vector = dy.concatenate([context_vector, last_concept_embedding])
            sum_attention_diff += attention_diff
        else:
            vector = dy.concatenate([input_matrix[i], last_concept_embedding])

        dec_state = dec_state.add_input(vector)
        out_vector = w * dec_state.output() + b
        probs = dy.softmax(out_vector)

        # QUICKFIX FOR NON ATTENTION -------- SHOULD FIGURE OUT A BETTER WAY
        if len(encoded_sequence) >= len(embedded_golden_concepts):
            i += 1


        # TODO: take last_concept_embedding with a probability from golden vs predicted --- error propagation problem
        last_concept_embedding = concepts_dynet_graph.output_embeddings[concept]
        loss.append(-dy.log(dy.pick(probs, concept)))

        # predict
        probs_vector = probs.vec_value()
        next_concept = probs_vector.index(max(probs_vector))

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])

    loss = dy.esum(loss)
    entry_attention_diff = sum_attention_diff / len(embedded_golden_concepts)
    return loss, predicted_concepts, entry_attention_diff


def predict_concepts(concepts_dynet_graph, encoded_sequence, use_attention):
    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    w1_input = None
    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]

    if use_attention:
        input_matrix = dy.concatenate_cols(encoded_sequence)
        dec_state = concepts_dynet_graph.dec_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))
    else:
        input_matrix = dy.transpose(dy.concatenate_cols(encoded_sequence))
        dec_state = concepts_dynet_graph.dec_lstm.initial_state()

    predicted_concepts = []
    count_EOS = 0
    sum_attention_diff = 0
    j = 0

    for i in range((len(encoded_sequence) - 1) * 2):
        if count_EOS == 2: break

        if use_attention:
            w1_input = w1_input or w1 * input_matrix
            context_vector, attention_diff = attend(concepts_dynet_graph, input_matrix, dec_state, w1_input,
                                                    len(encoded_sequence))
            vector = dy.concatenate([context_vector, last_concept_embedding])
            sum_attention_diff += attention_diff
        else:
            vector = dy.concatenate([input_matrix[j], last_concept_embedding])

        # QUICKFIX FOR NON ATTENTION -------- SHOULD FIGURE OUT A BETTER WAY
        if j < len(encoded_sequence):
            j += 1

        dec_state = dec_state.add_input(vector)
        out_vector = w * dec_state.output() + b
        probs_vector = dy.softmax(out_vector).vec_value()
        next_concept = probs_vector.index(max(probs_vector))
        last_concept_embedding = concepts_dynet_graph.output_embeddings[next_concept]

        if concepts_dynet_graph.concepts_vocab.i2w[next_concept] == EOS:
            count_EOS += 1
            j = 0
            continue

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])

    entry_attention_diff = sum_attention_diff / len(predicted_concepts)
    return predicted_concepts, entry_attention_diff


def train(concepts_dynet_graph, input_sequence, golden_concepts, use_glove, use_attention):
    dy.renew_cg()

    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence, use_glove)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)
    # if len(encoded_sequence) % 10 == 0:
    #    print(input_sequence)
    return decode(concepts_dynet_graph, encoded_sequence, golden_concepts, use_attention)


def test(concepts_dynet_graph, input_sequence, use_glove, use_attention):
    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence, use_glove)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)
    # if len(encoded_sequence) % 10 == 0:
    #    print(input_sequence)
    return predict_concepts(concepts_dynet_graph, encoded_sequence, use_attention)


# F-score
'''
Does not consider duplicate concepts in the same sentence.
'''


def compute_f_score(golden_concepts, predicted_concepts):
    true_positive = len(list(set(golden_concepts) & set(predicted_concepts)))
    false_positive = len(list(set(predicted_concepts).difference(set(golden_concepts))))
    false_negative = len(list(set(golden_concepts).difference(set(predicted_concepts))))
    precision = 0
    recall = 0
    if len(predicted_concepts) != 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    f_score = 0
    if precision + recall != 0:
        f_score = 2 * (precision * recall) / (precision + recall)

    return true_positive, precision, recall, f_score


# METRICS
'''
They do not consider dupliacte concepts in the same sentence.

ACCURACY
Count the right concepts on the correct positions / all predicted concepts.

ORDER METRICS
Check the order in each pair of correctly predicted concepts, count the correct orders in the predicted set.
Check the distance between the correctly predicted concepts in the predicted set, count instances where it is the same 
as in the golden set.
Divide by all checked combinations of correctly predicted concepts.
'''


def compute_metrics(golden_concepts, predicted_concepts):
    nb_predicted_concepts = len(predicted_concepts)

    correctly_predicted_concepts = list(set(golden_concepts) & set(predicted_concepts))
    nb_correctly_predicted_concepts = len(correctly_predicted_concepts)

    nb_concepts_on_correct_positions = 0
    golden_indexes = []
    predicted_indexes = []

    # Get indexes of correct words both for golden and predicted
    for concept in correctly_predicted_concepts:
        if golden_concepts.index(concept) == predicted_concepts.index(concept):
            nb_concepts_on_correct_positions += 1
        golden_indexes.append(golden_concepts.index(concept))
        predicted_indexes.append(predicted_concepts.index(concept))

    # Accuracy
    accuracy = nb_concepts_on_correct_positions / nb_predicted_concepts

    correct_order = 0
    correct_distances = 0
    total_combinations_checked = 0

    for i in range(len(golden_indexes) - 1):
        for j in range(i + 1, len(golden_indexes)):
            # Same as golden[i] - golden[j] == predicted[i] - predicted[j]
            if golden_indexes[i] - golden_indexes[j] - predicted_indexes[i] + predicted_indexes[j] == 0:
                correct_distances += 1
            # Check if order is correct regardless of distances
            if (golden_indexes[i] <= golden_indexes[j] and predicted_indexes[i] <= predicted_indexes[j]) or \
                    (golden_indexes[i] >= golden_indexes[j] and predicted_indexes[i] >= predicted_indexes[j]):
                correct_order += 1
            total_combinations_checked += 1

    # Order metrics
    correct_order_percentage = 0
    correct_distances_percentage = 0

    if total_combinations_checked != 0:
        correct_order_percentage = correct_order / total_combinations_checked
        correct_distances_percentage = correct_distances / total_combinations_checked

    if nb_correctly_predicted_concepts == 1:
        correct_order_percentage = 1
        correct_distances_percentage = 1

    return accuracy, correct_order_percentage, correct_distances_percentage


def train_sentence(concepts_dynet_graph, sentence, identified_concepts, use_glove, use_attention, train_flag):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    loss, predicted_concepts, entry_attention_diff = train(concepts_dynet_graph, input_sequence, golden_concepts,
                                                           use_glove, use_attention)
    loss_value = loss.value()
    if train_flag == True:
        loss.backward()
    concepts_dynet_graph.trainer.update()

    # BLEU score
    ''' 
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu([golden_concepts], predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    nb_correctly_predicted_concepts, precision, recall, f_score = compute_f_score(golden_concepts, predicted_concepts)

    accuracy, correct_order_percentage, correct_distances_percentage = \
        compute_metrics(golden_concepts, predicted_concepts)

    return predicted_concepts, loss_value, bleu_score, nb_correctly_predicted_concepts, precision, recall, f_score, \
           accuracy, correct_order_percentage, correct_distances_percentage, entry_attention_diff


def test_sentence(concepts_dynet_graph, sentence, identified_concepts, use_glove, use_attention):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts, entry_attention_diff = test(concepts_dynet_graph, input_sequence, use_glove, use_attention)

    # BLEU score
    ''' 
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu([golden_concepts], predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    nb_correctly_predicted_concepts, precision, recall, f_score = compute_f_score(golden_concepts, predicted_concepts)

    accuracy, correct_order_percentage, correct_distances_percentage = \
        compute_metrics(golden_concepts, predicted_concepts)

    return predicted_concepts, bleu_score, nb_correctly_predicted_concepts, precision, recall, f_score, \
           accuracy, correct_order_percentage, correct_distances_percentage, entry_attention_diff


def read_train_test_data():
    train_entries, nb_train_failed = generate_concepts_training_data(get_all_paths('training'))
    nb_train_entries = len(train_entries)
    print(str(nb_train_entries) + ' train entries processed ' + str(nb_train_failed) + ' train entries failed')
    test_entries, nb_test_failed = generate_concepts_training_data(get_all_paths('dev'))
    nb_test_entries = len(test_entries)
    print(str(nb_test_entries) + ' test entries processed ' + str(nb_test_failed) + ' test entries failed')
    return train_entries, nb_train_entries, test_entries, nb_test_entries


# log files
detail_logs_file_name = "logs/concept_extractor_detailed_logs.txt"
detail_test_logs_file_name = "logs/concept_extractor_detailed_test_logs.txt"
overview_logs_file_name = "logs/concept_extractor_overview_logs.txt"

if __name__ == "__main__":
    train_entries, nb_train_entries, test_entries, nb_test_entries = read_train_test_data()

    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    all_concepts = get_all_concepts(train_concepts)
    all_concepts.append(EOS)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)

    # Temporary, just until loss is computed on dev too
    test_concepts = [test_entry.identified_concepts for test_entry in test_entries]
    all_test_concepts = get_all_concepts(test_concepts)
    all_test_concepts.append(EOS)
    all_test_concepts_vocab = ds.Vocab.from_list(all_test_concepts)

    train_words = []
    for train_entry in train_entries:
        for word in train_entry.sentence.split():
            train_words.append(word)
    dev_words = []
    for test_entry in test_entries:
        for word in test_entry.sentence.split():
            dev_words.append(word)
    all_words = list(set(train_words + dev_words))
    all_words.append(EOS)
    all_words_vocab = ds.Vocab.from_list(all_words)

    word_glove_embeddings = read_glove_embeddings_from_file(WORDS_GLOVE_EMBEDDING_SIZE)
    words_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                          WORDS_GLOVE_EMBEDDING_SIZE,
                                                                          all_words_vocab)

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    detail_test_logs = open(detail_test_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, words_glove_embeddings_list,
                                              LSTM_NB_LAYERS, WORDS_EMBEDDING_SIZE, CONCEPTS_EMBEDDING_SIZE,
                                              WORDS_GLOVE_EMBEDDING_SIZE, STATE_SIZE, ATTENTION_SIZE,
                                              all_test_concepts_vocab)

    for epoch in range(1, NB_EPOCHS + 1):
        print("Epoch " + str(epoch) + "\n")
        detail_logs.write("Epoch " + str(epoch) + "\n\n")
        overview_logs.write("Epoch " + str(epoch) + "\n\n")

        # train
        sum_train_loss = 0
        sum_train_bleu = 0
        sum_train_nb_correctly_predicted_concepts = 0
        sum_train_precision = 0
        sum_train_recall = 0
        sum_train_f_score = 0
        sum_train_accuracy = 0
        sum_train_correct_order_percentage = 0
        sum_train_correct_distances_percentage = 0
        sum_train_attention_difference = 0

        train_entry: ConceptsTrainingEntry
        train_flag = True
        for train_entry in train_entries:
            (predicted_concepts, train_entry_loss, train_entry_bleu, train_entry_nb_correctly_predicted_concepts,
             train_entry_precision, train_entry_recall, train_entry_f_score, train_entry_accuracy,
             train_entry_correct_order_percentage, train_entry_correct_distances_percentage,
             train_entry_attention_diff) = \
                train_sentence(concepts_dynet_graph, train_entry.sentence, train_entry.identified_concepts,
                               USE_GLOVE, USE_ATTENTION, train_flag)

            sum_train_loss += train_entry_loss
            sum_train_bleu += train_entry_bleu
            sum_train_nb_correctly_predicted_concepts += train_entry_nb_correctly_predicted_concepts
            sum_train_precision += train_entry_precision
            sum_train_recall += train_entry_recall
            sum_train_f_score += train_entry_f_score
            sum_train_accuracy += train_entry_accuracy
            sum_train_correct_order_percentage += train_entry_correct_order_percentage
            sum_train_correct_distances_percentage += train_entry_correct_distances_percentage
            sum_train_attention_difference += train_entry_attention_diff

        avg_train_loss = sum_train_loss / nb_train_entries
        avg_train_bleu = sum_train_bleu / nb_train_entries
        avg_train_nb_correctly_predicted_concepts = sum_train_nb_correctly_predicted_concepts / nb_train_entries
        avg_train_precision = sum_train_precision / nb_train_entries
        avg_train_recall = sum_train_recall / nb_train_entries
        avg_train_f_score = sum_train_f_score / nb_train_entries
        avg_train_accuracy = sum_train_accuracy / nb_train_entries
        avg_train_correct_order_percentage = sum_train_correct_order_percentage / nb_train_entries
        avg_train_correct_distances_percentage = sum_train_correct_distances_percentage / nb_train_entries
        avg_train_attention_difference = sum_train_attention_difference / nb_train_entries

        print("LOSS: " + str(avg_train_loss))
        print("Train BLEU: " + str(avg_train_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_train_nb_correctly_predicted_concepts))
        print("Train PRECISION: " + str(avg_train_precision))
        print("Train RECALL: " + str(avg_train_recall))
        print("Train F-SCORE: " + str(avg_train_f_score))
        print("Train ACCURACY (correctly predicted concepts on correct positions): " + str(avg_train_accuracy))
        print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
              str(avg_train_correct_order_percentage))
        print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
              str(avg_train_correct_distances_percentage))
        print("Average (max - min) difference in attention: " + str(avg_train_attention_difference) + "\n")

        overview_logs.write("Train LOSS: " + str(avg_train_loss) + "\n")
        overview_logs.write("Train BLEU: " + str(avg_train_bleu) + "\n")
        overview_logs.write("Average number of correctly predicted concepts per sentence: " +
                            str(avg_train_nb_correctly_predicted_concepts) + "\n")
        overview_logs.write("Train PRECISION: " + str(avg_train_precision) + "\n")
        overview_logs.write("Train RECALL: " + str(avg_train_recall) + "\n")
        overview_logs.write("Train F-SCORE: " + str(avg_train_f_score) + "\n")
        overview_logs.write("Train ACCURACY (correctly predicted concepts on correct positions): " +
                            str(avg_train_accuracy) + "\n")
        overview_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                            str(avg_train_correct_order_percentage) + "\n")
        overview_logs.write("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
                            str(avg_train_correct_distances_percentage) + "\n")
        overview_logs.write("Average (max - min) difference in attention: " +
                            str(avg_train_attention_difference) + "\n")
        overview_logs.write("\n")

        # golden
        sum_loss = 0
        sum_bleu = 0
        sum_nb_correctly_predicted_concepts = 0
        sum_precision = 0
        sum_recall = 0
        sum_f_score = 0
        sum_accuracy = 0
        sum_correct_order_percentage = 0
        sum_correct_distances_percentage = 0
        sum_attention_difference = 0

        # test
        sum_test_bleu = 0
        sum_test_nb_correctly_predicted_concepts = 0
        sum_test_precision = 0
        sum_test_recall = 0
        sum_test_f_score = 0
        sum_test_accuracy = 0
        sum_test_correct_order_percentage = 0
        sum_test_correct_distances_percentage = 0
        sum_test_attention_difference = 0

        test_entry: ConceptsTrainingEntry
        train_flag = False
        for test_entry in test_entries:
            # With last_embedding from golden
            (predicted_concepts, entry_loss, entry_bleu, entry_nb_correctly_predicted_concepts, entry_precision,
             entry_recall, entry_f_score, entry_accuracy, entry_correct_order_percentage,
             entry_correct_distances_percentage, entry_attention_diff) = \
                train_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts, USE_GLOVE,
                               USE_ATTENTION, train_flag)

            sum_loss += entry_loss
            sum_bleu += entry_bleu
            sum_nb_correctly_predicted_concepts += entry_nb_correctly_predicted_concepts
            sum_precision += entry_precision
            sum_recall += entry_recall
            sum_f_score += entry_f_score
            sum_accuracy += entry_accuracy
            sum_correct_order_percentage += entry_correct_order_percentage
            sum_correct_distances_percentage += entry_correct_distances_percentage
            sum_attention_difference += entry_attention_diff

            detail_logs.write(test_entry.logging_info)
            detail_logs.write("PREDICTED concepts: " + str(predicted_concepts) + "\n")
            detail_logs.write("BLEU: " + str(entry_bleu) + "\n")
            detail_logs.write("Correctly predicted concepts: " + str(entry_nb_correctly_predicted_concepts) + "\n")
            detail_logs.write("PRECISION: " + str(entry_precision) + "\n")
            detail_logs.write("RECALL: " + str(entry_recall) + "\n")
            detail_logs.write("F-SCORE: " + str(entry_f_score) + "\n")
            detail_logs.write("ACCURACY (correctly predicted concepts on correct positions): " +
                              str(entry_accuracy) + "\n")
            detail_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                              str(entry_correct_order_percentage) + "\n")
            detail_logs.write("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
                              str(entry_correct_distances_percentage) + "\n")
            detail_logs.write("Average (max - min) difference in attention: " + str(entry_attention_diff) + "\n")
            detail_logs.write("\n")

            # With last_embedding from predictions
            (test_predicted_concepts, test_entry_bleu, test_entry_nb_correctly_predicted_concepts, test_entry_precision,
             test_entry_recall, test_entry_f_score, test_entry_accuracy, test_entry_correct_order_percentage,
             test_entry_correct_distances_percentage, test_entry_attention_diff) = \
                test_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts, USE_GLOVE,
                              USE_ATTENTION)

            sum_test_bleu += test_entry_bleu
            sum_test_nb_correctly_predicted_concepts += test_entry_nb_correctly_predicted_concepts
            sum_test_precision += test_entry_precision
            sum_test_recall += test_entry_recall
            sum_test_f_score += test_entry_f_score
            sum_test_accuracy += test_entry_accuracy
            sum_test_correct_order_percentage += test_entry_correct_order_percentage
            sum_test_correct_distances_percentage += test_entry_correct_distances_percentage
            sum_test_attention_difference += test_entry_attention_diff

            detail_test_logs.write(test_entry.logging_info)
            detail_test_logs.write("PREDICTED concepts: " + str(test_predicted_concepts) + "\n")
            detail_test_logs.write("BLEU: " + str(test_entry_bleu) + "\n")
            detail_test_logs.write("Correctly predicted concepts: " +
                                   str(test_entry_nb_correctly_predicted_concepts) + "\n")
            detail_test_logs.write("PRECISION: " + str(test_entry_precision) + "\n")
            detail_test_logs.write("RECALL: " + str(test_entry_recall) + "\n")
            detail_test_logs.write("F-SCORE: " + str(test_entry_f_score) + "\n")
            detail_test_logs.write("ACCURACY (correctly predicted concepts on correct positions): " +
                                   str(test_entry_accuracy) + "\n")
            detail_test_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                                   str(test_entry_correct_order_percentage) + "\n")
            detail_test_logs.write(
                "Percentage of concepts at correct distances (only for correctly predicted concepts): " +
                str(test_entry_correct_distances_percentage) + "\n")
            detail_test_logs.write("Average (max - min) difference in attention: " +
                                   str(test_entry_attention_diff) + "\n")
            detail_test_logs.write("\n")

        # With last_embedding from golden
        avg_loss = sum_loss / nb_test_entries
        avg_bleu = sum_bleu / nb_test_entries
        avg_nb_correctly_predicted_concepts = sum_nb_correctly_predicted_concepts / nb_test_entries
        avg_precision = sum_precision / nb_test_entries
        avg_recall = sum_recall / nb_test_entries
        avg_f_score = sum_f_score / nb_test_entries
        avg_accuracy = sum_accuracy / nb_test_entries
        avg_correct_order_percentage = sum_correct_order_percentage / nb_test_entries
        avg_correct_distances_percentage = sum_correct_distances_percentage / nb_test_entries
        avg_attention_difference = sum_attention_difference / nb_test_entries

        print("Golden test LOSS: " + str(avg_loss))
        print("Golden test BLEU: " + str(avg_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_nb_correctly_predicted_concepts))
        print("Golden test PRECISION: " + str(avg_precision))
        print("Golden test RECALL: " + str(avg_recall))
        print("Golden test F-SCORE: " + str(avg_f_score))
        print("Golden test ACCURACY (correctly predicted concepts on correct positions): " + str(avg_accuracy))
        print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
              str(avg_correct_order_percentage))
        print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
              str(avg_correct_distances_percentage))
        print("Average (max - min) difference in attention: " + str(avg_attention_difference) + "\n")

        overview_logs.write("Golden test LOSS: " + str(avg_loss) + "\n")
        overview_logs.write("Golden test BLEU: " + str(avg_bleu) + "\n")
        overview_logs.write("Average number of correctly predicted concepts per sentence: " +
                            str(avg_nb_correctly_predicted_concepts) + "\n")
        overview_logs.write("Golden test PRECISION: " + str(avg_precision) + "\n")
        overview_logs.write("Golden test RECALL: " + str(avg_recall) + "\n")
        overview_logs.write("Golden test F-SCORE: " + str(avg_f_score) + "\n")
        overview_logs.write("Golden test ACCURACY (correctly predicted concepts on correct positions): " +
                            str(avg_accuracy) + "\n")
        overview_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                            str(avg_correct_order_percentage) + "\n")
        overview_logs.write("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
                            str(avg_correct_distances_percentage) + "\n")
        overview_logs.write("Average (max - min) difference in attention: " + str(avg_attention_difference) + "\n")
        overview_logs.write("\n")

        # With last_embedding from predictions
        avg_test_bleu = sum_test_bleu / nb_test_entries
        avg_test_nb_correctly_predicted_concepts = sum_test_nb_correctly_predicted_concepts / nb_test_entries
        avg_test_precision = sum_test_precision / nb_test_entries
        avg_test_recall = sum_test_recall / nb_test_entries
        avg_test_f_score = sum_test_f_score / nb_test_entries
        avg_test_accuracy = sum_test_accuracy / nb_test_entries
        avg_test_correct_order_percentage = sum_test_correct_order_percentage / nb_test_entries
        avg_test_correct_distances_percentage = sum_test_correct_distances_percentage / nb_test_entries
        avg_test_attention_difference = sum_test_attention_difference / nb_test_entries

        print("Test BLEU: " + str(avg_test_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_test_nb_correctly_predicted_concepts))
        print("Test PRECISION: " + str(avg_test_precision))
        print("Test RECALL: " + str(avg_test_recall))
        print("Test F-SCORE: " + str(avg_test_f_score))
        print("Test ACCURACY (correctly predicted concepts on correct positions): " + str(avg_test_accuracy))
        print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
              str(avg_test_correct_order_percentage))
        print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
              str(avg_test_correct_distances_percentage))
        print("Average (max - min) difference in attention: " + str(avg_test_attention_difference) + "\n")

        overview_logs.write("Test BLEU: " + str(avg_test_bleu) + "\n")
        overview_logs.write("Average number of correctly predicted concepts per sentence: " +
                            str(avg_test_nb_correctly_predicted_concepts) + "\n")
        overview_logs.write("Test PRECISION: " + str(avg_test_precision) + "\n")
        overview_logs.write("Test RECALL: " + str(avg_test_recall) + "\n")
        overview_logs.write("Test F-SCORE: " + str(avg_test_f_score) + "\n")
        overview_logs.write("Test ACCURACY (correctly predicted concepts on correct positions): " +
                            str(avg_test_accuracy) + "\n")
        overview_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                            str(avg_test_correct_order_percentage) + "\n")
        overview_logs.write("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
                            str(avg_test_correct_distances_percentage) + "\n")
        overview_logs.write("Average (max - min) difference in attention: " + str(avg_test_attention_difference) + "\n")
        overview_logs.write("\n")

    print("Done")
    detail_logs.close()
    overview_logs.close()
