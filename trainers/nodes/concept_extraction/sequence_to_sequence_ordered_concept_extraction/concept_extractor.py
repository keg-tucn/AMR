# For memory problems
"""
import dynet_config
dynet_config.set(mem=1024)
"""

import dynet as dy

from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor \
    import generate_concepts_training_data, statistics_verbs_other_concepts, ConceptsTrainingEntry
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import get_all_paths
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import get_all_concepts

from deep_dynet import support as ds
from deep_dynet.support import Vocab

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# TODO:
# - BLEU score research for RL baseline
# - use other embeddings (character? Glove?)?
# - use tag info?

EOS = "<EOS>"

# TODO: move this when fully parameterized
LSTM_NUM_OF_LAYERS = 1
WORDS_EMBEDDING_SIZE = 25
CONCEPTS_EMBEDDING_SIZE = 25
STATE_SIZE = 100
ATTENTION_SIZE = 25

USE_ATTENTION = True

bleu_smoothing = SmoothingFunction()


class ConceptsDynetGraph:
    def __init__(self, words_vocab, concepts_vocab, lstm_num_of_layers, words_embedding_size, concepts_embedding_size,
                 state_size, attention_size):
        self.model = dy.Model()
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab

        self.input_embeddings = self.model.add_lookup_parameters((words_vocab.size(), words_embedding_size))
        self.output_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), concepts_embedding_size))

        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.enc_fwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, words_embedding_size, state_size, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, words_embedding_size, state_size, self.model)

        self.dec_lstm = dy.LSTMBuilder(lstm_num_of_layers, state_size * 2 + concepts_embedding_size, state_size,
                                       self.model)

        self.attention_w1 = self.model.add_parameters((attention_size, state_size * 2))
        self.attention_w2 = self.model.add_parameters((attention_size, state_size * lstm_num_of_layers * 2))
        self.attention_v = self.model.add_parameters((1, attention_size))

        self.decoder_w = self.model.add_parameters((concepts_vocab.size(), state_size))
        self.decoder_b = self.model.add_parameters((concepts_vocab.size()))


def embed_sequence(concepts_dynet_graph, sequence):
    # Sentence already comes as list of words
    sequence = list(sequence) + [EOS]

    sequence = [concepts_dynet_graph.words_vocab.w2i[word] for word in sequence]
    return [concepts_dynet_graph.input_embeddings[index] for index in sequence]


def encode_input_sequence(concepts_dynet_graph, sequence):
    sequence_reversed = list(reversed(sequence))

    fwd_init = concepts_dynet_graph.enc_fwd_lstm.initial_state()
    bwd_init = concepts_dynet_graph.enc_bwd_lstm.initial_state()

    fwd_vectors = fwd_init.transduce(sequence)
    bwd_vectors = bwd_init.transduce(sequence_reversed)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(concepts_dynet_graph, input_mat, state, w1dt):
    w2 = dy.parameter(concepts_dynet_graph.attention_w2)
    v = dy.parameter(concepts_dynet_graph.attention_v)

    # input_mat: (encoder_state x seqlen) => input vectors concatenated as columns
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2 * dy.concatenate(list(state.s()))

    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)

    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(concepts_dynet_graph, encoded_sequence, golden_concepts, use_attention):
    golden_concepts = list(golden_concepts) + [EOS]
    embedded_golden_concepts = [concepts_dynet_graph.concepts_vocab.w2i[concept] for concept in golden_concepts]

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    w1dt = None
    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]

    if use_attention:
        input_mat = dy.concatenate_cols(encoded_sequence)
        s = concepts_dynet_graph.dec_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))
    else:
        input_mat = dy.transpose(dy.concatenate_cols(encoded_sequence))
        s = concepts_dynet_graph.dec_lstm.initial_state()

    loss = []
    predicted_concepts = []

    i = 0

    for concept in embedded_golden_concepts:

        if use_attention:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_concept_embedding])
        else:
            vector = dy.concatenate([input_mat[i], last_concept_embedding])

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)

        # QUICKFIX FOR NON ATTENTION -------- SHOULD FIGURE OUT A BETTER WAY
        if len(encoded_sequence) >= len(embedded_golden_concepts):
            i = i + 1

        # TODO: take last_concept_embedding with a probability from golden vs predicted --- error propagation problem
        last_concept_embedding = concepts_dynet_graph.output_embeddings[concept]
        loss.append(-dy.log(dy.pick(probs, concept)))

        # predict
        probs_vec = probs.vec_value()
        next_concept = probs_vec.index(max(probs_vec))

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])

    loss = dy.esum(loss)
    return loss, predicted_concepts


def predict_concepts(concepts_dynet_graph, input_sequence, use_attention):
    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    w1dt = None
    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]

    if use_attention:
        input_mat = dy.concatenate_cols(encoded_sequence)
        s = concepts_dynet_graph.dec_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))
    else:
        input_mat = dy.transpose(dy.concatenate_cols(encoded_sequence))
        s = concepts_dynet_graph.dec_lstm.initial_state()

    predicted_concepts = []
    count_EOS = 0
    j = 0

    for i in range(len(input_sequence) * 2):
        if count_EOS == 2: break

        if use_attention:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_concept_embedding])
        else:
            vector = dy.concatenate([input_mat[j], last_concept_embedding])

        # QUICKFIX FOR NON ATTENTION -------- SHOULD FIGURE OUT A BETTER WAY
        if j < len(encoded_sequence):
            j = j + 1

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_concept = probs.index(max(probs))
        last_concept_embedding = concepts_dynet_graph.output_embeddings[next_concept]

        if concepts_dynet_graph.concepts_vocab.i2w[next_concept] == EOS:
            count_EOS += 1
            j = 0
            continue

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])
    return predicted_concepts


def train(concepts_dynet_graph, input_sequence, golden_concepts, use_attention):
    dy.renew_cg()

    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)
    return decode(concepts_dynet_graph, encoded_sequence, golden_concepts, use_attention)


# Accuracy
'''
Accuracy computation right now: count the right words on right places
Should consider - length (original vs predicted)?
                - right words even if not in right position?
'''


# Previous implementation

def compute_accuracy(golden_concepts, predicted_concepts):
    accuracy = 0

    correct_predictions = 0
    total_predictions = len(predicted_concepts)

    for concept_idx in range(min(len(golden_concepts), len(predicted_concepts))):
        if golden_concepts[concept_idx] == predicted_concepts[concept_idx]:
            correct_predictions += 1

    if total_predictions != 0:
        accuracy = correct_predictions / total_predictions

    return accuracy


def compute_stats(golden_concepts, predicted_concepts):
    nb_golden_concepts = len(golden_concepts)
    nb_predicted_concepts = len(predicted_concepts)

    # How many should have been predicted - how many were predicted
    diff_nb_golden_predicted = nb_golden_concepts - nb_predicted_concepts

    # Doesn't treat duplicate concepts
    correctly_predicted_concepts = list(set(golden_concepts) & set(predicted_concepts))
    nb_correctly_predicted_concepts = len(correctly_predicted_concepts)

    nb_concepts_on_correct_positions = 0
    golden_indexes = []
    predicted_indexes = []

    # Get indexes of correct words both for golden and predicted
    for concept in correctly_predicted_concepts:
        if golden_concepts.index(concept) == predicted_concepts.index(concept):
            nb_concepts_on_correct_positions = nb_concepts_on_correct_positions + 1
        golden_indexes.append(golden_concepts.index(concept))
        predicted_indexes.append(predicted_concepts.index(concept))

    correct_order = 0
    correct_distances = 0
    total_combinations_checked = 0
    for i in range(len(golden_indexes) - 1):
        for j in range(i + 1, len(golden_indexes)):
            # Same as golden[i] - golden[j] == predicted[i] - predicted[j]
            if golden_indexes[i] - golden_indexes[j] - predicted_indexes[i] + predicted_indexes[j] == 0:
                correct_distances = correct_distances + 1
            # Check if order is correct regardless of distances
            if (golden_indexes[i] <= golden_indexes[j] and predicted_indexes[i] <= predicted_indexes[j]) or (
                    golden_indexes[i] >= golden_indexes[j] and predicted_indexes[i] >= predicted_indexes[j]):
                correct_order = correct_order + 1
            total_combinations_checked = total_combinations_checked + 1

    correct_order_percentage = 0
    correct_distances_percentage = 0

    if total_combinations_checked != 0:
        correct_order_percentage = correct_order / total_combinations_checked
        correct_distances_percentage = correct_distances / total_combinations_checked

    if nb_correctly_predicted_concepts == 1:
        correct_order_percentage = 1
        correct_distances_percentage = 1

    # Orsi style accuracy
    accuracy = nb_concepts_on_correct_positions / nb_predicted_concepts

    return (diff_nb_golden_predicted, nb_correctly_predicted_concepts, nb_concepts_on_correct_positions, accuracy,
            correct_order_percentage, correct_distances_percentage)


# F-score
'''
Does not consider if a concept appears multiple times
'''


def compute_f_score(golden_concepts, predicted_concepts):
    true_pos = len(list(set(golden_concepts) & set(predicted_concepts)))
    false_pos = len(list(set(predicted_concepts).difference(set(golden_concepts))))
    false_neg = len(list(set(golden_concepts).difference(set(predicted_concepts))))
    prec = 0
    recall = 0
    if len(predicted_concepts) != 0:
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
    f_score = 0
    if prec + recall != 0:
        f_score = 2 * (prec * recall) / (prec + recall)

    return f_score


def train_sentence(concepts_dynet_graph, sentence, identified_concepts, use_attention):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    (loss, predicted_concepts) = train(concepts_dynet_graph, input_sequence, golden_concepts, use_attention)
    loss_value = loss.value()
    loss.backward()
    concepts_dynet_graph.trainer.update()

    accuracy = compute_accuracy(golden_concepts, predicted_concepts)

    # BLEU score
    '''
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu(golden_concepts, predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    f_score = compute_f_score(golden_concepts, predicted_concepts)

    (diff_nb_golden_predicted, nb_correctly_predicted_concepts, nb_concepts_on_correct_positions, orsi_accuracy,
     correct_order_percentage, correct_distances_percentage) = compute_stats(golden_concepts, predicted_concepts)

    return (loss_value, accuracy, bleu_score, f_score, diff_nb_golden_predicted, nb_correctly_predicted_concepts,
            nb_concepts_on_correct_positions, orsi_accuracy, correct_order_percentage, correct_distances_percentage)


def test_sentence(concepts_dynet_graph, sentence, identified_concepts, use_attention):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts = predict_concepts(concepts_dynet_graph, input_sequence, use_attention)

    accuracy = compute_accuracy(golden_concepts, predicted_concepts)

    # BLEU score
    ''' 
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu(golden_concepts, predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    f_score = compute_f_score(golden_concepts, predicted_concepts)

    (diff_nb_golden_predicted, nb_correctly_predicted_concepts, nb_concepts_on_correct_positions, orsi_accuracy,
     correct_order_percentage, correct_distances_percentage) = compute_stats(golden_concepts, predicted_concepts)

    return (
    predicted_concepts, accuracy, bleu_score, f_score, diff_nb_golden_predicted, nb_correctly_predicted_concepts,
    nb_concepts_on_correct_positions, orsi_accuracy, correct_order_percentage, correct_distances_percentage)


def read_train_test_data():
    train_entries, no_train_failed = generate_concepts_training_data(get_all_paths('training'))
    no_train_entries = len(train_entries)
    print(str(no_train_entries) + ' train entries processed ' + str(no_train_failed) + ' train entries failed')
    test_entries, no_test_failed = generate_concepts_training_data(get_all_paths('dev'))
    no_test_entries = len(test_entries)
    print(str(no_test_entries) + ' test entries processed ' + str(no_test_failed) + ' test entries failed')
    return (train_entries, no_train_entries, test_entries, no_test_entries)


# log files
detail_logs_file_name = "logs/concept_extractor_detailed_logs.txt"
overview_logs_file_name = "logs/concept_extractor_overview_logs.txt"

if __name__ == "__main__":
    (train_entries, no_train_entries, test_entries, no_test_entries) = read_train_test_data()

    stats = statistics_verbs_other_concepts(get_all_paths('training') + get_all_paths('dev'))

    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    all_concepts = get_all_concepts(train_concepts)
    all_concepts.append(EOS)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)

    # TODO: Split sentences just based on space? Use tokens?

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

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")

    # init plotting data
    plotting_data = {}

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, LSTM_NUM_OF_LAYERS,
                                              WORDS_EMBEDDING_SIZE, CONCEPTS_EMBEDDING_SIZE, STATE_SIZE, ATTENTION_SIZE)
    no_epochs = 20

    for epoch in range(1, no_epochs + 1):
        print("Epoch " + str(epoch) + '\n')
        detail_logs.write("Epoch " + str(epoch) + '\n')
        overview_logs.write("Epoch " + str(epoch) + '\n')

        # train
        sum_loss = 0
        sum_train_accuracy = 0
        sum_train_bleu = 0
        sum_train_f = 0

        # STATS STUFF
        sum_diff_nb_golden_predicted = 0
        sum_nb_correctly_predicted_concepts = 0
        sum_orsi_accuracy = 0
        sum_correct_order_percentage = 0
        sum_correct_distances_percentage = 0

        train_entry: ConceptsTrainingEntry
        for train_entry in train_entries:
            (entry_loss, train_entry_accuracy, train_entry_bleu, train_entry_f,
             diff_nb_golden_predicted, nb_correctly_predicted_concepts, nb_concepts_on_correct_positions, orsi_accuracy,
             correct_order_percentage, correct_distances_percentage) = \
                train_sentence(concepts_dynet_graph, train_entry.sentence, train_entry.identified_concepts,
                               USE_ATTENTION)
            sum_loss += entry_loss
            sum_train_accuracy += train_entry_accuracy
            sum_train_bleu += train_entry_bleu
            sum_train_f += train_entry_f

            # STATS STUFF
            sum_diff_nb_golden_predicted += diff_nb_golden_predicted
            sum_nb_correctly_predicted_concepts += nb_correctly_predicted_concepts
            sum_orsi_accuracy += orsi_accuracy
            sum_correct_order_percentage += correct_order_percentage
            sum_correct_distances_percentage += correct_distances_percentage

        avg_loss = sum_loss / no_train_entries
        avg_train_accuracy = sum_train_accuracy / no_train_entries
        avg_train_bleu = sum_train_bleu / no_train_entries
        avg_train_f = sum_train_f / no_train_entries

        # STATS STUFF
        avg_diff_nb_golden_predicted = sum_diff_nb_golden_predicted / no_train_entries
        avg_nb_correctly_predicted_concepts = sum_nb_correctly_predicted_concepts / no_train_entries
        avg_orsi_accuracy = sum_orsi_accuracy / no_train_entries
        avg_correct_order_percentage = sum_correct_order_percentage / no_train_entries
        avg_correct_distances_percentage = sum_correct_distances_percentage / no_train_entries

        print("Loss: " + str(avg_loss))
        print("Train accuracy: " + str(avg_train_accuracy))
        print("Train bleu: " + str(avg_train_bleu))
        print("Train F-score: " + str(avg_train_f) + '\n')
        print("STATISTICS---------------")
        print(
            "Average difference in number between golden and predicted concepts: " + str(avg_diff_nb_golden_predicted))
        print("Average number of correctly predicted concepts: " + str(avg_nb_correctly_predicted_concepts))
        print("Average ORSI accuracy (second comp, should be approx. the same as first): " + str(avg_orsi_accuracy))
        print("Order correctness (just for correctly predicted): " + str(avg_correct_order_percentage))
        print("Distance correctness (just for correctly predicted): " + str(avg_correct_distances_percentage) + '\n')
        overview_logs.write("Loss: " + str(avg_loss) + '\n')
        overview_logs.write("Train accuracy: " + str(avg_train_accuracy) + '\n')
        overview_logs.write("Train bleu: " + str(avg_train_bleu) + '\n')
        overview_logs.write("Train F-score: " + str(avg_train_f) + '\n')
        overview_logs.write("STATISTICS\n")
        overview_logs.write("Average difference in number between golden and predicted concepts: " + str(
            avg_diff_nb_golden_predicted) + '\n')
        overview_logs.write(
            "Average number of correctly predicted concepts: " + str(avg_nb_correctly_predicted_concepts) + '\n')
        overview_logs.write("Average ORSI accuracy (second comp, should be approx. the same as first): " + str(
            avg_orsi_accuracy) + '\n')
        overview_logs.write(
            "Order correctness (just for correctly predicted): " + str(avg_correct_order_percentage) + '\n')
        overview_logs.write(
            "Distance correctness (just for correctly predicted): " + str(avg_correct_distances_percentage) + '\n')

        # test
        sum_accuracy = 0
        sum_bleu = 0
        sum_f_score = 0

        # STATS STUFF
        sum_diff_nb_golden_predicted = 0
        sum_nb_correctly_predicted_concepts = 0
        sum_orsi_accuracy = 0
        sum_correct_order_percentage = 0
        sum_correct_distances_percentage = 0

        test_entry: ConceptsTrainingEntry
        for test_entry in test_entries:
            (predicted_concepts, entry_accuracy, entry_bleu, entry_f, diff_nb_golden_predicted,
             nb_correctly_predicted_concepts, nb_concepts_on_correct_positions, orsi_accuracy, correct_order_percentage,
             correct_distances_percentage) = \
                test_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts, USE_ATTENTION)
            sum_accuracy += entry_accuracy
            sum_bleu += entry_bleu
            sum_f_score += entry_f

            # STATS STUFF
            sum_diff_nb_golden_predicted += diff_nb_golden_predicted
            sum_nb_correctly_predicted_concepts += nb_correctly_predicted_concepts
            sum_orsi_accuracy += orsi_accuracy
            sum_correct_order_percentage += correct_order_percentage
            sum_correct_distances_percentage += correct_distances_percentage

            # logging
            detail_logs.write('Entry accuracy: ' + str(entry_accuracy) + '\n')
            detail_logs.write('Predicted concepts: ' + str(predicted_concepts) + '\n')
            detail_logs.write('Entry bleu: ' + str(entry_bleu) + '\n')
            detail_logs.write('Entry F-score: ' + str(entry_f) + '\n')
            detail_logs.write('Difference between nb of golden and predicted: ' + str(diff_nb_golden_predicted) + '\n')
            detail_logs.write('Nb of correctly predicted: ' + str(nb_correctly_predicted_concepts) + '\n')
            detail_logs.write('Nb concepts on correct positions: ' + str(nb_concepts_on_correct_positions) + '\n')
            detail_logs.write('ORSI ACCURACY, should be approx same as prev one: ' + str(orsi_accuracy) + '\n')
            detail_logs.write(
                'Correct Order percenatge (just correct concepts): ' + str(correct_order_percentage) + '\n')
            detail_logs.write(
                'Correct Distances percentage (jusr correct concepts): ' + str(correct_distances_percentage) + '\n')
            detail_logs.write(test_entry.logging_info)

        avg_accuracy = sum_accuracy / no_test_entries
        avg_bleu = sum_bleu / no_test_entries
        avg_f = sum_f_score / no_test_entries

        # STATS STUFF
        avg_diff_nb_golden_predicted = sum_diff_nb_golden_predicted / no_test_entries
        avg_nb_correctly_predicted_concepts = sum_nb_correctly_predicted_concepts / no_test_entries
        avg_orsi_accuracy = sum_orsi_accuracy / no_test_entries
        avg_correct_order_percentage = sum_correct_order_percentage / no_test_entries
        avg_correct_distances_percentage = sum_correct_distances_percentage / no_test_entries

        print("Test accuracy: " + str(avg_accuracy))
        print("Test bleu: " + str(avg_bleu))
        print("Test F-score: " + str(avg_f) + '\n')
        print("STATISTICS---------------")
        print(
            "Average difference in number between golden and predicted concepts: " + str(avg_diff_nb_golden_predicted))
        print("Average number of correctly predicted concepts: " + str(avg_nb_correctly_predicted_concepts))
        print("Average ORSI accuracy (second comp, should be approx. the same as first): " + str(avg_orsi_accuracy))
        print("Order correctness (just for correctly predicted): " + str(avg_correct_order_percentage))
        print("Distance correctness (just for correctly predicted): " + str(avg_correct_distances_percentage) + '\n')
        overview_logs.write("Test accuracy: " + str(avg_accuracy) + '\n')
        overview_logs.write("Test bleu: " + str(avg_bleu) + '\n')
        overview_logs.write("Test F-score: " + str(avg_f) + '\n')
        overview_logs.write("STATISTICS\n")
        overview_logs.write("Average difference in number between golden and predicted concepts: " + str(
            avg_diff_nb_golden_predicted) + '\n')
        overview_logs.write(
            "Average number of correctly predicted concepts: " + str(avg_nb_correctly_predicted_concepts) + '\n')
        overview_logs.write("Average ORSI accuracy (second comp, should be approx. the same as first): " + str(
            avg_orsi_accuracy) + '\n')
        overview_logs.write(
            "Order correctness (just for correctly predicted): " + str(avg_correct_order_percentage) + '\n')
        overview_logs.write(
            "Distance correctness (just for correctly predicted): " + str(avg_correct_distances_percentage) + '\n')


    print("Done")
    detail_logs.close()
    overview_logs.close()
