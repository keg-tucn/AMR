# For memory problems
import os
import pickle

import dynet as dy

from data_extraction.word_embeddings_reader import read_glove_embeddings_from_file
from deep_dynet.support import Vocab
from definitions import PROJECT_ROOT_DIR
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    construct_concept_glove_embeddings_list
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.trainer_util import is_verb, \
    compute_f_score, compute_metrics, ConceptsTrainerHyperparameters, get_golden_concept_indexes, initialize_decoders, \
    get_next_concept, get_last_concept_embedding, compute_bleu_score, create_vocabs, \
    get_word_index, get_model_name
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor import \
    ConceptsTrainingEntry, read_train_dev_data, read_test_data

# dynet_config.set(mem=1024)

# TODO:
# - RL


START_OF_SEQUENCE = "<SOS>"
END_OF_SEQUENCE = "<EOS>"

# TODO: delete this when fully parameterized
ENCODER_NB_LAYERS = 1
DECODER_NB_LAYERS = 1
VERB_NONVERB_CLASSIFIER_NB_LAYERS = 1
WORDS_EMBEDDING_SIZE = 50
WORDS_GLOVE_EMBEDDING_SIZE = 50
CONCEPTS_EMBEDDING_SIZE = 50
ENCODER_STATE_SIZE = 40
DECODER_STATE_SIZE = 40
VERB_NONVERB_CLASSIFIER_STATE_SIZE = 40
ATTENTION_SIZE = 40
DROPOUT_RATE = 0.6

USE_ATTENTION = True
USE_GLOVE = False
USE_VERB_NONVERB_DECODERS = False
USE_VERB_NONVERB_EMBEDDINGS_CLASSIFIER = False
ALIGNMENT = "isi"

NB_EPOCHS = 1

EXPERIMENTAL_RUN = False
TRAIN = False


class ConceptsDynetGraph:
    def __init__(self, words_vocab, concepts_vocab, words_glove_embeddings_list, concepts_verbs_vocab,
                 concepts_nonverbs_vocab, hyperparams, dev_concepts_vocab, model = None):
        if model is None:
            global_model = dy.Model()
        else:
            global_model = model
        self.model = global_model.add_subcollection("concepts")

        # BASE MODEL PARAMETERS
        # VOCABS
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab
        self.dev_concepts_vocab: Vocab = dev_concepts_vocab

        # EMBEDDINGS
        self.word_embeddings = self.model.add_lookup_parameters((words_vocab.size(), hyperparams.words_embedding_size))
        self.concept_embeddings = self.model.add_lookup_parameters(
            (concepts_vocab.size(), hyperparams.concepts_embedding_size))
        self.word_glove_embeddings = self.model.add_lookup_parameters(
            (words_vocab.size(), hyperparams.words_glove_embedding_size))
        self.word_glove_embeddings.init_from_array(dy.np.array(words_glove_embeddings_list))

        # ENCODER
        self.encoder_fwd = dy.GRUBuilder(hyperparams.encoder_nb_layers, hyperparams.words_embedding_size,
                                         hyperparams.encoder_state_size, self.model)
        self.encoder_bwd = dy.GRUBuilder(hyperparams.encoder_nb_layers, hyperparams.words_embedding_size,
                                         hyperparams.encoder_state_size, self.model)

        # DECODER
        self.decoder = dy.GRUBuilder(hyperparams.decoder_nb_layers,
                                     hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                     hyperparams.decoder_state_size, self.model)

        # EMBEDDINGS CLASSIFIER
        self.embeddings_classifier_w = self.model.add_parameters(
            (concepts_vocab.size(), hyperparams.decoder_state_size))
        self.embeddings_classifier_b = self.model.add_parameters((concepts_vocab.size()))

        self.dropout_rate = hyperparams.dropout_rate

        # ATTENTION
        self.attention_w1 = self.model.add_parameters((hyperparams.attention_size, hyperparams.encoder_state_size * 2))
        # for LSTMS w2's second size is (* 2)
        self.attention_w2 = self.model.add_parameters(
            (hyperparams.attention_size, hyperparams.decoder_state_size * hyperparams.decoder_nb_layers))
        self.attention_v = self.model.add_parameters((1, hyperparams.attention_size))

        # MODEL PARAMETERS WITH VERB-NONVERB CLASSIFICATION
        # VOCABS
        self.concepts_verbs_vocab: Vocab = concepts_verbs_vocab
        self.concepts_nonverbs_vocab: Vocab = concepts_nonverbs_vocab

        # EMBEDDINGS
        self.concept_verb_embeddings = self.model.add_lookup_parameters(
            (concepts_verbs_vocab.size(), hyperparams.concepts_embedding_size))
        self.concept_nonverb_embeddings = self.model.add_lookup_parameters(
            (concepts_nonverbs_vocab.size(), hyperparams.concepts_embedding_size))

        # DECODER
        self.verb_decoder = dy.GRUBuilder(hyperparams.decoder_nb_layers,
                                          hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                          hyperparams.decoder_state_size, self.model)
        self.nonverb_decoder = dy.GRUBuilder(hyperparams.decoder_nb_layers,
                                             hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                             hyperparams.decoder_state_size, self.model)

        # EMBEDDINGS CLASSIFIER
        self.verb_embeddings_classifier_w = self.model.add_parameters(
            (concepts_verbs_vocab.size(), hyperparams.decoder_state_size))
        self.verb_embeddings_classifier_b = self.model.add_parameters((concepts_verbs_vocab.size()))
        self.nonverb_embeddings_classifier_w = self.model.add_parameters(
            (concepts_nonverbs_vocab.size(), hyperparams.decoder_state_size))
        self.nonverb_embeddings_classifier_b = self.model.add_parameters((concepts_nonverbs_vocab.size()))

        # VERB - NONVERB CLASSIFIER
        self.classifier = dy.GRUBuilder(hyperparams.verb_nonverb_classifier_nb_layers,
                                        hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                        hyperparams.verb_nonverb_classifier_state_size, self.model)
        self.classifier_w = self.model.add_parameters(
            (2, hyperparams.verb_nonverb_classifier_state_size))
        self.classifier_b = self.model.add_parameters((2))

        # TRAINER
        self.trainer = dy.SimpleSGDTrainer(self.model)
        # self.trainer = dy.AdamTrainer(self.model)
        # self.trainer = dy.AdagradTrainer(self.model)


def classify_verb_nonverb(concepts_dynet_graph, input_vector, concept, classifier_init):
    w = dy.parameter(concepts_dynet_graph.classifier_w)
    b = dy.parameter(concepts_dynet_graph.classifier_b)

    out_label = is_verb(concept)

    # classifier_init = concepts_dynet_graph.classifier.initial_state()
    classifier_init = classifier_init.add_input(input_vector)

    out_vector = w * classifier_init.output() + b
    probs = dy.softmax(out_vector)
    loss = -dy.log(dy.pick(probs, out_label))

    return loss


def predict_verb_nonverb(concepts_dynet_graph, input_vector, classifier_init):
    w = dy.parameter(concepts_dynet_graph.classifier_w)
    b = dy.parameter(concepts_dynet_graph.classifier_b)

    # classifier_init = concepts_dynet_graph.classifier.initial_state()
    classifier_init = classifier_init.add_input(input_vector)

    out_vector = w * classifier_init.output() + b
    probs_vector = dy.softmax(out_vector).vec_value()
    predict_verb = probs_vector.index(max(probs_vector))

    return predict_verb


def embed_sequence(concepts_dynet_graph, sequence, hyperparams):
    # Add START and END markers to sequence
    sequence = [START_OF_SEQUENCE] + list(sequence) + [END_OF_SEQUENCE]

    index_sequence = []
    for word in sequence:
        index_sequence.append(get_word_index(concepts_dynet_graph, word))

    if hyperparams.use_glove:
        return [dy.lookup(concepts_dynet_graph.word_glove_embeddings, index, False) for index in index_sequence]

    return [dy.lookup(concepts_dynet_graph.word_embeddings, index) for index in index_sequence]


def encode_input_sequence(concepts_dynet_graph, sequence):
    sequence_reversed = list(reversed(sequence))

    fwd_init = concepts_dynet_graph.encoder_fwd.initial_state()
    bwd_init = concepts_dynet_graph.encoder_bwd.initial_state()

    fwd_vectors = fwd_init.transduce(sequence)
    bwd_vectors = bwd_init.transduce(sequence_reversed)
    bwd_vectors_reversed = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors_reversed)]

    return vectors


# DIMENSIONS
'''
input_matrix: ((encoder_state_size * 2) x sequence_length) - encoded sequence vectors concatenated as columns
w1_input: (attention_size x sequence_length) - w1 * input_matrix
w2_state: (attention_size x 1) - w2 * decoder_hidden_states
att_weights: (sequence_length x 1)
context: ((encoder_state_size * 2) x 1) - input_matrix * att_weights
'''


def attend(concepts_dynet_graph, input_matrix, decoder_state, w1_input):
    w2 = dy.parameter(concepts_dynet_graph.attention_w2)
    v = dy.parameter(concepts_dynet_graph.attention_v)

    w2_state = w2 * dy.concatenate(list(decoder_state.s()))

    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1_input, w2_state)))
    attention_weights = dy.softmax(unnormalized)
    context = input_matrix * attention_weights

    return context


def decode(concepts_dynet_graph, encoded_sequence, golden_concepts, hyperparams):
    golden_concepts = [START_OF_SEQUENCE] + list(golden_concepts) + [END_OF_SEQUENCE]
    golden_concept_indexes = get_golden_concept_indexes(concepts_dynet_graph, golden_concepts, hyperparams)

    w = dy.parameter(concepts_dynet_graph.embeddings_classifier_w)
    b = dy.parameter(concepts_dynet_graph.embeddings_classifier_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    # verb - nonverb parameters
    verb_w = dy.parameter(concepts_dynet_graph.verb_embeddings_classifier_w)
    verb_b = dy.parameter(concepts_dynet_graph.verb_embeddings_classifier_b)
    nonverb_w = dy.parameter(concepts_dynet_graph.nonverb_embeddings_classifier_w)
    nonverb_b = dy.parameter(concepts_dynet_graph.nonverb_embeddings_classifier_b)

    # initialize last embedding with START_OF_SEQUENCE
    w1_input = None
    if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
        last_concept_embedding = concepts_dynet_graph.concept_nonverb_embeddings[
            concepts_dynet_graph.concepts_nonverbs_vocab.w2i[START_OF_SEQUENCE]]
    else:
        last_concept_embedding = concepts_dynet_graph.concept_embeddings[
            concepts_dynet_graph.concepts_vocab.w2i[START_OF_SEQUENCE]]

    input_matrix = dy.concatenate_cols(encoded_sequence)
    if not hyperparams.use_attention:
        input_matrix = dy.transpose(input_matrix)

    decoder_state, verb_decoder_state, nonverb_decoder_state = initialize_decoders(concepts_dynet_graph,
                                                                                   last_concept_embedding, hyperparams)

    classifier_init = concepts_dynet_graph.classifier.initial_state()

    loss_list = []
    classifier_loss_list = []
    predicted_concepts = []

    # for the cases when loss is computed on dev
    # should this still be here??
    if hyperparams.validation_flag:
        input_matrix = input_matrix * (1 - concepts_dynet_graph.dropout_rate)

    i = 0
    non_attention_i = 0

    for concept in golden_concept_indexes:
        current_concept = golden_concepts[i]
        previous_concept = golden_concepts[i]
        if i != 0:
            previous_concept = golden_concepts[i - 1]

        if hyperparams.use_attention:
            w1_input = w1_input or w1 * input_matrix
            if hyperparams.use_verb_nonverb_decoders:
                # first prediction is START_OF_SEQUENCE => nonverb
                if i == 0:
                    context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
                else:
                    if is_verb(previous_concept) == 1:
                        context_vector = attend(concepts_dynet_graph, input_matrix, verb_decoder_state, w1_input)
                    else:
                        context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
            else:
                context_vector = attend(concepts_dynet_graph, input_matrix, decoder_state, w1_input)
            vector = dy.concatenate([context_vector, last_concept_embedding])
        else:
            vector = dy.concatenate([input_matrix[non_attention_i], last_concept_embedding])

        # dropout when training, but not at validation
        if not hyperparams.validation_flag:
            vector = dy.dropout(vector, concepts_dynet_graph.dropout_rate)

        classifier_loss = classify_verb_nonverb(concepts_dynet_graph, vector, golden_concepts[i], classifier_init)
        classifier_loss_list.append(classifier_loss)

        # TODO: take last_concept_embedding with a probability from golden vs predicted --- error propagation problem
        if hyperparams.use_verb_nonverb_decoders:
            if is_verb(current_concept) == 1:
                verb_decoder_state = verb_decoder_state.add_input(vector)
                out_vector = verb_w * verb_decoder_state.output() + verb_b
            else:
                nonverb_decoder_state = nonverb_decoder_state.add_input(vector)
                out_vector = nonverb_w * nonverb_decoder_state.output() + nonverb_b
        else:
            decoder_state = decoder_state.add_input(vector)
            if hyperparams.use_verb_nonverb_embeddings_classifier:
                if is_verb(golden_concepts[i]) == 1:
                    out_vector = verb_w * decoder_state.output() + verb_b
                else:
                    out_vector = nonverb_w * decoder_state.output() + nonverb_b
            else:
                out_vector = w * decoder_state.output() + b

        probs = dy.softmax(out_vector)
        last_concept_embedding = get_last_concept_embedding(concepts_dynet_graph, concept, is_verb(current_concept), hyperparams)

        # FOR NON ATTENTION -- REMOVE WHEN EXPERIMENTS FINISHED
        if len(encoded_sequence) >= len(golden_concept_indexes):
            non_attention_i += 1

        loss = -dy.log(dy.pick(probs, concept))
        loss_list.append(loss)

        # predict
        probs_vector = probs.npvalue()
        next_concept = dy.np.argmax(probs_vector)
        predicted_concepts.append(get_next_concept(concepts_dynet_graph, is_verb(current_concept), next_concept, hyperparams))

        i += 1

    # remove sequence markers from predicted sequence
    if predicted_concepts[0] == START_OF_SEQUENCE:
        del predicted_concepts[0]
    if len(predicted_concepts) != 0 and predicted_concepts[len(predicted_concepts) - 1] == END_OF_SEQUENCE:
        del predicted_concepts[-1]

    loss_sum = dy.esum(loss_list)
    classifier_loss_sum = dy.esum(classifier_loss_list)
    return loss_sum, predicted_concepts, classifier_loss_sum


def predict_concepts(concepts_dynet_graph, encoded_sequence, hyperparams):
    w = dy.parameter(concepts_dynet_graph.embeddings_classifier_w)
    b = dy.parameter(concepts_dynet_graph.embeddings_classifier_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    # verb - nonverb parameters
    verb_w = dy.parameter(concepts_dynet_graph.verb_embeddings_classifier_w)
    verb_b = dy.parameter(concepts_dynet_graph.verb_embeddings_classifier_b)
    nonverb_w = dy.parameter(concepts_dynet_graph.nonverb_embeddings_classifier_w)
    nonverb_b = dy.parameter(concepts_dynet_graph.nonverb_embeddings_classifier_b)

    # initialize last embedding with START_OF_SEQUENCE
    w1_input = None
    if not hyperparams.use_verb_nonverb_decoders and not hyperparams.use_verb_nonverb_embeddings_classifier:
        last_concept_embedding = concepts_dynet_graph.concept_embeddings[
            concepts_dynet_graph.concepts_vocab.w2i[START_OF_SEQUENCE]]
    else:
        last_concept_embedding = concepts_dynet_graph.concept_nonverb_embeddings[
            concepts_dynet_graph.concepts_nonverbs_vocab.w2i[START_OF_SEQUENCE]]

    input_matrix = dy.concatenate_cols(encoded_sequence)
    if not hyperparams.use_attention:
        input_matrix = dy.transpose(input_matrix)

    decoder_state, verb_decoder_state, nonverb_decoder_state = initialize_decoders(concepts_dynet_graph,
                                                                                   last_concept_embedding, hyperparams)
    classifier_init = concepts_dynet_graph.classifier.initial_state()

    predicted_concepts = []
    count_END_OF_SEQUENCE = 0
    j = 0

    for i in range((len(encoded_sequence) - 1) * 2):
        if count_END_OF_SEQUENCE == 1: break

        if hyperparams.use_attention:
            w1_input = w1_input or w1 * input_matrix
            if hyperparams.use_verb_nonverb_decoders:
                if i == 0:
                    context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
                else:
                    if predict_verb == 1:
                        context_vector = attend(concepts_dynet_graph, input_matrix, verb_decoder_state, w1_input)
                    else:
                        context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
            else:
                context_vector = attend(concepts_dynet_graph, input_matrix, decoder_state, w1_input)
            vector = dy.concatenate([context_vector, last_concept_embedding])
        else:
            vector = dy.concatenate([input_matrix[j], last_concept_embedding])

        # FOR NON ATTENTION -- REMOVE WHEN EXPERIMENTS FINISHED
        if j < len(encoded_sequence) - 1:
            j += 1

        predict_verb = predict_verb_nonverb(concepts_dynet_graph, vector, classifier_init)
        if hyperparams.use_verb_nonverb_decoders:
            if predict_verb == 1:
                verb_decoder_state = verb_decoder_state.add_input(vector)
                out_vector = verb_w * verb_decoder_state.output() + verb_b
            else:
                nonverb_decoder_state = nonverb_decoder_state.add_input(vector)
                out_vector = nonverb_w * nonverb_decoder_state.output() + nonverb_b
        else:
            decoder_state = decoder_state.add_input(vector)
            if hyperparams.use_verb_nonverb_embeddings_classifier:
                if predict_verb == 1:
                    out_vector = verb_w * decoder_state.output() + verb_b
                else:
                    out_vector = nonverb_w * decoder_state.output() + nonverb_b
            else:
                out_vector = w * decoder_state.output() + b

        probs_vector = dy.softmax(out_vector).npvalue()
        next_concept = dy.np.argmax(probs_vector)
        last_concept_embedding = get_last_concept_embedding(concepts_dynet_graph, next_concept, predict_verb, hyperparams)

        if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
            if predict_verb == 0 and concepts_dynet_graph.concepts_nonverbs_vocab.i2w[next_concept] == END_OF_SEQUENCE:
                count_END_OF_SEQUENCE += 1
                j = 0
                continue
        else:
            if concepts_dynet_graph.concepts_vocab.i2w[next_concept] == END_OF_SEQUENCE:
                count_END_OF_SEQUENCE += 1
                j = 0
                continue

        predicted_concepts.append(get_next_concept(concepts_dynet_graph, predict_verb, next_concept, hyperparams))

    # Remove sequence markers from predicted sequence
    if predicted_concepts[0] == START_OF_SEQUENCE:
        del predicted_concepts[0]
    if len(predicted_concepts) != 0 and predicted_concepts[len(predicted_concepts) - 1] == END_OF_SEQUENCE:
        del predicted_concepts[-1]

    return predicted_concepts


def train(concepts_dynet_graph, input_sequence, golden_concepts, hyperparams):
    dy.renew_cg()

    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence, hyperparams)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)

    return decode(concepts_dynet_graph, encoded_sequence, golden_concepts, hyperparams)


def test(concepts_dynet_graph, input_sequence, hyperparams):
    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence, hyperparams)
    encoded_sequence = encode_input_sequence(concepts_dynet_graph, embedded_sequence)

    return predict_concepts(concepts_dynet_graph, encoded_sequence, hyperparams)


def train_sentence(concepts_dynet_graph, sentence, identified_concepts, hyperparams):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    loss, predicted_concepts, classifier_loss = train(concepts_dynet_graph, input_sequence, golden_concepts,
                                                      hyperparams)
    loss_value = loss.value()
    classifier_loss_value = classifier_loss.value()

    if not hyperparams.validation_flag:
        loss.backward()
        if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
            classifier_loss.backward()
        concepts_dynet_graph.trainer.update()

    bleu_score = compute_bleu_score(golden_concepts, predicted_concepts)

    nb_correctly_predicted_concepts, precision, recall, f_score = compute_f_score(golden_concepts, predicted_concepts)

    accuracy, correct_order_percentage, correct_distances_percentage = \
        compute_metrics(golden_concepts, predicted_concepts)

    return predicted_concepts, loss_value, bleu_score, nb_correctly_predicted_concepts, precision, recall, f_score, \
           accuracy, correct_order_percentage, correct_distances_percentage, classifier_loss_value


def test_sentence(concepts_dynet_graph, sentence, identified_concepts, hyperparams):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts = test(concepts_dynet_graph, input_sequence, hyperparams)

    bleu_score = compute_bleu_score(golden_concepts, predicted_concepts)

    nb_correctly_predicted_concepts, precision, recall, f_score = compute_f_score(golden_concepts, predicted_concepts)

    accuracy, correct_order_percentage, correct_distances_percentage = \
        compute_metrics(golden_concepts, predicted_concepts)

    return predicted_concepts, bleu_score, nb_correctly_predicted_concepts, precision, recall, f_score, \
           accuracy, correct_order_percentage, correct_distances_percentage


def run_training(concepts_dynet_graph, hyperparams, train_entries, nb_train_entries, overview_logs, detail_logs):
    sum_train_loss = 0
    sum_train_bleu = 0
    sum_train_nb_correctly_predicted_concepts = 0
    sum_train_precision = 0
    sum_train_recall = 0
    sum_train_f_score = 0
    sum_train_accuracy = 0
    sum_train_correct_order_percentage = 0
    sum_train_correct_distances_percentage = 0
    sum_train_classifier_loss = 0

    train_entry: ConceptsTrainingEntry
    for train_entry in train_entries:
        (predicted_concepts, train_entry_loss, train_entry_bleu, train_entry_nb_correctly_predicted_concepts,
         train_entry_precision, train_entry_recall, train_entry_f_score, train_entry_accuracy,
         train_entry_correct_order_percentage, train_entry_correct_distances_percentage,
         train_entry_classifier_loss) = \
            train_sentence(concepts_dynet_graph, train_entry.sentence, train_entry.identified_concepts, hyperparams)

        sum_train_loss += train_entry_loss
        sum_train_bleu += train_entry_bleu
        sum_train_nb_correctly_predicted_concepts += train_entry_nb_correctly_predicted_concepts
        sum_train_precision += train_entry_precision
        sum_train_recall += train_entry_recall
        sum_train_f_score += train_entry_f_score
        sum_train_accuracy += train_entry_accuracy
        sum_train_correct_order_percentage += train_entry_correct_order_percentage
        sum_train_correct_distances_percentage += train_entry_correct_distances_percentage
        sum_train_classifier_loss += train_entry_classifier_loss

    avg_train_loss = sum_train_loss / nb_train_entries
    avg_train_bleu = sum_train_bleu / nb_train_entries
    avg_train_nb_correctly_predicted_concepts = sum_train_nb_correctly_predicted_concepts / nb_train_entries
    avg_train_precision = sum_train_precision / nb_train_entries
    avg_train_recall = sum_train_recall / nb_train_entries
    avg_train_f_score = sum_train_f_score / nb_train_entries
    avg_train_accuracy = sum_train_accuracy / nb_train_entries
    avg_train_correct_order_percentage = sum_train_correct_order_percentage / nb_train_entries
    avg_train_correct_distances_percentage = sum_train_correct_distances_percentage / nb_train_entries
    avg_train_classifier_loss = sum_train_classifier_loss / nb_train_entries

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
    print("CLASSIFIER LOSS: " + str(avg_train_classifier_loss))

    overview_logs.write("Train LOSS: " + str(avg_train_loss) + "\n")
    overview_logs.write("Train BLEU: " + str(avg_train_bleu) + "\n")
    overview_logs.write("Average number of correctly predicted concepts per train sentence: " +
                        str(avg_train_nb_correctly_predicted_concepts) + "\n")
    overview_logs.write("Train PRECISION: " + str(avg_train_precision) + "\n")
    overview_logs.write("Train RECALL: " + str(avg_train_recall) + "\n")
    overview_logs.write("Train F-SCORE: " + str(avg_train_f_score) + "\n")
    overview_logs.write("Train ACCURACY (correctly predicted concepts on correct positions): " +
                        str(avg_train_accuracy) + "\n")
    overview_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts) train: " +
                        str(avg_train_correct_order_percentage) + "\n")
    overview_logs.write(
        "Percentage of concepts at correct distances (only for correctly predicted concepts) train: " +
        str(avg_train_correct_distances_percentage) + "\n")
    overview_logs.write("Train CLASSIFIER LOSS: " + str(avg_train_classifier_loss) + "\n")
    overview_logs.write("\n")


def run_testing(concepts_dynet_graph, hyperparams, test_entries, nb_test_entries, overview_logs, detail_logs):
    sum_test_bleu = 0
    sum_test_nb_correctly_predicted_concepts = 0
    sum_test_precision = 0
    sum_test_recall = 0
    sum_test_f_score = 0
    sum_test_accuracy = 0
    sum_test_correct_order_percentage = 0
    sum_test_correct_distances_percentage = 0

    test_entry: ConceptsTrainingEntry
    for test_entry in test_entries:
        (test_predicted_concepts, test_entry_bleu, test_entry_nb_correctly_predicted_concepts, test_entry_precision,
         test_entry_recall, test_entry_f_score, test_entry_accuracy, test_entry_correct_order_percentage,
         test_entry_correct_distances_percentage) = \
            test_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts, hyperparams)

        sum_test_bleu += test_entry_bleu
        sum_test_nb_correctly_predicted_concepts += test_entry_nb_correctly_predicted_concepts
        sum_test_precision += test_entry_precision
        sum_test_recall += test_entry_recall
        sum_test_f_score += test_entry_f_score
        sum_test_accuracy += test_entry_accuracy
        sum_test_correct_order_percentage += test_entry_correct_order_percentage
        sum_test_correct_distances_percentage += test_entry_correct_distances_percentage

        detail_logs.write(test_entry.logging_info)
        detail_logs.write("PREDICTED concepts: " + str(test_predicted_concepts) + "\n")
        detail_logs.write("BLEU: " + str(test_entry_bleu) + "\n")
        detail_logs.write("Correctly predicted concepts: " + str(test_entry_nb_correctly_predicted_concepts) + "\n")
        detail_logs.write("PRECISION: " + str(test_entry_precision) + "\n")
        detail_logs.write("RECALL: " + str(test_entry_recall) + "\n")
        detail_logs.write("F-SCORE: " + str(test_entry_f_score) + "\n")
        detail_logs.write("ACCURACY (correctly predicted concepts on correct positions): " +
                              str(test_entry_accuracy) + "\n")
        detail_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts): " +
                              str(test_entry_correct_order_percentage) + "\n")
        detail_logs.write(
            "Percentage of concepts at correct distances (only for correctly predicted concepts): " +
            str(test_entry_correct_distances_percentage) + "\n")
        detail_logs.write("\n")

    avg_test_bleu = sum_test_bleu / nb_test_entries
    avg_test_nb_correctly_predicted_concepts = sum_test_nb_correctly_predicted_concepts / nb_test_entries
    avg_test_precision = sum_test_precision / nb_test_entries
    avg_test_recall = sum_test_recall / nb_test_entries
    avg_test_f_score = sum_test_f_score / nb_test_entries
    avg_test_accuracy = sum_test_accuracy / nb_test_entries
    avg_test_correct_order_percentage = sum_test_correct_order_percentage / nb_test_entries
    avg_test_correct_distances_percentage = sum_test_correct_distances_percentage / nb_test_entries

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

    overview_logs.write("Test BLEU: " + str(avg_test_bleu) + "\n")
    overview_logs.write("Average number of correctly predicted concepts per test sentence: " +
                        str(avg_test_nb_correctly_predicted_concepts) + "\n")
    overview_logs.write("Test PRECISION: " + str(avg_test_precision) + "\n")
    overview_logs.write("Test RECALL: " + str(avg_test_recall) + "\n")
    overview_logs.write("Test F-SCORE: " + str(avg_test_f_score) + "\n")
    overview_logs.write("Test ACCURACY (correctly predicted concepts on correct positions): " +
                        str(avg_test_accuracy) + "\n")
    overview_logs.write("Percentage of concepts in correct order (only for correctly predicted concepts) test: " +
                        str(avg_test_correct_order_percentage) + "\n")
    overview_logs.write(
        "Percentage of concepts at correct distances (only for correctly predicted concepts) test: " +
        str(avg_test_correct_distances_percentage) + "\n")
    overview_logs.write("\n")


def run_validation(concepts_dynet_graph, hyperparams, dev_entries, nb_dev_entries, overview_logs, detail_logs):
    sum_loss = 0
    sum_bleu = 0
    sum_nb_correctly_predicted_concepts = 0
    sum_precision = 0
    sum_recall = 0
    sum_f_score = 0
    sum_accuracy = 0
    sum_correct_order_percentage = 0
    sum_correct_distances_percentage = 0
    sum_classifier_loss = 0

    dev_entry: ConceptsTrainingEntry
    for dev_entry in dev_entries:
        (predicted_concepts, entry_loss, entry_bleu, entry_nb_correctly_predicted_concepts, entry_precision,
         entry_recall, entry_f_score, entry_accuracy, entry_correct_order_percentage,
         entry_correct_distances_percentage, entry_classifier_loss) = \
            train_sentence(concepts_dynet_graph, dev_entry.sentence, dev_entry.identified_concepts, hyperparams)

        sum_loss += entry_loss
        sum_bleu += entry_bleu
        sum_nb_correctly_predicted_concepts += entry_nb_correctly_predicted_concepts
        sum_precision += entry_precision
        sum_recall += entry_recall
        sum_f_score += entry_f_score
        sum_accuracy += entry_accuracy
        sum_correct_order_percentage += entry_correct_order_percentage
        sum_correct_distances_percentage += entry_correct_distances_percentage
        sum_classifier_loss += entry_classifier_loss

        detail_logs.write(dev_entry.logging_info)
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
        detail_logs.write("\n")

    avg_loss = sum_loss / nb_dev_entries
    avg_bleu = sum_bleu / nb_dev_entries
    avg_nb_correctly_predicted_concepts = sum_nb_correctly_predicted_concepts / nb_dev_entries
    avg_precision = sum_precision / nb_dev_entries
    avg_recall = sum_recall / nb_dev_entries
    avg_f_score = sum_f_score / nb_dev_entries
    avg_accuracy = sum_accuracy / nb_dev_entries
    avg_correct_order_percentage = sum_correct_order_percentage / nb_dev_entries
    avg_correct_distances_percentage = sum_correct_distances_percentage / nb_dev_entries
    avg_classifier_loss = sum_classifier_loss / nb_dev_entries

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
    print("Golden test CLASSIFIER LOSS: " + str(avg_classifier_loss))

    overview_logs.write("Golden test LOSS: " + str(avg_loss) + "\n")
    overview_logs.write("Golden test BLEU: " + str(avg_bleu) + "\n")
    overview_logs.write("Average number of correctly predicted concepts per golden test sentence: " +
                        str(avg_nb_correctly_predicted_concepts) + "\n")
    overview_logs.write("Golden test PRECISION: " + str(avg_precision) + "\n")
    overview_logs.write("Golden test RECALL: " + str(avg_recall) + "\n")
    overview_logs.write("Golden test F-SCORE: " + str(avg_f_score) + "\n")
    overview_logs.write("Golden test ACCURACY (correctly predicted concepts on correct positions): " +
                        str(avg_accuracy) + "\n")
    overview_logs.write(
        "Percentage of concepts in correct order (only for correctly predicted concepts) golden test: " +
        str(avg_correct_order_percentage) + "\n")
    overview_logs.write(
        "Percentage of concepts at correct distances (only for correctly predicted concepts) golden test: " +
        str(avg_correct_distances_percentage) + "\n")
    overview_logs.write("Golden test CLASSIFIER LOSS: " + str(avg_classifier_loss) + "\n")
    overview_logs.write("\n")

CONCEPTS_MODELS_PATH = PROJECT_ROOT_DIR + \
                  '/trainers/nodes/concept_extraction/sequence_to_sequence_ordered_concept_extraction/concept_extractor_models/'


def run_experiments(hyperparams):
    train_entries, nb_train_entries, dev_entries, nb_dev_entries = read_train_dev_data(hyperparams.alignment, hyperparams)

    all_concepts_vocab, all_verbs_vocab, all_nonverbs_vocab, all_dev_concepts_vocab, all_words_vocab = \
        create_vocabs(train_entries, dev_entries)

    word_glove_embeddings = read_glove_embeddings_from_file(hyperparams.words_glove_embedding_size)
    words_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                          hyperparams.words_glove_embedding_size,
                                                                          all_words_vocab)

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, words_glove_embeddings_list,
                                              all_verbs_vocab, all_nonverbs_vocab, hyperparams,
                                              all_dev_concepts_vocab)

    model_name = get_model_name(hyperparams)

    logs_path = "concept_extractor_logs/" + model_name

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # log files
    detail_logs_file_name = logs_path + "/concept_extractor_detailed_logs.txt"
    detail_test_logs_file_name = logs_path + "/concept_extractor_detailed_test_logs.txt"
    overview_logs_file_name = logs_path + "/concept_extractor_overview_logs.txt"

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    detail_test_logs = open(detail_test_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")

    for epoch in range(1, hyperparams.nb_epochs + 1):
        print("Epoch " + str(epoch) + "\n")
        detail_logs.write("Epoch " + str(epoch) + "\n\n")
        detail_test_logs.write("Epoch " + str(epoch) + "\n\n")
        overview_logs.write("Epoch " + str(epoch) + "\n\n")

        hyperparams.validation_flag = False
        run_training(concepts_dynet_graph, hyperparams, train_entries, nb_train_entries, overview_logs, detail_logs)

        hyperparams.validation_flag = True
        run_validation(concepts_dynet_graph, hyperparams, dev_entries, nb_dev_entries, overview_logs, detail_logs)

        run_testing(concepts_dynet_graph, hyperparams, dev_entries, nb_dev_entries, overview_logs, detail_test_logs)

    models_path = CONCEPTS_MODELS_PATH + model_name
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # save vocabs
    with open(models_path + "/words_vocab", "wb") as f:
        pickle.dump(all_words_vocab, f)
    with open(models_path + "/concepts_vocab", "wb") as f:
        pickle.dump(all_concepts_vocab, f)
    with open(models_path + "/verbs_vocab", "wb") as f:
        pickle.dump(all_verbs_vocab, f)
    with open(models_path + "/nonverbs_vocab", "wb") as f:
        pickle.dump(all_nonverbs_vocab, f)
    with open(models_path + "/dev_concepts_vocab", "wb") as f:
        pickle.dump(all_dev_concepts_vocab, f)
    with open(models_path + "/glove_embeddings_list", "wb") as f:
        pickle.dump(words_glove_embeddings_list, f)

    # save model
    concepts_dynet_graph.model.save(models_path + "/graph")

    print("Done")
    detail_logs.close()
    detail_test_logs.close()
    overview_logs.close()


def training(hyperparams):
    train_entries, nb_train_entries, dev_entries, nb_dev_entries = read_train_dev_data(hyperparams.alignment, hyperparams)

    train_entries += dev_entries
    nb_train_entries += nb_dev_entries

    all_concepts_vocab, all_verbs_vocab, all_nonverbs_vocab, all_dev_concepts_vocab, all_words_vocab = \
        create_vocabs(train_entries, dev_entries)

    word_glove_embeddings = read_glove_embeddings_from_file(hyperparams.words_glove_embedding_size)
    words_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                          hyperparams.words_glove_embedding_size,
                                                                          all_words_vocab)

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, words_glove_embeddings_list,
                                              all_verbs_vocab, all_nonverbs_vocab, hyperparams,
                                              all_dev_concepts_vocab)

    model_name = get_model_name(hyperparams)

    logs_path = "concept_extractor_logs/" + model_name

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    # log files
    detail_logs_file_name = logs_path + "/concept_extractor_detailed_logs.txt"
    overview_logs_file_name = logs_path + "/concept_extractor_overview_logs.txt"

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")

    for epoch in range(1, hyperparams.nb_epochs + 1):
        print("Epoch " + str(epoch) + "\n")
        detail_logs.write("Epoch " + str(epoch) + "\n\n")
        overview_logs.write("Epoch " + str(epoch) + "\n\n")

        hyperparams.validation_flag = False
        run_training(concepts_dynet_graph, hyperparams, train_entries, nb_train_entries, overview_logs, detail_logs)

    models_path = CONCEPTS_MODELS_PATH + model_name
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # save vocabs
    with open(models_path + "/words_vocab", "wb") as f:
        pickle.dump(all_words_vocab, f)
    with open(models_path + "/concepts_vocab", "wb") as f:
        pickle.dump(all_concepts_vocab, f)
    with open(models_path + "/verbs_vocab", "wb") as f:
        pickle.dump(all_verbs_vocab, f)
    with open(models_path + "/nonverbs_vocab", "wb") as f:
        pickle.dump(all_nonverbs_vocab, f)
    with open(models_path + "/dev_concepts_vocab", "wb") as f:
        pickle.dump(all_dev_concepts_vocab, f)
    with open(models_path + "/glove_embeddings_list", "wb") as f:
        pickle.dump(words_glove_embeddings_list, f)

    # save model
    concepts_dynet_graph.model.save(models_path + "/graph")

    print("Done")
    detail_logs.close()
    overview_logs.close()


def load_concepts_model(hyperparams, model = None):
    model_name = get_model_name(hyperparams)
    models_path = CONCEPTS_MODELS_PATH + model_name
    default_model_name = "WRITE SOME NAME HERE"
    if not os.path.exists(models_path):
        print("No such trained model ("+models_path+"). Loading default model: " + default_model_name)
        return None
    else:
        # get vocabs
        with open(models_path + "/words_vocab", "rb") as f:
            all_words_vocab = pickle.load(f)
        with open(models_path + "/concepts_vocab", "rb") as f:
            all_concepts_vocab = pickle.load(f)
        with open(models_path + "/verbs_vocab", "rb") as f:
            all_verbs_vocab = pickle.load(f)
        with open(models_path + "/nonverbs_vocab", "rb") as f:
            all_nonverbs_vocab = pickle.load(f)
        with open(models_path + "/dev_concepts_vocab", "rb") as f:
            all_dev_concepts_vocab = pickle.load(f)
        with open(models_path + "/glove_embeddings_list", "rb") as f:
            words_glove_embeddings_list = pickle.load(f)

        # create graph
        concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, words_glove_embeddings_list,
                                                  all_verbs_vocab, all_nonverbs_vocab, hyperparams,
                                                  all_dev_concepts_vocab, model)

        # get model
        concepts_dynet_graph.model.populate(models_path + "/graph")

        return concepts_dynet_graph


def testing(hyperparams):

    concepts_dynet_graph = load_concepts_model(hyperparams)

    if concepts_dynet_graph is not None:

        test_entries: [ConceptsTrainingEntry]
        test_entries, nb_test_entries = read_test_data(hyperparams.alignment, hyperparams)

        model_name = get_model_name(hyperparams)
        logs_path = "concept_extractor_logs/test/" + model_name

        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # log files
        detail_logs_file_name = logs_path + "/concept_extractor_detailed_logs.txt"
        overview_logs_file_name = logs_path + "/concept_extractor_overview_logs.txt"

        # open log files
        detail_logs = open(detail_logs_file_name, "w")
        overview_logs = open(overview_logs_file_name, "w")

        run_testing(concepts_dynet_graph, hyperparams, test_entries, nb_test_entries, overview_logs, detail_logs)

    print("Done")


if __name__ == "__main__":

    hyperparams = ConceptsTrainerHyperparameters(encoder_nb_layers=ENCODER_NB_LAYERS,
                                                 decoder_nb_layers=DECODER_NB_LAYERS,
                                                 verb_nonverb_classifier_nb_layers=VERB_NONVERB_CLASSIFIER_NB_LAYERS,
                                                 words_embedding_size=WORDS_EMBEDDING_SIZE,
                                                 words_glove_embedding_size=WORDS_GLOVE_EMBEDDING_SIZE,
                                                 concepts_embedding_size=CONCEPTS_EMBEDDING_SIZE,
                                                 encoder_state_size=ENCODER_STATE_SIZE,
                                                 decoder_state_size=DECODER_STATE_SIZE,
                                                 verb_nonverb_classifier_state_size=VERB_NONVERB_CLASSIFIER_STATE_SIZE,
                                                 attention_size=ATTENTION_SIZE,
                                                 dropout_rate=DROPOUT_RATE,
                                                 use_attention=USE_ATTENTION,
                                                 use_glove=USE_GLOVE,
                                                 use_verb_nonverb_decoders=USE_VERB_NONVERB_DECODERS,
                                                 use_verb_nonverb_embeddings_classifier=USE_VERB_NONVERB_EMBEDDINGS_CLASSIFIER,
                                                 nb_epochs=NB_EPOCHS,
                                                 alignment=ALIGNMENT,
                                                 validation_flag=False,
                                                 experimental_run=EXPERIMENTAL_RUN,
                                                 train=TRAIN)
    if EXPERIMENTAL_RUN:
        run_experiments(hyperparams)
    elif TRAIN:
        training(hyperparams)
    else:
        testing(hyperparams)


