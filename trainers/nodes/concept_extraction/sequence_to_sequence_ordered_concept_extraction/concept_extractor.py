# For memory problems
import dynet_config
# dynet_config.set(mem=1024)

import dynet as dy

from data_extraction.dataset_reading_util import get_all_paths
from data_extraction.word_embeddings_reader import read_glove_embeddings_from_file
from deep_dynet import support as ds
from deep_dynet.support import Vocab
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import get_all_concepts
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import \
    construct_concept_glove_embeddings_list
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.trainer_util import is_verb, \
    compute_f_score, compute_metrics, ConceptsTrainerHyperparameters, get_golden_concept_indexes, initialize_decoders, \
    generate_verbs_nonverbs, get_next_concept, get_last_concept_embedding
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor import \
    generate_concepts_training_data, ConceptsTrainingEntry

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# TODO:
# - MOVE LOGS AND VOCAB CREATIONS FROM MAIN INTO FUNCTIONS
# - USE ROUGE SCORE
# - Save trained model parameters to files
# - Include Hyperparam option for choosing alignment type
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
USE_PREPROCESSING = True
USE_VERB_NONVERB_CLASSIFICATION = True

MAX_SENTENCE_LENGTH = 50
NB_EPOCHS = 40

bleu_smoothing = SmoothingFunction()


class ConceptsDynetGraph:
    def __init__(self, words_vocab, concepts_vocab, words_glove_embeddings_list, concepts_verbs_vocab,
                 concepts_nonverbs_vocab, hyperparams, test_concepts_vocab):
        self.model = dy.Model()

        # BASE MODEL PARAMETERS
        # VOCABS
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab
        # REMOVE WHEN LOSS NOT COMPUTED FOR DEV
        self.test_concepts_vocab: Vocab = test_concepts_vocab

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
        self.decoder_w = self.model.add_parameters((concepts_vocab.size(), hyperparams.decoder_state_size))
        self.decoder_b = self.model.add_parameters((concepts_vocab.size()))
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
        self.verb_decoder_w = self.model.add_parameters((concepts_verbs_vocab.size(), hyperparams.decoder_state_size))
        self.verb_decoder_b = self.model.add_parameters((concepts_verbs_vocab.size()))
        self.nonverb_decoder = dy.GRUBuilder(hyperparams.decoder_nb_layers,
                                             hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                             hyperparams.decoder_state_size, self.model)
        self.nonverb_decoder_w = self.model.add_parameters(
            (concepts_nonverbs_vocab.size(), hyperparams.decoder_state_size))
        self.nonverb_decoder_b = self.model.add_parameters((concepts_nonverbs_vocab.size()))

        # CLASSIFIER
        self.classifier = dy.GRUBuilder(hyperparams.verb_nonverb_classifier_nb_layers,
                                        hyperparams.encoder_state_size * 2 + hyperparams.concepts_embedding_size,
                                        hyperparams.verb_nonverb_classifier_state_size, self.model)
        self.classifier_w = self.model.add_parameters(
            (concepts_vocab.size(), hyperparams.verb_nonverb_classifier_state_size))
        self.classifier_b = self.model.add_parameters((concepts_vocab.size()))

        # TRAINER
        self.trainer = dy.SimpleSGDTrainer(self.model)
        # self.trainer = dy.AdamTrainer(self.model)
        # self.trainer = dy.AdagradTrainer(self.model)


def classify_verb_nonverb(concepts_dynet_graph, input_vector, concept):
    w = dy.parameter(concepts_dynet_graph.classifier_w)
    b = dy.parameter(concepts_dynet_graph.classifier_b)

    out_label = is_verb(concept)

    classifier_init = concepts_dynet_graph.classifier.initial_state()
    classifier_init = classifier_init.add_input(input_vector)

    out_vector = w * classifier_init.output() + b
    probs = dy.softmax(out_vector)
    loss = -dy.log(dy.pick(probs, out_label))

    return loss


def predict_verb_nonverb(concepts_dynet_graph, input_vector):
    w = dy.parameter(concepts_dynet_graph.classifier_w)
    b = dy.parameter(concepts_dynet_graph.classifier_b)

    classifier_init = concepts_dynet_graph.classifier.initial_state()
    classifier_init = classifier_init.add_input(input_vector)

    out_vector = w * classifier_init.output() + b
    probs_vector = dy.softmax(out_vector).vec_value()
    predict_verb = probs_vector.index(max(probs_vector))

    return predict_verb


def embed_sequence(concepts_dynet_graph, sequence, hyperparams):
    # Add START and END markers to sequence
    sequence = [START_OF_SEQUENCE] + list(sequence) + [END_OF_SEQUENCE]

    sequence = [concepts_dynet_graph.words_vocab.w2i[word] for word in sequence]

    if hyperparams.use_glove:
        return [dy.lookup(concepts_dynet_graph.word_glove_embeddings, index, False) for index in sequence]

    return [dy.lookup(concepts_dynet_graph.word_embeddings, index) for index in sequence]


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

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    # verb - nonverb parameters
    verb_w = dy.parameter(concepts_dynet_graph.verb_decoder_w)
    verb_b = dy.parameter(concepts_dynet_graph.verb_decoder_b)
    nonverb_w = dy.parameter(concepts_dynet_graph.nonverb_decoder_w)
    nonverb_b = dy.parameter(concepts_dynet_graph.nonverb_decoder_b)

    # initialize last embedding with START_OF_SEQUENCE
    w1_input = None
    if hyperparams.use_verb_nonverb_classification:
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

    loss_list = []
    classifier_loss_list = []
    predicted_concepts = []

    # REMOVE WHEN LOSS NOT COMPUTED FOR DEV
    if not hyperparams.train_flag:
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
            if hyperparams.use_verb_nonverb_classification:
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

        # dropout -- REMOVE CONDITION WHEN LOSS NOT COMPUTED FOR DEV
        if hyperparams.train_flag:
            vector = dy.dropout(vector, concepts_dynet_graph.dropout_rate)

        # SHOULD THE ALREADY DROPOUT VECTOR GO INTO THE CLASSIFIER AS WELL?
        classifier_loss = classify_verb_nonverb(concepts_dynet_graph, vector, golden_concepts[i])
        classifier_loss_list.append(classifier_loss)

        # TODO: take last_concept_embedding with a probability from golden vs predicted --- error propagation problem
        # SHOULD CREATE A FUNCTION FOR THESE 2 ROWS?
        if hyperparams.use_verb_nonverb_classification:
            if is_verb(current_concept) == 1:
                verb_decoder_state = verb_decoder_state.add_input(vector)
                out_vector = verb_w * verb_decoder_state.output() + verb_b
            else:
                nonverb_decoder_state = nonverb_decoder_state.add_input(vector)
                out_vector = nonverb_w * nonverb_decoder_state.output() + nonverb_b
        else:
            decoder_state = decoder_state.add_input(vector)
            out_vector = w * decoder_state.output() + b

        probs = dy.softmax(out_vector)
        last_concept_embedding = get_last_concept_embedding(concepts_dynet_graph, concept, is_verb(current_concept), hyperparams)

        # FOR NON ATTENTION -- REMOVE WHEN EXPERIMENTS FINISHED
        if len(encoded_sequence) >= len(golden_concept_indexes):
            non_attention_i += 1

        # SHOULD THERE BE DIFFERENT LOSSES FOR THE TWO DECODERS?
        loss = -dy.log(dy.pick(probs, concept))
        loss_list.append(loss)

        # predict
        probs_vector = probs.vec_value()
        next_concept = probs_vector.index(max(probs_vector))
        predicted_concepts.append(get_next_concept(concepts_dynet_graph, is_verb(current_concept), next_concept, hyperparams))

        i += 1

    # remove sequence markers from predicted sequence
    if predicted_concepts[0] == START_OF_SEQUENCE:
        del predicted_concepts[0]
    if predicted_concepts[len(predicted_concepts) - 1] == END_OF_SEQUENCE:
        del predicted_concepts[-1]

    # SEE LOSS WITH AVERAGE?
    loss_sum = dy.esum(loss_list)
    classifier_loss_sum = dy.esum(classifier_loss_list)
    return loss_sum, predicted_concepts, classifier_loss_sum


def predict_concepts(concepts_dynet_graph, encoded_sequence, hyperparams):
    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)

    # verb - nonverb parameters
    verb_w = dy.parameter(concepts_dynet_graph.verb_decoder_w)
    verb_b = dy.parameter(concepts_dynet_graph.verb_decoder_b)
    non_verb_w = dy.parameter(concepts_dynet_graph.nonverb_decoder_w)
    non_verb_b = dy.parameter(concepts_dynet_graph.nonverb_decoder_b)

    # initialize last embedding with START_OF_SEQUENCE
    w1_input = None
    if not hyperparams.use_verb_nonverb_classification:
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

    predicted_concepts = []
    count_END_OF_SEQUENCE = 0
    j = 0

    # dropout
    input_matrix = input_matrix * (1 - concepts_dynet_graph.dropout_rate)

    for i in range((len(encoded_sequence) - 1) * 2):
        if count_END_OF_SEQUENCE == 1: break

        if hyperparams.use_attention:
            w1_input = w1_input or w1 * input_matrix
            if hyperparams.use_verb_nonverb_classification:
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

        predict_verb = predict_verb_nonverb(concepts_dynet_graph, vector)
        if hyperparams.use_verb_nonverb_classification:
            if predict_verb == 1:
                verb_decoder_state = verb_decoder_state.add_input(vector)
                out_vector = verb_w * verb_decoder_state.output() + verb_b
            elif predict_verb == 0:
                nonverb_decoder_state = nonverb_decoder_state.add_input(vector)
                out_vector = non_verb_w * nonverb_decoder_state.output() + non_verb_b
        else:
            decoder_state = decoder_state.add_input(vector)
            out_vector = w * decoder_state.output() + b

        probs_vector = dy.softmax(out_vector).vec_value()
        next_concept = probs_vector.index(max(probs_vector))
        last_concept_embedding = get_last_concept_embedding(concepts_dynet_graph, next_concept, predict_verb, hyperparams)

        if hyperparams.use_verb_nonverb_classification:
            if concepts_dynet_graph.concepts_nonverbs_vocab.i2w[next_concept] == END_OF_SEQUENCE:
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
    if predicted_concepts[len(predicted_concepts) - 1] == END_OF_SEQUENCE:
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

    if hyperparams.train_flag:
        loss.backward()
        if hyperparams.use_verb_nonverb_classification:
            classifier_loss.backward()
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
           accuracy, correct_order_percentage, correct_distances_percentage, classifier_loss_value


def test_sentence(concepts_dynet_graph, sentence, identified_concepts, hyperparams):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts = test(concepts_dynet_graph, input_sequence, hyperparams)

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
           accuracy, correct_order_percentage, correct_distances_percentage


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

    # CREATE FUNCTION FOR CREATING VOCABS !!!
    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    all_concepts = get_all_concepts(train_concepts)
    all_concepts.append(START_OF_SEQUENCE)
    all_concepts.append(END_OF_SEQUENCE)
    all_verbs, all_nonverbs = generate_verbs_nonverbs(all_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)
    all_verbs_vocab = ds.Vocab.from_list(all_verbs)
    all_nonverbs_vocab = ds.Vocab.from_list(all_nonverbs)

    # REMOVE WHEN LOSS NOT COMPUTED FOR DEV
    test_concepts = [test_entry.identified_concepts for test_entry in test_entries]
    all_test_concepts = get_all_concepts(test_concepts)
    all_test_concepts.append(START_OF_SEQUENCE)
    all_test_concepts.append(END_OF_SEQUENCE)
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
    all_words.append(START_OF_SEQUENCE)
    all_words.append(END_OF_SEQUENCE)
    all_words_vocab = ds.Vocab.from_list(all_words)

    word_glove_embeddings = read_glove_embeddings_from_file(WORDS_GLOVE_EMBEDDING_SIZE)
    words_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                          WORDS_GLOVE_EMBEDDING_SIZE,
                                                                          all_words_vocab)

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    detail_test_logs = open(detail_test_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")

    train_flag = False;

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
                                                 use_preprocessing=USE_PREPROCESSING,
                                                 use_verb_nonverb_classification=USE_VERB_NONVERB_CLASSIFICATION,
                                                 max_sentence_length=MAX_SENTENCE_LENGTH,
                                                 nb_epochs=NB_EPOCHS,
                                                 train_flag=train_flag)

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab, words_glove_embeddings_list,
                                              all_verbs_vocab, all_nonverbs_vocab, hyperparams,
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
        sum_train_classifier_loss = 0

        train_entry: ConceptsTrainingEntry
        hyperparams.train_flag = True
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
        # print("Train BLEU: " + str(avg_train_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_train_nb_correctly_predicted_concepts))
        # print("Train PRECISION: " + str(avg_train_precision))
        # print("Train RECALL: " + str(avg_train_recall))
        print("Train F-SCORE: " + str(avg_train_f_score))
        print("Train ACCURACY (correctly predicted concepts on correct positions): " + str(avg_train_accuracy))
        # print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
        #      str(avg_train_correct_order_percentage))
        # print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
        #      str(avg_train_correct_distances_percentage))
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
        sum_classifier_loss = 0

        # test
        sum_test_bleu = 0
        sum_test_nb_correctly_predicted_concepts = 0
        sum_test_precision = 0
        sum_test_recall = 0
        sum_test_f_score = 0
        sum_test_accuracy = 0
        sum_test_correct_order_percentage = 0
        sum_test_correct_distances_percentage = 0

        test_entry: ConceptsTrainingEntry
        hyperparams.train_flag = False
        for test_entry in test_entries:
            # With last_embedding from golden
            (predicted_concepts, entry_loss, entry_bleu, entry_nb_correctly_predicted_concepts, entry_precision,
             entry_recall, entry_f_score, entry_accuracy, entry_correct_order_percentage,
             entry_correct_distances_percentage, entry_classifier_loss) = \
                train_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts, hyperparams)

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
            detail_logs.write("\n")

            # With last_embedding from predictions
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
        avg_classifier_loss = sum_classifier_loss / nb_test_entries

        print("Golden test LOSS: " + str(avg_loss))
        # print("Golden test BLEU: " + str(avg_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_nb_correctly_predicted_concepts))
        # print("Golden test PRECISION: " + str(avg_precision))
        # print("Golden test RECALL: " + str(avg_recall))
        print("Golden test F-SCORE: " + str(avg_f_score))
        print("Golden test ACCURACY (correctly predicted concepts on correct positions): " + str(avg_accuracy))
        # print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
        #      str(avg_correct_order_percentage))
        # print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
        #      str(avg_correct_distances_percentage))
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

        # With last_embedding from predictions
        avg_test_bleu = sum_test_bleu / nb_test_entries
        avg_test_nb_correctly_predicted_concepts = sum_test_nb_correctly_predicted_concepts / nb_test_entries
        avg_test_precision = sum_test_precision / nb_test_entries
        avg_test_recall = sum_test_recall / nb_test_entries
        avg_test_f_score = sum_test_f_score / nb_test_entries
        avg_test_accuracy = sum_test_accuracy / nb_test_entries
        avg_test_correct_order_percentage = sum_test_correct_order_percentage / nb_test_entries
        avg_test_correct_distances_percentage = sum_test_correct_distances_percentage / nb_test_entries

        # print("Test BLEU: " + str(avg_test_bleu))
        print("Average number of correctly predicted concepts per sentence: " +
              str(avg_test_nb_correctly_predicted_concepts))
        # print("Test PRECISION: " + str(avg_test_precision))
        # print("Test RECALL: " + str(avg_test_recall))
        print("Test F-SCORE: " + str(avg_test_f_score))
        print("Test ACCURACY (correctly predicted concepts on correct positions): " + str(avg_test_accuracy))
        # print("Percentage of concepts in correct order (only for correctly predicted concepts): " +
        #      str(avg_test_correct_order_percentage))
        # print("Percentage of concepts at correct distances (only for correctly predicted concepts): " +
        #      str(avg_test_correct_distances_percentage))

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

    print("Done")
    detail_logs.close()
    overview_logs.close()
