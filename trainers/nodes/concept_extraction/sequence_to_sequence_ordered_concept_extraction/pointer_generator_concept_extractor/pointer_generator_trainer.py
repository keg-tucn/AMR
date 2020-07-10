import os
import pickle
from typing import List

import dynet as dy

# read data
from bleu import list_bleu

from data_extraction.dataset_reading_util import get_all_paths_for_alignment
from deep_dynet.support import Vocab

# initialize model
from definitions import PROJECT_ROOT_DIR
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    read_train_test_data, ArcsTraingAndTestData, ArcsTrainingEntry, generate_arcs_training_data
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.pointer_generator_concept_extractor.pointer_generator_trainer_util import \
    create_vocabs_for_pointer_generator_network, EOS, SOS, PointerGeneratorConceptExtractorGraphHyperparams, \
    SGD_trainer, ADAM_trainer
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.trainer_util import \
    compute_f_score, is_verb, compute_metrics

INPUT_EMBEDDINGS_SIZE = 50
OUTPUT_EMBEDDINGS_SIZE = 50
RNN_HIDDEN_STATES = 40
DROPOUT_RATE = 0.4
VERB_NON_VERB_STATE_SIZE = 40

class PointerGeneratorConceptExtractorGraph:
    def __init__(self, words_vocab, concepts_vocab, verbs_concepts, nonverbs_concepts,
                 hyperparams: PointerGeneratorConceptExtractorGraphHyperparams,
                 model=None):
        if model is None:
            global_model = dy.Model()
        else:
            global_model = model
        self.model = global_model.add_subcollection("pointgen")
        self.hyperparams = hyperparams
        # this should be word embeddings
        # TODO: vocabs
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab
        self.verbs_vocab = verbs_concepts
        self.nonverbs_vocab = nonverbs_concepts

        no_input_tokens = len(self.words_vocab.w2i.keys())
        no_output_tokens = len(self.concepts_vocab.w2i.keys())

        self.tokens_lookup_param = self.model.add_lookup_parameters((no_input_tokens, INPUT_EMBEDDINGS_SIZE))
        # this should be concept embeddings
        self.concept_lookup_param = self.model.add_lookup_parameters((no_output_tokens, OUTPUT_EMBEDDINGS_SIZE))
        # params for encoding
        self.fwdGru = dy.GRUBuilder(1, INPUT_EMBEDDINGS_SIZE, RNN_HIDDEN_STATES, self.model)
        self.bwdGru = dy.GRUBuilder(1, INPUT_EMBEDDINGS_SIZE, RNN_HIDDEN_STATES, self.model)
        # params for alignment scores
        self.weights_alignment = self.model.add_parameters((1, RNN_HIDDEN_STATES))
        self.weights_decoder = self.model.add_parameters((RNN_HIDDEN_STATES, RNN_HIDDEN_STATES))
        self.weights_encoder = self.model.add_parameters((RNN_HIDDEN_STATES, 2 * RNN_HIDDEN_STATES))
        # weights for classiffier
        self.weights_classiffier = self.model.add_parameters((no_output_tokens, RNN_HIDDEN_STATES))
        self.bias_classifier = self.model.add_parameters((no_output_tokens))
        # params for decoder
        self.decoderGru = dy.GRUBuilder(1, 2 * RNN_HIDDEN_STATES + OUTPUT_EMBEDDINGS_SIZE, RNN_HIDDEN_STATES,
                                        self.model)
        # if two classifiers
        if hyperparams.two_classifiers:
            # embeddings
            no_verbs = len(self.verbs_vocab.w2i.keys())
            no_nonverbs = len(self.nonverbs_vocab.w2i.keys())
            self.verb_lookup_param = self.model.add_lookup_parameters((no_verbs, OUTPUT_EMBEDDINGS_SIZE))
            self.nonverb_lookup_param = self.model.add_lookup_parameters(
                (no_nonverbs, OUTPUT_EMBEDDINGS_SIZE))
            #classifier verbs/non verbs
            self.classifier = dy.GRUBuilder(1,
                                            RNN_HIDDEN_STATES * 2 + OUTPUT_EMBEDDINGS_SIZE,
                                            VERB_NON_VERB_STATE_SIZE, self.model)
            self.classifier_w = self.model.add_parameters(
                (2, VERB_NON_VERB_STATE_SIZE))
            self.classifier_b = self.model.add_parameters((2))
            #the two classifiers
            self.verb_embeddings_classifier_w = self.model.add_parameters(
                (self.verbs_vocab.size(), RNN_HIDDEN_STATES))
            self.verb_embeddings_classifier_b = self.model.add_parameters((self.verbs_vocab.size()))
            self.nonverb_embeddings_classifier_w = self.model.add_parameters(
                (self.nonverbs_vocab.size(), RNN_HIDDEN_STATES))
            self.nonverb_embeddings_classifier_b = self.model.add_parameters((self.nonverbs_vocab.size()))

        if self.hyperparams.trainer == SGD_trainer:
            self.trainer = dy.SimpleSGDTrainer(self.model)
        elif self.hyperparams.trainer == ADAM_trainer:
            self.trainer = dy.AdamTrainer(self.model)

    # functions
    def get_word_index(self, is_token, token, is_verb = None):
        #TODO: modify for classifiers
        if is_token:
            w2i = self.words_vocab.w2i
        else:
            if hyperparams.two_classifiers:
                if is_verb:
                    w2i = self.verbs_vocab.w2i
                else:
                    w2i = self.nonverbs_vocab.w2i
            else:
                w2i = self.concepts_vocab.w2i
        if token in w2i.keys() or not is_token:
            embedding = w2i[token]
        else:
            embedding = w2i['UNK']
        return embedding

    def embed_input_sentence(self, sentence_tokens: List[str]):
        sentence_embeddings = []
        for sentence_token in sentence_tokens:
            token_index = self.get_word_index(True, sentence_token)
            embedding = self.tokens_lookup_param[token_index]
            sentence_embeddings.append(embedding)
        return sentence_embeddings

    def encode_sentence(self, sentence_embeddings):
        # idea: first show it with .output, then replace it with transduce
        f_init = self.fwdGru.initial_state()
        b_init = self.bwdGru.initial_state()
        fw_exps = f_init.transduce(sentence_embeddings)
        bw_exps = b_init.transduce(reversed(sentence_embeddings))
        bi = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
        input_mat = dy.concatenate_cols(bi)
        first_reversed = bw_exps[0]
        return input_mat, first_reversed

    def calculate_alignment_scores(self, decoder_prev_state, prod_encoder):
        wd = dy.parameter(self.weights_decoder)
        wa = dy.parameter(self.weights_alignment)
        # size n x 1
        # concatenation necessary because .s() returns tuple of expressions
        prod_decoder = wd * dy.concatenate(list(decoder_prev_state.s()))
        alignment_scores = wa * dy.tanh(dy.colwise_add(prod_encoder, prod_decoder))
        return alignment_scores

    def calculate_context_vector(self, decoder_prev_state, encoder_states, prod_encoder):
        # calculate alignment scores
        alignment_scores = self.calculate_alignment_scores(decoder_prev_state, prod_encoder)
        # softmax (calculate annotation weights)
        # need to transpose because softmax is done by column (was row vector)
        annotation_weights = dy.softmax(dy.transpose(alignment_scores))
        # multiply hidden states with alignment scores
        return encoder_states * annotation_weights

    def decode_word(self, decoder_state, last_output_embedding, context_vector, is_train=False):
        # should I concatenate it differentely??
        # TODO: concatenate it differentel
        concatenated_vector = dy.concatenate([context_vector, last_output_embedding])
        if is_train:
            # add dropout
            concatenated_vector = dy.dropout(concatenated_vector, DROPOUT_RATE)
        decoder_state = decoder_state.add_input(concatenated_vector)
        return decoder_state, concatenated_vector

    def get_output_embeddings_prob(self, decoder_output, is_verb):
        if self.hyperparams.two_classifiers:
            if is_verb:
                w = dy.parameter(self.verb_embeddings_classifier_w)
                b = dy.parameter(self.verb_embeddings_classifier_b)
            else:
                w = dy.parameter(self.nonverb_embeddings_classifier_w)
                b = dy.parameter(self.nonverb_embeddings_classifier_b)
        else:
            w = dy.parameter(self.weights_classiffier)
            b = dy.parameter(self.bias_classifier)
        out_vector = w * decoder_output + b
        probs = dy.softmax(out_vector)
        return probs

    def get_classified_embedding(self, index, is_verb):
        if self.hyperparams.two_classifiers:
            if is_verb:
                return self.verb_lookup_param[index]
            else:
                return self.nonverb_lookup_param[index]
        else:
            return self.concept_lookup_param[index]

    def get_predicted_concept(self, index, is_verb):
        if self.hyperparams.two_classifiers:
            if is_verb:
                return self.verbs_vocab.i2w[index]
            else:
                return self.nonverbs_vocab.i2w[index]
        else:
            return self.concepts_vocab.i2w[index]

    def classify_train(self, decoder_output, gold_word_idx, is_verb):
        probs = self.get_output_embeddings_prob(decoder_output,is_verb)
        loss = -dy.log(dy.pick(probs, gold_word_idx))
        last_output_embedding = self.get_classified_embedding(gold_word_idx, is_verb)
        predicted_word_idx = dy.np.argmax(probs.npvalue())
        predicted_word = self.get_predicted_concept(predicted_word_idx, is_verb)
        return last_output_embedding, predicted_word, loss

    def classify_test(self, decoder_output, is_verb):
        probs = self.get_output_embeddings_prob(decoder_output,is_verb)
        predicted_word_idx = dy.np.argmax(probs.npvalue())
        last_output_embedding = self.get_classified_embedding(predicted_word_idx, is_verb)
        predicted_word = self.get_predicted_concept(predicted_word_idx, is_verb)
        return last_output_embedding, predicted_word

    def remove_SOS_EOS(self, input_sentence, gold_concepts, predicted_concepts):
        # remove SOS and EOS from input sentence
        del input_sentence[0]
        del input_sentence[-1]
        # remove SOS and EOS from gold concepts
        del gold_concepts[0]
        del gold_concepts[-1]
        # remove SOS and EOS from predicted concepts (if the case)
        if predicted_concepts[0] == SOS:
            del predicted_concepts[0]
        if predicted_concepts[-1] == EOS:
            del predicted_concepts[-1]

    def get_initial_last_embedding(self):
        if self.hyperparams.two_classifiers:
            return self.nonverb_lookup_param[self.nonverbs_vocab.w2i[SOS]]
        else:
            return self.concept_lookup_param[self.concepts_vocab.w2i[SOS]]

    def classify_verb_nonverb(self, input_vector, concept, classifier_init):
        w = dy.parameter(self.classifier_w)
        b = dy.parameter(self.classifier_b)

        out_label = is_verb(concept)

        # classifier_init = self.classifier.initial_state()
        classifier_init = classifier_init.add_input(input_vector)

        out_vector = w * classifier_init.output() + b
        probs = dy.softmax(out_vector)
        loss = -dy.log(dy.pick(probs, out_label))

        return loss

    def predict_verb_nonverb(self, input_vector, classifier_init):
        w = dy.parameter(self.classifier_w)
        b = dy.parameter(self.classifier_b)

        # classifier_init = self.classifier.initial_state()
        classifier_init = classifier_init.add_input(input_vector)

        out_vector = w * classifier_init.output() + b
        probs_vector = dy.softmax(out_vector).vec_value()
        predict_verb = probs_vector.index(max(probs_vector))

        return predict_verb

    def train_model(self, train_data: List[ArcsTrainingEntry]):
        avg_loss = 0
        avg_fscore = 0
        for train_entry in train_data:
            input_sequence = train_entry.sentence_tokens
            input_sequence.insert(0, SOS)
            input_sequence.append(EOS)
            output_sequence = train_entry.gold_concept_names
            output_sequence.insert(0, SOS)
            output_sequence.append(EOS)
            # build new graph
            dy.renew_cg()

            # encoder
            input_embeddings = self.embed_input_sentence(input_sequence)
            encoded_input, fr = self.encode_sentence(input_embeddings)

            last_output_embedding = self.get_initial_last_embedding()
            decoder_state = self.decoderGru.initial_state().add_input(
                dy.concatenate([dy.vecInput(RNN_HIDDEN_STATES * 2), last_output_embedding]))

            # for each word in the output
            losses = []
            # size n x Tx
            we = dy.parameter(self.weights_encoder)
            prod_encoder = we * encoded_input
            predicted_sequence = []
            classifier_init = None
            if self.hyperparams.two_classifiers:
                classifier_init = self.classifier.initial_state()
            losses_verbs = []
            for output_word in output_sequence:
                context_vector = self.calculate_context_vector(decoder_state, encoded_input, prod_encoder)
                decoder_state, concatenated_vector = self.decode_word(decoder_state, last_output_embedding, context_vector)
                # see if verb or not
                if hyperparams.two_classifiers:
                    loss_vb_non_vb = self.classify_verb_nonverb(concatenated_vector, output_word, classifier_init)
                    losses_verbs.append(loss_vb_non_vb)
                is_concept_verb = is_verb(output_word)
                last_output_embedding, \
                predicted_concept, \
                word_loss = self.classify_train(decoder_state.output(),
                                                self.get_word_index(False, output_word, is_concept_verb), is_concept_verb)
                predicted_sequence.append(predicted_concept)
                losses.append(word_loss)
            self.remove_SOS_EOS(input_sequence, output_sequence, predicted_sequence)

            true_positive, precision, recall, f_score = compute_f_score(output_sequence, predicted_sequence)
            avg_fscore += f_score

            #loss concepts
            loss = dy.esum(losses)
            avg_loss += loss.value()
            loss.backward()

            # loss verbs non verbs
            if hyperparams.two_classifiers:
                loss_vb_non_vb = dy.esum(losses_verbs)
                loss_vb_non_vb.backward()

            self.trainer.update()

        avg_loss = avg_loss / len(train_data)
        avg_fscore = avg_fscore / len(train_data)
        return avg_loss, avg_fscore

    # test_data: pairs of tokenized english, german sentences
    def test_model(self, test_data: List[ArcsTrainingEntry]):
        gold_sentences = []
        predicted_sentences = []
        avg_f_score = 0
        avg_acc = 0
        avg_cop = 0
        avg_cdp = 0
        for train_entry in test_data:
            input_sequence = train_entry.sentence_tokens
            input_sequence.insert(0, SOS)
            input_sequence.append(EOS)
            output_sequence = train_entry.gold_concept_names
            output_sequence.insert(0, SOS)
            output_sequence.append(EOS)
            # build new graph
            dy.renew_cg()
            # encoder
            input_embeddings = self.embed_input_sentence(input_sequence)
            encoded_input, fr = self.encode_sentence(input_embeddings)

            last_output_embedding = self.get_initial_last_embedding()
            decoder_state = self.decoderGru.initial_state().add_input(
                dy.concatenate([dy.vecInput(RNN_HIDDEN_STATES * 2), last_output_embedding]))

            predicted_sequence = []
            predicted_word = None
            no_predictions = 0
            # size n x Tx
            we = dy.parameter(self.weights_encoder)
            prod_encoder = we * encoded_input
            classifier_init = True
            if self.hyperparams.two_classifiers:
                classifier_init = self.classifier.initial_state()
            while predicted_word != EOS and \
                    no_predictions < 2 * len(input_sequence):
                context_vector = self.calculate_context_vector(decoder_state, encoded_input, prod_encoder)
                decoder_state, concatenated_vector = self.decode_word(decoder_state, last_output_embedding, context_vector)
                is_verb = False
                if self.hyperparams.two_classifiers:
                    is_verb = self.predict_verb_nonverb(concatenated_vector, classifier_init)
                last_output_embedding, predicted_word = self.classify_test(decoder_state.output(), is_verb)
                # print predicted embedding
                no_predictions += 1
                predicted_sequence.append(predicted_word)

            self.remove_SOS_EOS(input_sequence, output_sequence, predicted_sequence)
            true_positive, precision, recall, f_score = compute_f_score(output_sequence, predicted_sequence)
            accuracy, correct_order_percentage, correct_distances_percentage = \
                compute_metrics(output_sequence, predicted_sequence)
            avg_f_score += f_score
            avg_acc += accuracy
            avg_cop += correct_order_percentage
            avg_cdp += correct_distances_percentage
            gold_sentences.append(" ".join(output_sequence))
            predicted_sentences.append(" ".join(predicted_sequence))
        # TODO: remove this to see how it impacts the speed
        bleu_score = list_bleu([gold_sentences], predicted_sentences)
        no_entries = len(test_data)
        avg_f_score = avg_f_score / no_entries
        avg_acc = avg_acc / no_entries
        avg_cop = avg_cop / no_entries
        avg_cdp = avg_cdp / no_entries
        return bleu_score, avg_f_score, avg_acc, avg_cop, avg_cdp


POINTGEN_MODELS_PATH = PROJECT_ROOT_DIR + \
                       '/trainers/nodes/concept_extraction/sequence_to_sequence_ordered_concept_extraction/' \
                       'pointer_generator_concept_extractor/pointer_generator_models/'


def save_model(graph: PointerGeneratorConceptExtractorGraph, hyperparams):
    model_name = str(hyperparams)
    models_path = POINTGEN_MODELS_PATH + model_name
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # save vocabs
    with open(models_path + "/words_vocab", "wb") as f:
        pickle.dump(graph.words_vocab, f)
    with open(models_path + "/concepts_vocab", "wb") as f:
        pickle.dump(graph.concepts_vocab, f)
    with open(models_path + "/verbs_vocab", "wb") as f:
        pickle.dump(graph.verbs_vocab, f)
    with open(models_path + "/nonverbs_vocab", "wb") as f:
        pickle.dump(graph.nonverbs_vocab, f)

    # save model
    graph.model.save(models_path + "/graph")


def load_model(hyperparams, model=None):
    model_name = str(hyperparams)
    models_path = POINTGEN_MODELS_PATH + model_name
    with open(models_path + "/words_vocab", "rb") as f:
        all_words_vocab = pickle.load(f)
    with open(models_path + "/concepts_vocab", "rb") as f:
        all_concepts_vocab = pickle.load(f)
    with open(models_path + "/verbs_vocab", "rb") as f:
        verbs_vocab = pickle.load(f)
    with open(models_path + "/nonverbs_vocab", "rb") as f:
        nonverbs_vocab = pickle.load(f)
    # create graph
    concepts_dynet_graph = PointerGeneratorConceptExtractorGraph(all_words_vocab,
                                                                 all_concepts_vocab,
                                                                 verbs_vocab,
                                                                 nonverbs_vocab,
                                                                 hyperparams,
                                                                 model)
    concepts_dynet_graph.model.populate(models_path + "/graph")
    return concepts_dynet_graph


def run_experiment(hyperparams: PointerGeneratorConceptExtractorGraphHyperparams):
    # get train entries
    # TODO: refactor ArcsTraingAndTestData to be used by the concept extraction as well (should contain everything)
    train_test_data: ArcsTraingAndTestData = read_train_test_data(unaligned_tolerance=0,
                                                                  max_sentence_len=hyperparams.max_sentence_len,
                                                                  # I really don't care about this one
                                                                  max_no_parent_vectors=1,
                                                                  # maybe should set it to true in the future
                                                                  use_preprocessing=hyperparams.use_preprocessing,
                                                                  alignment=hyperparams.alignment)

    # construct concept vocabs (from train data)
    all_concepts_vocab, all_words_vocab, verbs_vocab, non_verbs_vocab \
        = create_vocabs_for_pointer_generator_network(train_test_data.train_entries)

    pointGenGraph = PointerGeneratorConceptExtractorGraph(all_words_vocab, all_concepts_vocab,
                                                          verbs_vocab, non_verbs_vocab,
                                                          hyperparams)

    logs_path = 'logs/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    file_path = logs_path + str(hyperparams) + '.txt'
    logs_file = open(file_path, 'w')

    # run training
    for epoch in range(0, hyperparams.no_epochs):
        loss, train_f_score = pointGenGraph.train_model(train_test_data.train_entries)
        bleu_score, avg_f_score, avg_acc, avg_cop, avg_cdp = pointGenGraph.test_model(
            train_test_data.test_entries)
        print('Epoch ' + str(epoch + 1))
        print('Loss ' + str(loss))
        print('Train f-score ' + str(train_f_score))
        print('Bleu ' + str(bleu_score))
        print('Test f-score' + str(avg_f_score))
        print('Avg (fancy) acc ' + str(avg_acc))
        print('Avg COP ' + str(avg_cop))
        print('Avg CDP ' + str(avg_cdp))

        logs_file.write('Epoch ' + str(epoch + 1) + '\n')
        logs_file.write('Loss ' + str(loss) + '\n')
        logs_file.write('Train f-score ' + str(train_f_score) + '\n')
        logs_file.write('Bleu ' + str(bleu_score) + '\n')
        logs_file.write('Test f-score' + str(avg_f_score) + '\n')
        logs_file.write('Avg (fancy) acc ' + str(avg_acc) + '\n')
        logs_file.write('Avg COP ' + str(avg_cop) + '\n')
        logs_file.write('Avg CDP ' + str(avg_cdp) + '\n')

    save_model(pointGenGraph, hyperparams)


def run_training_train_dev(hyperparams: PointerGeneratorConceptExtractorGraphHyperparams):
    train_test_data: ArcsTraingAndTestData = read_train_test_data(unaligned_tolerance=0,
                                                                  max_sentence_len=hyperparams.max_sentence_len,
                                                                  # I really don't care about this one
                                                                  max_no_parent_vectors=1,
                                                                  # maybe should set it to true in the future
                                                                  use_preprocessing=hyperparams.use_preprocessing,
                                                                  alignment=hyperparams.alignment)

    # construct concept vocabs (from train data)
    train_entries = train_test_data.train_entries + train_test_data.test_entries
    all_concepts_vocab, all_words_vocab, verbs_vocab, nonverbs_vocab = create_vocabs_for_pointer_generator_network(train_entries)

    pointGenGraph = PointerGeneratorConceptExtractorGraph(all_words_vocab,
                                                          all_concepts_vocab,
                                                          verbs_vocab,
                                                          nonverbs_vocab,
                                                          hyperparams)

    logs_path = 'logs/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    file_path = logs_path + 'train_' + str(hyperparams) + '.txt'
    logs_file = open(file_path, 'w')

    # run training
    for epoch in range(0, hyperparams.no_epochs):
        loss, train_f_score = pointGenGraph.train_model(train_entries)
        print('Epoch ' + str(epoch + 1))
        print('Loss ' + str(loss))
        print('Train f-score ' + str(train_f_score))

        logs_file.write('Epoch ' + str(epoch + 1) + '\n')
        logs_file.write('Loss ' + str(loss) + '\n')
        logs_file.write('Train f-score ' + str(train_f_score) + '\n')

    save_model(pointGenGraph, hyperparams)


def run_testing_test(hyperparams: PointerGeneratorConceptExtractorGraphHyperparams):
    graph = load_model(hyperparams)

    logs_path = 'logs/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    file_path = logs_path + 'test_' + str(hyperparams) + '.txt'
    logs_file = open(file_path, 'w')

    test_entries, no_failed, hist = generate_arcs_training_data(
        get_all_paths_for_alignment('test', hyperparams.alignment),
        0,
        hyperparams.max_sentence_len,
        1,
        hyperparams.use_preprocessing,
        False)

    # run training
    bleu_score, avg_f_score, avg_acc, avg_cop, avg_cdp = graph.test_model(test_entries)
    print('Bleu ' + str(bleu_score))
    print('Test f-score' + str(avg_f_score))
    print('Avg (fancy) acc ' + str(avg_acc))
    print('Avg COP ' + str(avg_cop))
    print('Avg CDP ' + str(avg_cdp))
    logs_file.write('Bleu ' + str(bleu_score) + '\n')
    logs_file.write('Test f-score' + str(avg_f_score) + '\n')
    logs_file.write('Avg (fancy) acc ' + str(avg_acc) + '\n')
    logs_file.write('Avg COP ' + str(avg_cop) + '\n')
    logs_file.write('Avg CDP ' + str(avg_cdp) + '\n')


if __name__ == "__main__":

    EXP_RUN = True
    TRAIN = False
    hyperparams = PointerGeneratorConceptExtractorGraphHyperparams(no_epochs=40,
                                                                   max_sentence_len=20,
                                                                   use_preprocessing=True,
                                                                   alignment='isi',
                                                                   experimental_run=EXP_RUN,
                                                                   two_classifiers=True,
                                                                   dropout=DROPOUT_RATE,
                                                                   trainer=SGD_trainer)
    if EXP_RUN:
        run_experiment(hyperparams)
    elif TRAIN:
        run_training_train_dev(hyperparams)
    else:
        run_testing_test(hyperparams)
