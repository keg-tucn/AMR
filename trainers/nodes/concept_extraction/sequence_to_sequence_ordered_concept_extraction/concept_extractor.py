import dynet_config
dynet_config.set(mem=1024)

import dynet as dy

from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor \
    import generate_concepts_training_data, ConceptsTrainingEntry
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import get_all_paths, plot_train_test_acc_loss
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import get_all_concepts

from deep_dynet import support as ds
from deep_dynet.support import Vocab

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
bleu_smoothing = SmoothingFunction()

# TODO?:
# - Personalize preprocessing (consider the input that comes)
# - Revise model
# - Use tag info?

EOS = "<EOS>"

LSTM_NUM_OF_LAYERS = 2
WORDS_EMBEDDING_SIZE = 150
CONCEPTS_EMBEDDING_SIZE = 150
STATE_SIZE = 150
ATTENTION_SIZE = 150

'''
dyparams = dy.DynetParams()
dyparams.set_mem(2048)
dyparams.init()
'''

class ConceptsDynetGraph:
    def __init__(self, words_vocab, concepts_vocab):
        self.model = dy.Model()
        self.words_vocab: Vocab = words_vocab
        self.concepts_vocab: Vocab = concepts_vocab

        self.input_embeddings = self.model.add_lookup_parameters((words_vocab.size(), WORDS_EMBEDDING_SIZE))
        self.output_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_EMBEDDING_SIZE))

        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, WORDS_EMBEDDING_SIZE, STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, WORDS_EMBEDDING_SIZE, STATE_SIZE, self.model)

        self.dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + CONCEPTS_EMBEDDING_SIZE, STATE_SIZE, self.model)

        self.attention_w1 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2))
        self.attention_w2 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, ATTENTION_SIZE))

        self.decoder_w = self.model.add_parameters((concepts_vocab.size(), STATE_SIZE))
        self.decoder_b = self.model.add_parameters((concepts_vocab.size()))


def embed_sequence(concepts_dynet_graph, sequence):
    # Sentence already comes as list of words
    sequence = list(sequence) + [EOS]

    sequence = [concepts_dynet_graph.words_vocab.w2i[word] for word in sequence]
    return [concepts_dynet_graph.input_embeddings[index] for index in sequence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sequence(concepts_dynet_graph, sequence):
    sequence_rev = list(reversed(sequence))

    fwd_vectors = run_lstm(concepts_dynet_graph.enc_fwd_lstm.initial_state(), sequence)
    bwd_vectors = run_lstm(concepts_dynet_graph.enc_bwd_lstm.initial_state(), sequence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(concepts_dynet_graph, input_mat, state, w1dt):
    w2 = dy.parameter(concepts_dynet_graph.attention_w2)
    v = dy.parameter(concepts_dynet_graph.attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2 * dy.concatenate(list(state.s()))

    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)

    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(concepts_dynet_graph, encoded_sequence, golden_concepts):
    golden_concepts = list(golden_concepts) + [EOS]
    embedded_golden_concepts = [concepts_dynet_graph.concepts_vocab.w2i[concept] for concept in golden_concepts]

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)
    input_mat = dy.concatenate_cols(encoded_sequence)
    w1dt = None

    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]
    s = concepts_dynet_graph.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))
    loss = []

    predicted_concepts = []
    count_EOS = 0

    for concept in embedded_golden_concepts:

        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_concept_embedding])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)

        # TODO: take last_concept_embedding with a probability from golden vs predicted --- error propagation problem
        last_concept_embedding = concepts_dynet_graph.output_embeddings[concept]
        loss.append(-dy.log(dy.pick(probs, concept)))

        # predict
        probs_vec = probs.vec_value()
        next_concept = probs_vec.index(max(probs_vec))

        if concepts_dynet_graph.concepts_vocab.i2w[next_concept] == EOS:
            count_EOS += 1
            continue

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])

    loss = dy.esum(loss)
    return (loss, predicted_concepts)


def predict_concepts(concepts_dynet_graph, input_sequence):
    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence)
    encoded_sequence = encode_sequence(concepts_dynet_graph, embedded_sequence)

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)
    input_mat = dy.concatenate_cols(encoded_sequence)
    w1dt = None

    last_concept_embedding = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]
    s = concepts_dynet_graph.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_concept_embedding]))

    predicted_concepts = []
    count_EOS = 0

    for i in range(len(input_sequence) * 2):
        if count_EOS == 2: break

        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_concept_embedding])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_concept = probs.index(max(probs))
        last_concept_embedding = concepts_dynet_graph.output_embeddings[next_concept]

        if concepts_dynet_graph.concepts_vocab.i2w[next_concept] == EOS:
            count_EOS += 1
            continue

        predicted_concepts.append(concepts_dynet_graph.concepts_vocab.i2w[next_concept])
    return predicted_concepts


def get_loss(concepts_dynet_graph, input_sequence, golden_concepts):
    dy.renew_cg()

    embedded_sequence = embed_sequence(concepts_dynet_graph, input_sequence)
    encoded_sequence = encode_sequence(concepts_dynet_graph, embedded_sequence)
    return decode(concepts_dynet_graph, encoded_sequence, golden_concepts)


def train_sentence(concepts_dynet_graph, sentence, identified_concepts):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    (loss, predicted_concepts) = get_loss(concepts_dynet_graph, input_sequence, golden_concepts)
    loss_value = loss.value()
    loss.backward()
    concepts_dynet_graph.trainer.update()

    # Accuracy
    '''
    Accuracy computation right now: count the right words on right places
    Should consider - length (original vs predicted)?
                    - right words even if not in right position?
    '''
    accuracy = 0

    correct_predictions = 0
    total_predictions = len(predicted_concepts)

    for concept_idx in range(min(len(golden_concepts), len(predicted_concepts))):
        if golden_concepts[concept_idx] == predicted_concepts[concept_idx]:
            correct_predictions += 1

    if total_predictions != 0:
        accuracy = correct_predictions / total_predictions

    #BLEU score
    '''
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu(golden_concepts, predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    # F-score
    '''
    Does not consider if a concept appears multiple times
    '''
    true_pos = len(list(set(golden_concepts) & set(predicted_concepts)))
    false_pos = len(list(set(predicted_concepts).difference(set(golden_concepts))))
    false_neg = len(list(set(golden_concepts).difference(set(predicted_concepts))))
    prec = 0
    recall = 0
    if total_predictions != 0:
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
    f_score = 0
    if prec + recall != 0:
        f_score = 2 * (prec * recall) / (prec + recall)

    return (loss_value, accuracy, bleu_score, f_score)


def test_sentence(concepts_dynet_graph, sentence, identified_concepts):
    input_sequence = sentence.split()
    golden_concepts = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts = predict_concepts(concepts_dynet_graph, input_sequence)

    # Accuracy
    '''
    Accuracy computation right now: count the right words on right places
    Should consider - length (original vs predicted)?
                    - right words even if not in right position?
    '''
    accuracy = 0

    correct_predictions = 0
    total_predictions = len(predicted_concepts)

    for concept_idx in range(min(len(golden_concepts), len(predicted_concepts))):
        if golden_concepts[concept_idx] == predicted_concepts[concept_idx]:
            correct_predictions += 1

    if total_predictions != 0:
        accuracy = correct_predictions / total_predictions

    # BLEU score
    ''' 
    What should be the weights for the n-grams?
    See which smoothing method fits best
    '''
    bleu_score = sentence_bleu(golden_concepts, predicted_concepts,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=bleu_smoothing.method3)

    # F-score
    '''
    Does not consider if a concept appears multiple times
    '''
    true_pos = len(list(set(golden_concepts) & set(predicted_concepts)))
    false_pos = len(list(set(predicted_concepts).difference(set(golden_concepts))))
    false_neg = len(list(set(golden_concepts).difference(set(predicted_concepts))))
    prec = 0
    recall = 0
    if total_predictions != 0:
        prec = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
    f_score = 0
    if prec + recall != 0:
        f_score = 2 * (prec * recall) / (prec + recall)

    return (predicted_concepts, accuracy, bleu_score, f_score)


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

    concepts_dynet_graph = ConceptsDynetGraph(all_words_vocab, all_concepts_vocab)
    no_epochs = 20

    for epoch in range(1, no_epochs + 1):
        print("Epoch " + str(epoch))
        detail_logs.write("Epoch " + str(epoch) + '\n')
        overview_logs.write("Epoch " + str(epoch) + '\n')

        # train
        sum_loss = 0
        sum_train_accuracy = 0
        sum_train_bleu = 0
        sum_train_f = 0
        train_entry: ConceptsTrainingEntry
        for train_entry in train_entries:
            (entry_loss, train_entry_accuracy, train_entry_bleu, train_entry_f) = \
                train_sentence(concepts_dynet_graph, train_entry.sentence, train_entry.identified_concepts)
            sum_loss += entry_loss
            sum_train_accuracy += train_entry_accuracy
            sum_train_bleu += train_entry_bleu
            sum_train_f += train_entry_f

        avg_loss = sum_loss / no_train_entries
        avg_train_accuracy = sum_train_accuracy / no_train_entries
        avg_train_bleu = sum_train_bleu / no_train_entries
        avg_train_f = sum_train_f / no_train_entries

        print("Loss: " + str(avg_loss))
        print("Train accuracy: " + str(avg_train_accuracy))
        print("Train bleu: " + str(avg_train_bleu))
        print("Train F score: " + str(avg_train_f) + '\n')
        overview_logs.write("Loss: " + str(avg_loss) + '\n')
        overview_logs.write("Train accuracy: " + str(avg_train_accuracy) + '\n')
        overview_logs.write("Train bleu: " + str(avg_train_bleu) + '\n')
        overview_logs.write("Train F-score: " + str(avg_train_f) + '\n')

        # test
        sum_accuracy = 0
        sum_bleu = 0
        sum_f_score = 0
        test_entry: ConceptsTrainingEntry
        for test_entry in test_entries:
            (predicted_concepts, entry_accuracy, entry_bleu, entry_f) = \
                test_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts)
            sum_accuracy += entry_accuracy
            sum_bleu += entry_bleu
            sum_f_score += entry_f

            # logging
            detail_logs.write('Entry accuracy: ' + str(entry_accuracy) + '\n')
            detail_logs.write('Predicted concepts: ' + str(predicted_concepts) + '\n')
            detail_logs.write('Entry bleu: ' + str(entry_bleu) + '\n')
            detail_logs.write('Entry F-score: ' + str(entry_f) + '\n')
            detail_logs.write(test_entry.logging_info)


        avg_accuracy = sum_accuracy / no_test_entries
        avg_bleu = sum_bleu / no_test_entries
        avg_f = sum_f_score / no_test_entries
        print("Test accuracy: " + str(avg_accuracy))
        print("Test bleu: " + str(avg_bleu))
        print("Test F-score: " + str(avg_f) + '\n')
        overview_logs.write("Test accuracy: " + str(avg_accuracy) + '\n')
        overview_logs.write("Test bleu: " + str(avg_bleu) + '\n')
        overview_logs.write("Test F-score: " + str(avg_f) + '\n')

        plotting_data[epoch] = (avg_loss, avg_train_accuracy, avg_accuracy)      # 0 is avg train accuracy

    print("Done")
    detail_logs.close()
    overview_logs.close()
    plot_train_test_acc_loss(plotting_data)