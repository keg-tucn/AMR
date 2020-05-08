import dynet as dy

from models.amr_data import CustomizedAMR
from models.concept import IdentifiedConcepts

from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor \
    import generate_concepts_training_data, ConceptsTrainingEntry
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import get_all_paths, plot_train_test_acc_loss
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import get_all_concepts

from deep_dynet import support as ds
from deep_dynet.support import Vocab

# TODO?:
# - Personalize preprocessing (consider the input that comes)
# - Figure out out-of-index problem in generate at big losses (even though it shouldn't be a problem once the model is trained)
# - Revise model
# - Add last predicted concept?
# - Use tag info?

EOS = "<EOS>"

LSTM_NUM_OF_LAYERS = 2
CONCEPTS_EMBEDDING_SIZE = 150
STATE_SIZE = 150
ATTENTION_SIZE = 150

class ConceptsDynetGraph:
    def __init__(self, concepts_vocab):
        self.model = dy.Model()
        self.concepts_vocab: Vocab = concepts_vocab

        self.input_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_EMBEDDING_SIZE))
        self.output_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_EMBEDDING_SIZE))

        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, CONCEPTS_EMBEDDING_SIZE, STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, CONCEPTS_EMBEDDING_SIZE, STATE_SIZE, self.model)

        self.dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + CONCEPTS_EMBEDDING_SIZE, STATE_SIZE, self.model)

        self.attention_w1 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2))
        self.attention_w2 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2))
        self.attention_v = self.model.add_parameters((1, ATTENTION_SIZE))

        self.decoder_w = self.model.add_parameters((concepts_vocab.size(), STATE_SIZE))
        self.decoder_b = self.model.add_parameters((concepts_vocab.size()))

# BEFORE REFACTORING
'''
def embed_sentence(sentence):
    # Sentence already comes as list of words
    sentence = list(sentence) + [EOS]

    global input_lookup

    sentence = [word2index[word] for word in sentence]
    return [input_lookup[index] for index in sentence]
    
    
def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors
    
    
def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

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
    
    
def decode(dec_lstm, vectors, output):
    output = list(output) + [EOS]
    output = [word2index[word] for word in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[word2index[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))
    loss = []

    for word in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        # take last_embedding with a prob from gold vs predicted --- error propagation problem
        last_output_embeddings = output_lookup[word]
        loss.append(-dy.log(dy.pick(probs, word)))
    loss = dy.esum(loss)
    return loss
  
    
def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[word2index[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_word = probs.index(max(probs))

        last_output_embeddings = output_lookup[next_word]
        if index2word[next_word] == EOS:
            count_EOS += 1
            continue

        out.append(index2word[next_word])
    return out
    
    
def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)    
'''


def embed_sentence(concepts_dynet_graph, sentence):
    # Sentence already comes as list of words
    sentence = list(sentence) + [EOS]

    sentence = [concepts_dynet_graph.concepts_vocab.w2i[word] for word in sentence]
    return [concepts_dynet_graph.input_embeddings[index] for index in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(concepts_dynet_graph, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(concepts_dynet_graph.enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(concepts_dynet_graph.enc_bwd_lstm.initial_state(), sentence_rev)
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


def decode(concepts_dynet_graph, vectors, output):
    output = list(output) + [EOS]
    output = [concepts_dynet_graph.concepts_vocab.w2i[word] for word in output]

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]
    s = concepts_dynet_graph.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))
    loss = []

    count_EOS = 0
    out = []

    for word in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)

        probs_vec = probs.vec_value()
        # take last_embedding with a prob from gold vs predicted --- error propagation problem
        last_output_embeddings = concepts_dynet_graph.output_embeddings[word]
        loss.append(-dy.log(dy.pick(probs, word)))

        #predict
        next_word = probs_vec.index(max(probs_vec))
        last_output_embeddings = concepts_dynet_graph.output_embeddings[next_word]

        if concepts_dynet_graph.concepts_vocab.i2w[next_word] == EOS:
            count_EOS += 1
            continue

        out.append(concepts_dynet_graph.concepts_vocab.i2w[next_word])

    loss = dy.esum(loss)
    return (loss, out)


def generate(concepts_dynet_graph, in_seq):
    embedded = embed_sentence(concepts_dynet_graph, in_seq)
    encoded = encode_sentence(concepts_dynet_graph, embedded)

    w = dy.parameter(concepts_dynet_graph.decoder_w)
    b = dy.parameter(concepts_dynet_graph.decoder_b)
    w1 = dy.parameter(concepts_dynet_graph.attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = concepts_dynet_graph.output_embeddings[concepts_dynet_graph.concepts_vocab.w2i[EOS]]
    s = concepts_dynet_graph.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(concepts_dynet_graph, input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_word = probs.index(max(probs))
        last_output_embeddings = concepts_dynet_graph.output_embeddings[next_word]

        if concepts_dynet_graph.concepts_vocab.i2w[next_word] == EOS:
            count_EOS += 1
            continue

        out.append(concepts_dynet_graph.concepts_vocab.i2w[next_word])
    return out


def get_loss(concepts_dynet_graph, input_sentence, output_sentence):
    dy.renew_cg()

    embedded = embed_sentence(concepts_dynet_graph, input_sentence)
    encoded = encode_sentence(concepts_dynet_graph, embedded)
    return decode(concepts_dynet_graph, encoded, output_sentence)


def train_sentence(concepts_dynet_graph, sentence, identified_concepts):

    input_sentence = sentence.split()
    generated_concept_list = [concept.name for concept in identified_concepts.ordered_concepts]

    (loss, predicted_concepts) = get_loss(concepts_dynet_graph, input_sentence, generated_concept_list)
    loss_value = loss.value()
    loss.backward()
    concepts_dynet_graph.trainer.update()

    correct_predictions = 0
    total_predictions = len(predicted_concepts)

    for concept_idx in range(min(len(generated_concept_list), len(predicted_concepts))):
        if generated_concept_list[concept_idx] == predicted_concepts[concept_idx]:
            correct_predictions += 1

    # Accuracy computation right now: count the right words on right places
    # Should consider - length (original vs predicted)?
    #                 - right words even if not in right position?
    accuracy = 0

    if total_predictions != 0:
        accuracy = correct_predictions / total_predictions

    return (loss_value, accuracy)


def test_sentence(concepts_dynet_graph, sentence, identified_concepts):
    input_sentence = sentence.split()
    generated_concept_list = [concept.name for concept in identified_concepts.ordered_concepts]

    predicted_concepts = generate(concepts_dynet_graph, input_sentence)

    correct_predictions = 0
    total_predictions = len(predicted_concepts)

    for concept_idx in range(min(len(generated_concept_list), len(predicted_concepts))):
        if generated_concept_list[concept_idx] == predicted_concepts[concept_idx]:
            correct_predictions += 1

    # Accuracy computation right now: count the right words on right places
    # Should consider - length (original vs predicted)?
    #                 - right words even if not in right position?
    accuracy = 0

    if total_predictions != 0:
        accuracy = correct_predictions / total_predictions

    return (predicted_concepts, accuracy)


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

    # i'm not sure I should include this into the Vocabulary
    dev_concepts = [test_entry.identified_concepts for test_entry in test_entries]

    # when using tokenizer for sentences, use it when embedding the sentence too!!!!!
    train_words = []
    for train_entry in train_entries:
        for word in train_entry.sentence.split():
            train_words.append(word)
    dev_words = []
    for test_entry in test_entries:
        for word in test_entry.sentence.split():
            dev_words.append(word)

    all_concept_names = get_all_concepts(train_concepts + dev_concepts)
    all_concept_names.append(EOS)
    all_concept_words = list(set(train_words + dev_words))
    all_concepts = list(set(all_concept_names + all_concept_words))
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)

    # open log files
    detail_logs = open(detail_logs_file_name, "w")
    overview_logs = open(overview_logs_file_name, "w")
    # init plotting data
    plotting_data = {}

    concepts_dynet_graph = ConceptsDynetGraph(all_concepts_vocab)
    no_epochs = 20

    for epoch in range(1, no_epochs + 1):
        print("Epoch " + str(epoch))
        detail_logs.write("Epoch " + str(epoch) + '\n')
        overview_logs.write("Epoch " + str(epoch) + '\n')

        # train
        sum_loss = 0
        train_entry: ConceptsTrainingEntry
        sum_train_accuracy = 0
        for train_entry in train_entries:
            (entry_loss, train_entry_accuracy) = train_sentence(concepts_dynet_graph, train_entry.sentence, train_entry.identified_concepts)
            sum_loss += entry_loss
            sum_train_accuracy += train_entry_accuracy

        avg_loss = sum_loss / no_train_entries
        avg_train_accuracy = sum_train_accuracy / no_train_entries

        print("Loss " + str(avg_loss) + '\n')
        print("Training accuracy " + str(avg_train_accuracy) + '\n')
        overview_logs.write("Loss " + str(avg_loss) + '\n')
        overview_logs.write("Training accuracy " + str(avg_train_accuracy) + '\n')

        # test
        sum_accuracy = 0
        test_entry: ConceptsTrainingEntry
        for test_entry in test_entries:
            (predicted_concepts, entry_accuracy) = test_sentence(concepts_dynet_graph, test_entry.sentence, test_entry.identified_concepts)
            sum_accuracy += entry_accuracy

            # logging
            detail_logs.write('Entry accuracy: ' + str(entry_accuracy) + '\n')
            detail_logs.write('Predicted concepts: ' + str(predicted_concepts) + '\n')
            detail_logs.write(test_entry.logging_info)


        avg_accuracy = sum_accuracy / no_test_entries
        print("Test accuracy " + str(avg_accuracy) + '\n')
        overview_logs.write("Test accuracy " + str(avg_accuracy) + '\n')
        plotting_data[epoch] = (avg_loss, 0, avg_accuracy)      # 0 is avg train accuracy
    print("Done")
    detail_logs.close()
    overview_logs.close()
    plot_train_test_acc_loss(plotting_data)