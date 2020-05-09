import dynet as dy
from models.amr_data import CustomizedAMR
from models.concept import IdentifiedConcepts

# TDOD:
# - Rename file so it fits the rest :) concept_extractor
# - Personalize preprocessing (consider the input that comes)
# - Figure out out-of-index problem in generate at big losses (even though it shouldn't be a problem once the model is trained)
# - Revise model
# - Add last predicted concept?
# - Use tag info?

EOS = "<EOS>"

# USE FLORIN's METHODS support_dynet

# Dictionary to hold an index for each word (basically one-hot?)
word2index = {}
# List to get word based on index
index2word = []

# Should vocabulary size depend on the nb of different words in corpus? I think it should
# Initially wanted to make it 400000 (nb of vectors in GloVe, but this seems to be it's limit before I get SIGKILL)
VOCAB_SIZE = 10000
LSTM_NUM_OF_LAYERS = 2

EMBEDDINGS_SIZE = 200
STATE_SIZE = 200
ATTENTION_SIZE = 200

model = dy.Model()

enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + EMBEDDINGS_SIZE, STATE_SIZE, model)

input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))

attention_w1 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2))
attention_w2 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2))
attention_v = model.add_parameters((1, ATTENTION_SIZE))

decoder_w = model.add_parameters((VOCAB_SIZE, STATE_SIZE))
decoder_b = model.add_parameters((VOCAB_SIZE))

output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))

def embed_sentence(sentence):
    # Sentence already comes as list of words
    sentence = list(sentence) + [EOS]

    global input_lookup

    sentence = [word2index[word] for word in sentence]
    return [input_lookup[index] for index in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


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
        # print(probs)
        last_output_embeddings = output_lookup[next_word]
        # print(last_output_embeddings)
        # print(next_word)
        # print(index2word[next_word])
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


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = "It should be vigorously advocated ."

def train(model):
    trainer = dy.SimpleSGDTrainer(model)

    # Train Sentence
    train_sentence = "It should be vigorously advocated ."

    # Create concept sequence for train sentence
    custom_amr: CustomizedAMR = CustomizedAMR()
    custom_amr.tokens_to_concepts_dict = {0: ('i', 'it'),
                                          1: ('r', 'recommend-01'),
                                          3: ('v', 'vigorous'),
                                          4: ('a', 'advocate-01')}
    custom_amr.tokens_to_concept_list_dict = {0: [('i', 'it')],
                                              1: [('r', 'recommend-01')],
                                              3: [('v', 'vigorous')],
                                              4: [('a', 'advocate-01')]}
    # (child,parent) : (relation, children of child, token aligned to child)
    custom_amr.relations_dict = {('i', 'a'): ('ARG1', [], ['0']),
                                 ('v', 'a'): ('manner', [], ['3']),
                                 ('r', ''): ('', ['a'], ['1']),
                                 ('a', 'r'): ('ARG1', ['i', 'v'], ['4'])}
    custom_amr.parent_dict = {'i': 'a', 'v': 'a', 'a': 'r', 'r': ''}
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_custom_amr('amr_id_1', custom_amr)

    generated_concept_list = [concept.name for concept in generated_concepts.ordered_concepts]

    # Preprocess data (create dictionary + list) --- SHOULD HAVE ITS OWN METHOD
    global word2index

    word2index[EOS] = len(word2index)

    # Should use tokenizer to spilt sentence? Depends on the format I'll be getting it in
    input = train_sentence.split()

    # With multiple sentences, should I first add all the sentences, and then all the concept lists - or sentence - concept list - sentence - concept list?
    for word in input:
         # Should be word2index.get(word) == None? Is it faster?
         if word not in word2index.keys():
             word2index[word] = len(word2index)

    for word in generated_concept_list:
         # Should be word2index.get(word) == None? Is it faster?
         if word not in word2index.keys():
             word2index[word] = len(word2index)

    global index2word

    index2word = list(word2index)
    # Preprocess ends here

    # Training with different sentences looks different
    for i in range(300):
        loss = get_loss(input, generated_concept_list, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print(loss_value)
            if loss_value < 10:
                print(generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))


train(model)