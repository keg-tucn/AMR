import dynet as dy
import nltk

from deep_dynet import support as ds
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import get_all_concepts

START_OF_SEQUENCE = "<SOS>"
END_OF_SEQUENCE = "<EOS>"


class ConceptsTrainerHyperparameters:
    def __init__(self, encoder_nb_layers,
                 decoder_nb_layers,
                 verb_nonverb_classifier_nb_layers,
                 words_embedding_size,
                 words_glove_embedding_size,
                 concepts_embedding_size,
                 encoder_state_size,
                 decoder_state_size,
                 verb_nonverb_classifier_state_size,
                 attention_size,
                 dropout_rate,
                 use_attention,
                 use_glove,
                 use_verb_nonverb_decoders,
                 use_verb_nonverb_embeddings_classifier,
                 nb_epochs,
                 alignment,
                 validation_flag,
                 experimental_run,
                 train):
        self.encoder_nb_layers = encoder_nb_layers
        self.decoder_nb_layers = decoder_nb_layers
        self.verb_nonverb_classifier_nb_layers = verb_nonverb_classifier_nb_layers
        self.words_embedding_size = words_embedding_size
        self.words_glove_embedding_size = words_glove_embedding_size
        self.concepts_embedding_size = concepts_embedding_size
        self.encoder_state_size = encoder_state_size
        self.decoder_state_size = decoder_state_size
        self.verb_nonverb_classifier_state_size = verb_nonverb_classifier_state_size
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_glove = use_glove
        self.use_verb_nonverb_decoders = use_verb_nonverb_decoders
        self.use_verb_nonverb_embeddings_classifier = use_verb_nonverb_embeddings_classifier
        self.nb_epochs = nb_epochs
        self.alignment = alignment
        self.validation_flag = validation_flag
        self.experimental_run = experimental_run
        self.train = train


def get_model_name(hyperparams):
    model_name = "model_"

    model_type = "base_"
    if hyperparams.use_verb_nonverb_decoders:
        model_type = "vb-nonvb-dec_"
    elif hyperparams.use_verb_nonverb_embeddings_classifier:
        model_type = "vb-nonvb-emb-class_"

    model_name += model_type
    model_name += (hyperparams.alignment + "_")

    if hyperparams.use_glove:
        model_name += ("glove-" + str(hyperparams.words_glove_embedding_size) + "_")
    else:
        model_name += ("trainable-" + str(hyperparams.words_embedding_size) + "_")

    model_name += ("concept-" + str(hyperparams.concepts_embedding_size) + "_")
    model_name += ("enclay-" + str(hyperparams.encoder_nb_layers) + "-encsize-" + str(hyperparams.encoder_state_size) + "_")
    model_name += ("declay-" + str(hyperparams.decoder_nb_layers) + "-decsize-" + str(hyperparams.decoder_state_size) + "_")

    if hyperparams.use_verb_nonverb_decoders:
        model_name += ("classiflay-" + str(hyperparams.verb_nonverb_classifier_nb_layers) + "-classifsize-" +
                       str(hyperparams.verb_nonverb_classifier_state_size) + "_")

    if hyperparams.use_attention:
        model_name += ("attsize-" + str(hyperparams.attention_size) + "_")

    model_name += ("dropout-" + str(hyperparams.dropout_rate) + "_")
    model_name += ("epochs-" + str(hyperparams.nb_epochs))

    if hyperparams.experimental_run:
        model_name += "_noDev"

    return model_name


def get_word_index(concepts_dynet_graph, word):
    if word in concepts_dynet_graph.words_vocab.w2i.keys():
        index = concepts_dynet_graph.words_vocab.w2i[word]
    else:
        index = concepts_dynet_graph.words_vocab.w2i['UNKNOWN']
    return index


def is_verb(concept):
    splitted_concept = concept.split('-')
    if splitted_concept[len(splitted_concept) - 1].isdigit():
        return 1
    return 0


def generate_verbs_nonverbs(concepts):
    verbs = []
    nonverbs = []

    for concept in concepts:
        splitted_concept = concept.split('-')
        if splitted_concept[len(splitted_concept) - 1].isdigit():
            verbs.append(concept)
        else:
            nonverbs.append(concept)

    return verbs, nonverbs


def get_golden_concept_indexes(concepts_dynet_graph, golden_concepts, hyperparams):
    golden_concept_indexes = []

    for concept in golden_concepts:
        if concept in concepts_dynet_graph.concepts_vocab.w2i:
            if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
                if is_verb(concept) == 1:
                    golden_concept_indexes.append(concepts_dynet_graph.concepts_verbs_vocab.w2i[concept])
                else:
                    golden_concept_indexes.append(concepts_dynet_graph.concepts_nonverbs_vocab.w2i[concept])
            else:
                golden_concept_indexes.append(concepts_dynet_graph.concepts_vocab.w2i[concept])
        # REMOVE WHEN LOSS NOT COMPUTED FOR DEV
        else:
            golden_concept_indexes.append(concepts_dynet_graph.dev_concepts_vocab.w2i[concept])

    return golden_concept_indexes


def initialize_decoders(concepts_dynet_graph, last_concept_embedding, hyperparams):
    if hyperparams.use_attention:
        decoder_state = concepts_dynet_graph.decoder.initial_state().add_input(
            dy.concatenate([dy.vecInput(hyperparams.encoder_state_size * 2), last_concept_embedding]))
        verb_decoder_state = concepts_dynet_graph.verb_decoder.initial_state().add_input(
            dy.concatenate([dy.vecInput(hyperparams.encoder_state_size * 2), last_concept_embedding]))
        nonverb_decoder_state = concepts_dynet_graph.nonverb_decoder.initial_state().add_input(
            dy.concatenate([dy.vecInput(hyperparams.encoder_state_size * 2), last_concept_embedding]))
    else:
        decoder_state = concepts_dynet_graph.decoder.initial_state()
        verb_decoder_state = concepts_dynet_graph.verb_decoder.initial_state()
        nonverb_decoder_state = concepts_dynet_graph.nonverb_decoder.initial_state()

    return decoder_state, verb_decoder_state, nonverb_decoder_state


def get_last_concept_embedding(concepts_dynet_graph, concept, is_concept_verb, hyperparams):
    if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
        if is_concept_verb == 1:
            return concepts_dynet_graph.concept_verb_embeddings[concept]
        else:
            return concepts_dynet_graph.concept_nonverb_embeddings[concept]
    else:
        return concepts_dynet_graph.concept_embeddings[concept]


def get_next_concept(concepts_dynet_graph, is_concept_verb, next_concept, hyperparams):
    if hyperparams.use_verb_nonverb_decoders or hyperparams.use_verb_nonverb_embeddings_classifier:
        if is_concept_verb and next_concept in concepts_dynet_graph.concepts_verbs_vocab.i2w:
            return concepts_dynet_graph.concepts_verbs_vocab.i2w[next_concept]
        else:
            return concepts_dynet_graph.concepts_nonverbs_vocab.i2w[next_concept]
    else:
        return concepts_dynet_graph.concepts_vocab.i2w[next_concept]


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


# BLEU-SCORE


def construct_bleu_weights(predicted_sequence_length, golden_sequence_length):
    min_len = min(predicted_sequence_length, golden_sequence_length)
    if min_len >= 4:
        return (0.25, 0.25, 0.25, 0.25)

    weight_value = 1.0 / min_len
    weight_list = [weight_value for i in range(0, min_len)]
    return tuple(weight_list)


def compute_bleu_score(golden_concepts, predicted_concepts):
    bleu_score = 0
    if len(predicted_concepts) != 0:
        bleu_score = nltk.translate.bleu_score.sentence_bleu([golden_concepts], predicted_concepts,
                                                             weights=construct_bleu_weights(len(predicted_concepts),
                                                                                            len(golden_concepts)))
    return bleu_score


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
    accuracy = 0
    if nb_predicted_concepts != 0:
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


def create_vocabs(train_entries, dev_entries):
    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    all_concepts = get_all_concepts(train_concepts)
    all_concepts.append(START_OF_SEQUENCE)
    all_concepts.append(END_OF_SEQUENCE)
    all_verbs, all_nonverbs = generate_verbs_nonverbs(all_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)
    all_verbs_vocab = ds.Vocab.from_list(all_verbs)
    all_nonverbs_vocab = ds.Vocab.from_list(all_nonverbs)

    # REMOVE WHEN LOSS NOT COMPUTED FOR DEV
    dev_concepts = [dev_entry.identified_concepts for dev_entry in dev_entries]
    all_dev_concepts = get_all_concepts(dev_concepts)
    all_dev_concepts.append(START_OF_SEQUENCE)
    all_dev_concepts.append(END_OF_SEQUENCE)
    all_dev_concepts_vocab = ds.Vocab.from_list(all_dev_concepts)

    train_words = []
    for train_entry in train_entries:
        for word in train_entry.sentence.split():
            train_words.append(word)
    dev_words = []
    for test_entry in dev_entries:
        for word in test_entry.sentence.split():
            dev_words.append(word)
    all_words = list(set(train_words + dev_words))
    all_words.append(START_OF_SEQUENCE)
    all_words.append(END_OF_SEQUENCE)
    all_words.append('UNKNOWN')
    all_words_vocab = ds.Vocab.from_list(all_words)

    return all_concepts_vocab, all_verbs_vocab, all_nonverbs_vocab, all_dev_concepts_vocab, all_words_vocab
