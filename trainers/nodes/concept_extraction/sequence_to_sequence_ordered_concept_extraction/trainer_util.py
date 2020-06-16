import dynet as dy

# from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.concept_extractor import attend


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
                 use_preprocessing,
                 use_verb_nonverb_classification,
                 max_sentence_length,
                 nb_epochs):
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
        self.use_preprocessing = use_preprocessing
        self.use_verb_nonverb_classification = use_verb_nonverb_classification
        self.max_sentence_length = max_sentence_length
        self.nb_epochs = nb_epochs


def is_verb(concept):
    splitted_concept = concept.split('-')
    if splitted_concept[len(splitted_concept) - 1].isdigit():
        return True
    return False


def get_golden_concept_indexes(concepts_dynet_graph, golden_concepts, hyperparams):
    golden_concept_indexes = []

    if hyperparams.use_verb_nonverb_classification:
        for concept in golden_concepts:
            if concept in concepts_dynet_graph.concepts_vocab.w2i:
                if is_verb(concept):
                    golden_concept_indexes.append(concepts_dynet_graph.concepts_verbs_vocab.w2i[concept])
                else:
                    golden_concept_indexes.append(concepts_dynet_graph.concepts_nonverbs_vocab.w2i[concept])
            # Temporary, just until loss is computed on dev too
            else:
                golden_concept_indexes.append(concepts_dynet_graph.test_concepts_vocab.w2i[concept])
    else:
        for concept in golden_concepts:
            if concept in concepts_dynet_graph.concepts_vocab.w2i:
                golden_concept_indexes.append(concepts_dynet_graph.concepts_vocab.w2i[concept])
            # Temporary, just until loss is computed on dev too
            else:
                golden_concept_indexes.append(concepts_dynet_graph.test_concepts_vocab.w2i[concept])
        # embedded_golden_concepts = [concepts_dynet_graph.concepts_vocab.w2i[concept] for concept in golden_concepts]

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


def get_decoder_input(concepts_dynet_graph, input_matrix, w1_input, previous_concept, last_concept_embedding, i, decoder_state, verb_decoder_state, nonverb_decoder_state, hyperparams):
    if hyperparams.use_attention:
        if hyperparams.use_verb_nonverb_classification:
            ####### THIS FEELS WEIRD! ATTENTION GETS DIFFERENT STATES EACH TIME? ----- NOTE TO SELF, WHEN THIS WAS WITHOUT -1, THE CLASSIFIER LOSSES WERE 0.5 and 0.6 instead of 5 or 6
            if i == 0:
                context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
            else:
                if is_verb(previous_concept):
                    context_vector = attend(concepts_dynet_graph, input_matrix, verb_decoder_state, w1_input)
                else:
                    context_vector = attend(concepts_dynet_graph, input_matrix, nonverb_decoder_state, w1_input)
        else:
            context_vector = attend(concepts_dynet_graph, input_matrix, decoder_state, w1_input)
        vector = dy.concatenate([context_vector, last_concept_embedding])
    else:
        vector = dy.concatenate([input_matrix[i], last_concept_embedding])

    return vector


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