import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer

from constants import __AMR_RELATIONS
from definitions import PROJECT_ROOT_DIR, TRAINED_MODELS_DIR, RESULT_METRICS_DIR
from Baseline import reentrancy_restoring
from amr_util import tokenizer_util, word_embeddings_util, keras_plotter
from postprocessing import action_concept_transfer, action_sequence_reconstruction
from smatch import smatch_amr, smatch_util
import models.actions as act
from models.model_parameters import ModelParameters
from models.parser_parameters import ParserParameters
from feature_extraction import feature_vector_generator
from data_extraction import dataset_loader

SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NONE = 5

coref_handling = False

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(5))


def get_predictions_from_distr(predictions_distr):
    predictions = [np.argmax(p) for p in predictions_distr]
    return predictions


def pretty_print_actions(acts_i):
    print actions_to_string(acts_i)


def actions_to_string(acts_i):
    str = ""
    for a in acts_i:
        str += act.acts[a] + " "
    str += "\n"
    return str


def pretty_print_sentence(tokens, index_to_word_map):
    print tokens_to_sentence(tokens, index_to_word_map)


def tokens_to_sentence(tokens, index_to_word_map):
    str = ""
    for t in tokens:
        str += index_to_word_map[t] + " "
    str += "\n"
    return str


def generate_parsed_files():
    dataset_loader.generate_parsed_graphs_files()
    tokenizer_util.generate_tokenizer()
    dataset_loader.generate_parsed_data_files()


def make_prediction(model, x_test, dependencies, parser_parameters):
    tokens_buffer = x_test
    tokens_stack = []

    current_step = 0
    max_len = parser_parameters.max_len

    no_word_index = tokenizer_util.get_no_word_index()

    buffer_token = np.zeros((1, max_len))
    stack_token0 = np.zeros((1, max_len))
    stack_token1 = np.zeros((1, max_len))
    stack_token2 = np.zeros((1, max_len))
    prev_action = np.zeros((1, max_len, 5))

    if parser_parameters.with_enhanced_dep_info:
        dep_info = np.zeros((1, max_len, 6 * len(__AMR_RELATIONS)))
    else:
        dep_info = np.zeros((1, max_len, 6 * 1))

    buffer_token[0][current_step] = tokens_buffer[0]
    stack_token0[0][current_step] = no_word_index
    stack_token1[0][current_step] = no_word_index
    stack_token2[0][current_step] = no_word_index
    prev_action[0][current_step] = [0, 0, 0, 0, 0]

    if parser_parameters.with_enhanced_dep_info:
        dep_info[0][current_step] = np.repeat(feature_vector_generator.oh_encode_amr_rel(None), 6)
    else:
        dep_info[0][current_step] = np.repeat(0, 6)

    final_prediction = []

    while (len(tokens_buffer) != 0 or len(tokens_stack) != 1) and current_step < max_len - 1:
        prediction = model.predict([buffer_token, stack_token0, stack_token1, stack_token2, prev_action, dep_info])
        current_actions_distr_ordered = np.argsort(prediction[0][current_step])[::-1]
        current_inspected_action_index = 0
        current_action = current_actions_distr_ordered[current_inspected_action_index]
        invalid = True
        while invalid:
            if current_inspected_action_index == 5:
                return []
            invalid = False
            current_action = current_actions_distr_ordered[current_inspected_action_index]
            current_inspected_action_index += 1
            if current_action == SH:
                if len(tokens_buffer) == 0:
                    invalid = True
                    continue
                tokens_stack = [tokens_buffer[0]] + tokens_stack
                tokens_buffer = tokens_buffer[1:]
            if current_action == RL:
                if len(tokens_stack) < 2:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[0]] + tokens_stack[2:]
            if current_action == RR:
                if len(tokens_stack) < 2:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[1]] + tokens_stack[2:]
            if current_action == DN:
                if len(tokens_buffer) == 0:
                    invalid = True
                    continue
                tokens_buffer = tokens_buffer[1:]
            if current_action == SW:
                if len(tokens_stack) < 3:
                    invalid = True
                    continue
                tokens_stack = [tokens_stack[0], tokens_stack[2], tokens_stack[1]] + tokens_stack[3:]
        final_prediction.append(current_action)
        current_step += 1
        if len(tokens_buffer) > 0:
            buffer_token[0][current_step] = tokens_buffer[0]
        else:
            buffer_token[0][current_step] = no_word_index

        if len(tokens_stack) > 3:
            stack_token0[0][current_step] = tokens_stack[0]
            stack_token1[0][current_step] = tokens_stack[1]
            stack_token2[0][current_step] = tokens_stack[2]
        else:
            if len(tokens_stack) > 2:
                stack_token0[0][current_step] = tokens_stack[0]
                stack_token1[0][current_step] = tokens_stack[1]
                stack_token2[0][current_step] = no_word_index
            else:
                if len(tokens_stack) > 1:
                    stack_token0[0][current_step] = tokens_stack[0]
                    stack_token1[0][current_step] = no_word_index
                    stack_token2[0][current_step] = no_word_index
                else:
                    stack_token0[0][current_step] = no_word_index
                    stack_token1[0][current_step] = no_word_index
                    stack_token2[0][current_step] = no_word_index

        prev_action[0][current_step] = label_binarizer.transform([current_action])[0, :]

        dep_info[0][current_step] = feature_vector_generator.get_dependency_features(stack_token0[0][current_step],
                                                                                     stack_token1[0][current_step],
                                                                                     stack_token2[0][current_step],
                                                                                     buffer_token[0][current_step],
                                                                                     dependencies, parser_parameters)

    print "Buffer and stack at end of prediction"
    print tokens_buffer
    print tokens_stack
    return final_prediction


def extract_amr_relations_from_dataset(file_path):
    test_data_action_sequences = [d.action_sequence for d in dataset_loader.read_data("test", cache=True)]
    train_data_action_sequences = [d.action_sequence for d in dataset_loader.read_data("training", cache=True)]
    dev_data_action_sequences = [d.action_sequence for d in dataset_loader.read_data("dev", cache=True)]

    action_sequences = test_data_action_sequences + train_data_action_sequences + dev_data_action_sequences

    amr_relations_set = set()
    for action_sequence in action_sequences:
        for action in action_sequence:
            if action.action == "RL" or action.action == "RR":
                amr_relations_set.add(action.label)

    amr_relations_list = list(amr_relations_set)
    amr_relations_list.sort()

    with open(file_path, "w") as f:
        for rel in amr_relations_list:
            f.write("%s\n" % rel)


def get_optimizer(model_parameters):
    lr = model_parameters.learning_rate
    return SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)


def get_model(embedding_matrix, parser_parameters, model_parameters):
    max_len = parser_parameters.max_len

    buffer_input = Input(shape=(max_len,), dtype="int32")
    stack_input_0 = Input(shape=(max_len,), dtype="int32")
    stack_input_1 = Input(shape=(max_len,), dtype="int32")
    stack_input_2 = Input(shape=(max_len,), dtype="int32")
    prev_action_input = Input(shape=(max_len, 5), dtype="float32")

    if parser_parameters.with_enhanced_dep_info:
        dep_info_input = Input(shape=(max_len, 6 * len(__AMR_RELATIONS)), dtype="float32")
    else:
        dep_info_input = Input(shape=(max_len, 6), dtype="float32")

    embedding = Embedding(len(embedding_matrix), model_parameters.embeddings_dim, weights=[embedding_matrix],
                          input_length=max_len, trainable=False)

    buffer_emb = embedding(buffer_input)
    stack_emb_0 = embedding(stack_input_0)
    stack_emb_1 = embedding(stack_input_1)
    stack_emb_2 = embedding(stack_input_2)

    x = concatenate([buffer_emb, stack_emb_0, stack_emb_1, stack_emb_2, prev_action_input, dep_info_input])

    lstm_output = LSTM(model_parameters.hidden_layer_size, return_sequences=True, dropout=model_parameters.dropout,
                       recurrent_dropout=model_parameters.recurrent_dropout)(x)

    if parser_parameters.with_target_semantic_labels:
        dense = TimeDistributed(Dense(5 + 2 * len(__AMR_RELATIONS), activation="softmax"))(lstm_output)
    else:
        dense = TimeDistributed(Dense(5, activation="softmax"))(lstm_output)

    model = Model([buffer_input, stack_input_0, stack_input_1, stack_input_2, prev_action_input, dep_info_input], dense)

    optimizer = get_optimizer(model_parameters)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # plot_model(model, to_file="model.png")
    print model.summary()

    return model


def train(model_name, train_case_name, train_data, test_data, parser_parameters, model_parameters):
    model_path = TRAINED_MODELS_DIR + "/{}_{}".format(model_name, train_case_name)
    print "Model path is: %s" % model_path

    [train_data, test_data] = dataset_loader.partition_dataset((train_data, test_data), partition_sizes=[0.9, 0.1],
                                                               shuffle_data=parser_parameters.shuffle_data)

    print "Overlapping instances before: %d" % dataset_loader.check_data_partitions_overlap(train_data, test_data)
    train_data, test_data = dataset_loader.remove_overlapping_instances(train_data, test_data)
    print "Overlapping instances after: %d" % dataset_loader.check_data_partitions_overlap(train_data, test_data)

    (x_train, y_train, dependencies_train, train_amr_str, train_amr_ids, train_named_entities, train_date_entities) = \
        feature_vector_generator.extract_data_components(train_data)

    (x_test, y_test, dependencies_test, test_amr_str, test_amr_ids, test_named_entities, test_date_entities) = \
        feature_vector_generator.extract_data_components(test_data)

    print "Training data shape: "
    print x_train.shape

    print "Test data shape: "
    print x_test.shape

    # Prepare the proper data set:
    # Input: Buffer top, First three elements on the stack, previous action index, stack[0] deps on stack[1],
    # stack[1] deps on stack[0], stack[0] deps on buffer[0], buffer[0] deps on stack[1], stack[0] deps on stack[2],
    # stack[2] deps on stack[0].
    # If the current action is shift, the next action will have the next token in the buffer and updated stack elements.
    # Else, the same element on the buffer is fed and the elements from the stack are updated
    # Do not consider instances with more than 30 actions for the moment.

    (x_train_full, y_train_full, lengths_train, filtered_count_train) = \
        feature_vector_generator.generate_feature_vectors(x_train, y_train, dependencies_train, train_amr_ids,
                                                          parser_parameters)

    (x_test_full, y_test_full, lengths_test, filtered_count_test) = \
        feature_vector_generator.generate_feature_vectors(x_test, y_test, dependencies_test, test_amr_ids,
                                                          parser_parameters)

    print "Mean length %s " % np.asarray(lengths_train).mean()
    print "Max length %s" % np.asarray(lengths_train).max()
    print "Filtered %s" % filtered_count_train
    print "Final train data shape"
    print (x_train_full.shape)
    print "Final test data shape"
    print (x_test_full.shape)

    embedding_matrix = word_embeddings_util.get_embeddings_matrix(model_parameters.embeddings_dim)

    model = get_model(embedding_matrix, parser_parameters, model_parameters)
    plot_model(model, to_file=PROJECT_ROOT_DIR + "/model.png")

    history = model.fit([x_train_full[:, :, 0], x_train_full[:, :, 1], x_train_full[:, :, 2], x_train_full[:, :, 3],
                         x_train_full[:, :, 4:9], x_train_full[:, :, 9:]], y_train_full,
                        epochs=model_parameters.train_epochs, batch_size=16, validation_split=0.1,
                        callbacks=[
                            ModelCheckpoint(model_path, monitor="val_acc", verbose=0, save_best_only=True,
                                            save_weights_only=False, mode="auto", period=1),
                            EarlyStopping(monitor="val_acc", min_delta=0, patience=model_parameters.train_epochs,
                                          verbose=0, mode="auto")])

    keras_plotter.plot_history(history, model_name, train_case_name)

    model.load_weights(model_path, by_name=False)

    smatch_results = smatch_util.SmatchAccumulator()

    errors = 0

    for i in range(len(x_test)):
        print "%d/%d" % (i, len(x_test))
        # Step1: input a processed test entity test
        prediction = make_prediction(model, x_test[i], dependencies_test[i], parser_parameters)

        if len(prediction) > 0:
            act = action_concept_transfer.ActionConceptTransfer()
            act.load_from_action_objects(y_test[i])
            pred_label = act.populate_new_actions(prediction)
            print "Predictions with old labels: "
            print pred_label

            # Step2: output: Graph respecting the predicted structure
            # Step2": predict concepts
            # Step2"": predict relations
            # Step3: replace named entities & date date_entities

            predicted_amr = action_sequence_reconstruction.reconstruct_all_ne(x_test[i], pred_label,
                                                                              test_named_entities[i],
                                                                              test_date_entities[i], parser_parameters)
            predicted_amr_str = predicted_amr.amr_print()

            # handling coreference(postprocessing)
            if coref_handling:
                predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            # Step4: compute smatch
            original_amr = smatch_amr.AMR.parse_AMR_line(test_amr_str[i])
            predicted_amr = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)

            print "Original Amr"
            print test_amr_str[i]
            print "Predicted Amr"
            print predicted_amr_str

            if original_amr is not None and predicted_amr is not None:
                smatch_f_score = smatch_results.compute_and_add(predicted_amr, original_amr)
                print "Smatch f-score %f" % smatch_f_score
        else:
            errors += 1

    model_accuracy = model.evaluate([x_test_full[:, :, 0], x_test_full[:, :, 1], x_test_full[:, :, 2],
                                     x_test_full[:, :, 3], x_test_full[:, :, 4:9], x_test_full[:, :, 9:]], y_test_full)

    save_trial_results(train_data_shape=x_train.shape, filtered_train_data_shape=x_train_full.shape,
                       train_lengths=lengths_train, test_data_shape=x_test.shape,
                       filtered_test_data_shape=x_test_full.shape, test_lengths=lengths_test,
                       model_accuracy=model_accuracy, smatch_results=smatch_results, errors=errors,
                       model_name=model_name, trial_name=train_case_name)


def test(model_name, test_case_name, data, parser_parameters, model_parameters):
    model_path = TRAINED_MODELS_DIR + "/{}_{}".format(model_name, test_case_name)
    print "Model path is: %s" % model_path

    word_index_map = tokenizer_util.get_word_index_map()
    index_word_map = tokenizer_util.get_index_word_map()

    print "Word index len: "
    print len(word_index_map)

    (x_test, y_test, dependencies_test, test_amr_str, test_amr_ids, test_named_entities, test_date_entities) = \
        feature_vector_generator.extract_data_components(data)

    print "Test data shape: "
    print x_test.shape

    embedding_matrix = word_embeddings_util.get_embeddings_matrix(model_parameters.embeddings_dim)

    x_test_full, y_test_full, lengths_test, filtered_count_test = \
        feature_vector_generator.generate_feature_vectors(x_test, y_test, dependencies_test, test_amr_ids,
                                                          parser_parameters)

    model = get_model(embedding_matrix, parser_parameters, model_parameters)

    print model.summary()
    print "Word embeddings matrix len: "
    print len(embedding_matrix)

    model.load_weights(model_path, by_name=False)

    smatch_results = smatch_util.SmatchAccumulator()

    predictions = []
    errors = 0

    for i in range(len(x_test)):
        print "%d/%d" % (i, len(x_test))
        prediction = make_prediction(model, x_test[i], dependencies_test[i], parser_parameters)

        predictions.append(prediction)
        print "Sentence"
        pretty_print_sentence(x_test[i], index_word_map)
        print "Predicted"
        pretty_print_actions(prediction)
        print "Actual"
        pretty_print_actions([action.index for action in y_test[i]])

        if len(prediction) > 0:
            act = action_concept_transfer.ActionConceptTransfer()
            act.load_from_action_objects(y_test[i])
            pred_label = act.populate_new_actions(prediction)
            print "Predictions with old labels: "
            print pred_label

            predicted_amr = asr.reconstruct_all_ne(x_test[i], pred_label, test_named_entities[i], test_date_entities[i],
                                                   parser_parameters)
            predicted_amr_str = predicted_amr.amr_print()

            if coref_handling:
                predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            original_amr = smatch_amr.AMR.parse_AMR_line(test_amr_str[i])
            predicted_amr = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)

            print "Original Amr"
            print test_amr_str[i]
            print "Predicted AMR"
            print predicted_amr_str

            if original_amr is not None and predicted_amr is not None:
                smatch_f_score = smatch_results.compute_and_add(predicted_amr, original_amr)
                print "Smatch f-score %f" % smatch_f_score

        else:
            errors += 1

    model_accuracy = model.evaluate([x_test_full[:, :, 0], x_test_full[:, :, 1], x_test_full[:, :, 2],
                                     x_test_full[:, :, 3], x_test_full[:, :, 4:9], x_test_full[:, :, 9:]], y_test_full)

    save_trial_results(train_data_shape=None, filtered_train_data_shape=None, train_lengths=None,
                       test_data_shape=x_test.shape, filtered_test_data_shape=x_test_full.shape,
                       test_lengths=lengths_test, model_accuracy=model_accuracy, smatch_results=smatch_results,
                       errors=errors, model_name=model_name, trial_name=test_case_name)

    return predictions


def test_one_sentence(i, amr_str, x, y, named_entities, date_entities, model, dependencies, parser_parameters,
                      smatch_results):
    print "%d" % i
    prediction = make_prediction(model, x, dependencies, parser_parameters)

    if len(prediction) > 0:
        act = action_concept_transfer.ActionConceptTransfer()
        act.load_from_action_objects(y)
        pred_label = act.populate_new_actions(prediction)
        print "Predictions with old labels: "
        print pred_label

        predicted_amr = action_sequence_reconstruction.reconstruct_all_ne(x, pred_label, named_entities, date_entities,
                                                                          parser_parameters)
        predicted_amr_str = predicted_amr.amr_print()

        if coref_handling:
            predicted_amr_str = reentrancy_restoring(predicted_amr_str)

        original_amr = smatch_amr.AMR.parse_AMR_line(amr_str)
        predicted_amr = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)

        print "Original Amr"
        print amr_str
        print "Predicted AMR"
        print predicted_amr_str

        if original_amr is not None and predicted_amr is not None:
            smatch_f_score = smatch_results.compute_and_add(predicted_amr, original_amr)
            print "Smatch f-score %f" % smatch_f_score
        return 0
    else:
        return 1


def save_trial_results(train_data_shape, filtered_train_data_shape, train_lengths,
                       test_data_shape, filtered_test_data_shape, test_lengths,
                       model_accuracy, smatch_results, errors, model_name, trial_name):
    model_path = TRAINED_MODELS_DIR + "/{}".format(model_name)

    if train_data_shape is not None:
        trial_type = "train"
    else:
        trial_type = "test"

    file_name = RESULT_METRICS_DIR + "/{}_results_{}_{}".format(model_name, trial_type, trial_name)

    with open(file_name, "w") as f:
        f.write("------------------------------------------------------------------------------------------------\n")
        if train_data_shape is not None:
            f.write("Train data shape: \n")
            f.write(str(train_data_shape) + "\n")
        f.write("Test data shape: " + "\n")
        f.write(str(test_data_shape) + "\n")

        if filtered_train_data_shape is not None:
            f.write("Final train data shape:" + "\n")
            f.write(str(filtered_train_data_shape) + "\n")
        f.write("Final test data shape:" + "\n")
        f.write(str(filtered_test_data_shape) + "\n")

        if train_data_shape is not None and filtered_train_data_shape is not None:
            f.write("Filtered train: \n")
            f.write(str(train_data_shape[0] - filtered_train_data_shape[0]) + "\n")

        f.write("Filtered test: \n")
        f.write(str(test_data_shape[0] - filtered_test_data_shape[0]) + "\n")

        if train_lengths is not None:
            f.write("Mean train length: %s \n" % np.asarray(train_lengths).mean())
            f.write("Max train length: %s \n" % np.asarray(train_lengths).max())

        f.write("Mean test length: %s \n" % np.asarray(test_lengths).mean())
        f.write("Max test length: %s \n" % np.asarray(test_lengths).max())

        f.write("Scores for model {}\n".format(model_path))

        f.write("Min: %f\n" % np.min(smatch_results.smatch_scores))
        f.write("Max: %f\n" % np.max(smatch_results.smatch_scores))
        f.write("Arithm. mean %s\n" % (smatch_results.smatch_sum / smatch_results.n))
        f.write("Harm. mean %s\n" % (smatch_results.n / smatch_results.inv_smatch_sum))
        f.write("Global smatch f-score %s\n" % smatch_results.smatch_per_node_mean())

        f.write("Model test accuracy\n")
        f.write(str(model_accuracy[1]) + "\n")
        f.write("Errors\n")
        f.write(str(errors) + "\n")


def test_without_amr(model_name, data, parser_parameters, model_parameters):
    model_path = TRAINED_MODELS_DIR + "/{}".format(model_name)
    print "Model path is:"
    print model_path

    sentences = [d[0] for d in data]

    dependencies = [d[1] for d in data]

    named_entities = [d[2] for d in data]

    tokenizer = tokenizer_util.get_tokenizer()
    word_index_map = tokenizer_util.get_word_index_map()
    index_word_map = tokenizer_util.get_index_word_map()
    print "Word index len: "
    print len(word_index_map)

    sequences = np.asarray(tokenizer.texts_to_sequences(sentences))

    indices = np.arange(sequences.shape[0])
    sequences = sequences[indices]

    dependencies = [dependencies[i] for i in indices]

    x_test = sequences
    dependencies_test = dependencies

    print "Test data shape: "
    print x_test.shape
    print len(dependencies_test)

    embedding_matrix = word_embeddings_util.get_embeddings_matrix(model_parameters.embeddings_dim)

    model = get_model(embedding_matrix, parser_parameters, model_parameters)

    print model.summary()

    model.load_weights(model_path, by_name=False)

    for i in range(len(x_test)):
        prediction = make_prediction(model, x_test[i], dependencies_test[i], parser_parameters.max_len)
        print "Sentence"
        pretty_print_sentence(x_test[i], index_word_map)
        print "Predicted"
        pretty_print_actions(prediction)

        if len(prediction) > 0:
            act = action_concept_transfer.ActionConceptTransfer()
            pred_label = act.populate_new_actions(prediction)
            print "AMR skeleton without labels: "
            print pred_label

            if parser_parameters.with_reattach is True:
                predicted_amr_str = action_sequence_reconstruction.reconstruct_all_ne(x_test[i], pred_label,
                                                                                      named_entities, [],
                                                                                      parser_parameters)
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)
            else:
                predicted_amr_str = action_sequence_reconstruction.reconstruct_all_ne(x_test[i], pred_label, [], [],
                                                                                      parser_parameters)
                # handling coreference(postprocessing)
                if coref_handling:
                    predicted_amr_str = reentrancy_restoring(predicted_amr_str)

            print "Predicted Amr"
            print predicted_amr_str

    return prediction


def train_file(model_name, train_case_name, train_data_path, test_data_path, parser_parameters, model_parameters):
    train_data = dataset_loader.read_data("training", train_data_path, cache=True)
    test_data = dataset_loader.read_data("dev", test_data_path, cache=True)

    train(model_name, train_case_name, train_data, test_data, parser_parameters, model_parameters)


def test_file(model_name, test_case_name, test_data_path, parser_parameters, model_parameters):
    test_data = dataset_loader.read_data("test", test_data_path, cache=True)

    return test(model_name, test_case_name, test_data, parser_parameters, model_parameters)


if __name__ == "__main__":
    data_sets = ["bolt", "consensus", "dfa", "proxy", "xinhua", "all"]

    # generate_parsed_files()

    train_data_path = "proxy"
    test_data_path = "proxy"
    trial_name = "full_deoverlapped"

    max_len = 30
    embeddings_dim = 200
    train_epochs = 50
    hidden_layer_size = 1024

    model_name = "{}_epochs={}_maxlen={}_embeddingsdim={}" \
        .format(train_data_path, train_epochs, max_len, embeddings_dim)

    model_parameters = ModelParameters(embeddings_dim=embeddings_dim, train_epochs=train_epochs)

    parser_parameters = ParserParameters(max_len=max_len, with_enhanced_dep_info=False,
                                         with_target_semantic_labels=False, with_reattach=True,
                                         with_gold_concept_labels=False, with_gold_relation_labels=False)

    word_embeddings_util.init_embeddings_matrix(model_parameters.embeddings_dim)

    if train_data_path == "all":
        train_data_path = None
    if test_data_path == "all":
        test_data_path = None

    train_file(model_name=model_name, train_case_name=trial_name, train_data_path=train_data_path,
               test_data_path=test_data_path, parser_parameters=parser_parameters, model_parameters=model_parameters)

    test_file(model_name=model_name, test_case_name=trial_name, test_data_path=test_data_path,
              parser_parameters=parser_parameters, model_parameters=model_parameters)
