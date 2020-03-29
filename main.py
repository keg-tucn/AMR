from keras_lstm_flow import *

if __name__ == "__main__":

    model_parameters = ModelParameters(no_stack_tokens=3,
                                       no_buffer_tokens=1,
                                       no_dep_features=0,
                                       embeddings_dim=200,
                                       train_epochs=50,
                                       hidden_layer_size=1024,
                                       learning_rate=0.01,
                                       dropout=0.0,
                                       recurrent_dropout=0.0)

    asg_parameters = ASGParameters(asg_alg="simple",
                                   no_swaps=1,
                                   swap_distance=1,
                                   rotate=False)

    parser_parameters = ParserParameters(asg_parameters=asg_parameters,
                                         model_parameters=model_parameters,
                                         max_len=50,
                                         shuffle_data=False,
                                         with_enhanced_dep_info=True,
                                         with_target_semantic_labels=False,
                                         with_reattach=True,
                                         with_gold_concept_labels=True,
                                         with_gold_relation_labels=True)

    ActionSet.actions = SIMPLE_ACTION_SET

    init_util_services()

    # generate_parsed_files(parser_parameters)

    data_sets = ["dfa", "proxy", "all"]

    trial_name = "initial_enhanced_deps"

    for data_set in data_sets:
        train_data_path = test_data_path = data_set
        model_name = get_model_name(parser_parameters, data_set)

        if train_data_path == "all":
            train_data_path = None

        if test_data_path == "all":
            test_data_path = None

        train_file(model_name=model_name, train_case_name=trial_name, train_data_path=train_data_path,
                   test_data_path=test_data_path, parser_parameters=parser_parameters)

        test_file(model_name=model_name, test_case_name=trial_name, test_data_path=test_data_path,
                  parser_parameters=parser_parameters)
