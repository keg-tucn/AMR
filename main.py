from keras_lstm_flow import *

if __name__ == "__main__":
    data_sets = ["bolt", "consensus", "dfa", "proxy", "xinhua", "all"]

    train_data_path = test_data_path = "proxy"
    trial_name = "full_deoverlapped"

    max_len = 30
    embeddings_dim = 200
    train_epochs = 50
    hidden_layer_size = 1024

    model_parameters = ModelParameters(embeddings_dim=embeddings_dim, train_epochs=train_epochs)

    parser_parameters = ParserParameters(max_len=max_len, with_enhanced_dep_info=False,
                                         with_target_semantic_labels=False, with_reattach=True,
                                         with_gold_concept_labels=False, with_gold_relation_labels=True)

    init_util_services(model_parameters.embeddings_dim)

    model_name = get_model_name(parser_parameters, train_data_path, trial_name)

    if train_data_path == "all":
        train_data_path = None
    if test_data_path == "all":
        test_data_path = None

    train_file(model_name=model_name, train_case_name=trial_name, train_data_path=train_data_path,
               test_data_path=test_data_path, parser_parameters=parser_parameters)

    test_file(model_name=model_name, test_case_name=trial_name, test_data_path=test_data_path,
              parser_parameters=parser_parameters)

    generate_amr_dicts_files()
    generate_parsed_files(ParserParameters())
