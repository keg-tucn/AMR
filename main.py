from keras_lstm_flow import *

if __name__ == "__main__":
    data_sets = ["bolt", "consensus", "dfa", "proxy", "xinhua", "all"]

    train_data_path = "dfa"
    test_data_path = "dfa"
    trial_name = "full_extended_action_set"

    max_len = 30
    embeddings_dim = 200
    train_epochs = 50
    hidden_layer_size = 1024

    model_name = "{}_epochs={}_maxlen={}_embeddingsdim={}" \
        .format(train_data_path, train_epochs, max_len, embeddings_dim)

    model_parameters = ModelParameters(embeddings_dim=embeddings_dim, train_epochs=train_epochs)

    parser_parameters = ParserParameters(max_len=max_len, with_enhanced_dep_info=False,
                                         with_target_semantic_labels=False, with_reattach=True,
                                         with_gold_concept_labels=True, with_gold_relation_labels=True)

    # generate_parsed_files(parser_parameters)

    word_embeddings_util.init_embeddings_matrix(model_parameters.embeddings_dim)

    if train_data_path == "all":
        train_data_path = None
    if test_data_path == "all":
        test_data_path = None

    train_file(model_name=model_name, train_case_name=trial_name, train_data_path=train_data_path,
               test_data_path=test_data_path, parser_parameters=parser_parameters)

    test_file(model_name=model_name, test_case_name=trial_name, test_data_path=test_data_path,
              parser_parameters=parser_parameters)
