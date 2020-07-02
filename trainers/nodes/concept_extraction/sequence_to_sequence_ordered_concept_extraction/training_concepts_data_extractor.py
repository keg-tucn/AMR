from typing import List

from data_extraction import input_file_parser
from data_extraction.dataset_reading_util import get_all_paths, get_all_paths_for_alignment
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept
from pre_post_processing.standford_pre_post_processing import train_pre_processing, inference_preprocessing


class ConceptsTrainingEntry:
    def __init__(self,
                 identified_concepts: IdentifiedConcepts,
                 sentence: str,
                 logging_info: str,
                 amr_str: str):
        self.identified_concepts = identified_concepts
        self.sentence = sentence
        self.logging_info = logging_info
        # needed for smatch
        self.amr_str = amr_str


def generate_dataset_entry(amr_id: str, amr_str: str, sentence: str):
    amr = AMR.parse_string(amr_str)
    # Paul's quickfix
    # if hyperparams.train_flag
    if "080104" not in sentence and "030714" not in sentence and "North Korean media denies involvement" not in sentence:
        amr, new_sentence, metadata = train_pre_processing(amr, sentence)
    else:
        # DON'T FORGET TO USE THIS AT TEST TIME !!!
        # if not hyperparams.train_flag
        # new_sentence, metadata = inference_preprocessing(sentence)
        new_sentence = sentence
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr(amr_id, amr)
    if identified_concepts.ordered_concepts is None:
        return None
    if len(identified_concepts.ordered_concepts) == 0:
        return None

    logging_info = "AMR with id " + amr_id + "\n" + new_sentence + "\n" + \
                   "ORDERED concepts: " + str(identified_concepts.ordered_concepts) + "\n"

    return ConceptsTrainingEntry(identified_concepts, new_sentence, logging_info, amr_str)


def generate_concepts_training_data_per_file(file_path, max_sentence_len=20):
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    entries: List[ConceptsTrainingEntry] = []

    nb_entries_not_processed = 0

    for sentence, amr_str, amr_id in sentence_amr_triples:
        # filter out AMRs with more than max_sentence_len
        # TODO: extract tokens better
        sentence_len = len(sentence.split(" "))
        if sentence_len <= max_sentence_len:
            entry = generate_dataset_entry(amr_id, amr_str, sentence)
            if entry is not None:
                entries.append(entry)
            else:
                nb_entries_not_processed += 1
    return entries, nb_entries_not_processed


def generate_concepts_training_data(file_paths: List[str], max_sentence_len=20):
    all_entries = []

    nb_all_entries_not_processed = 0
    # for file_path in file_paths:
    for i in range(1):
        entries, nb_entries_not_processed = generate_concepts_training_data_per_file(file_paths[i], max_sentence_len)
        all_entries = all_entries + entries
        nb_all_entries_not_processed += nb_entries_not_processed

    return all_entries, nb_all_entries_not_processed


def read_train_dev_data(alignment):
    train_entries, nb_train_failed = generate_concepts_training_data(get_all_paths_for_alignment('training', alignment))
    nb_train_entries = len(train_entries)
    print(str(nb_train_entries) + ' train entries processed ' + str(nb_train_failed) + ' train entries failed')
    dev_entries, nb_dev_failed = generate_concepts_training_data(get_all_paths_for_alignment('dev', alignment))
    nb_dev_entries = len(dev_entries)
    print(str(nb_dev_entries) + ' dev entries processed ' + str(nb_dev_failed) + ' dev entries failed')
    return train_entries, nb_train_entries, dev_entries, nb_dev_entries


def read_test_data(alignment):
    test_entries, nb_test_failed = generate_concepts_training_data(get_all_paths_for_alignment('test', alignment))
    nb_test_entries = len(test_entries)
    print(str(nb_test_entries) + ' test entries processed ' + str(nb_test_failed) + ' test entries failed')
    test_entries, nb_test_entries