import functools
from collections import OrderedDict
from typing import List
import matplotlib.pyplot as plt
from data_extraction import input_file_parser
from data_extraction.dataset_reading_util import get_all_paths
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept

from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_vectors

# Already using '' for root variable in custom_amr.parent_dict
ROOT_CONCEPT_VAR = ''
ROOT_CONCEPT_NAME = 'ROOT'


class ArcsTrainingEntry:
    def __init__(self,
                 identified_concepts: IdentifiedConcepts,
                 parent_vectors: List[List[int]],
                 logging_info: str,
                 amr_str: str):
        self.identified_concepts = identified_concepts
        self.parent_vectors = parent_vectors
        self.logging_info = logging_info
        # needed for smatch
        self.amr_str = amr_str


def add_false_root(identified_concepts: IdentifiedConcepts):
    """add a ROOT node at pos 0 in the ordered concepts list"""
    root_concept = Concept(ROOT_CONCEPT_VAR, ROOT_CONCEPT_NAME)
    identified_concepts.ordered_concepts.insert(0, root_concept)


def generate_dataset_entry(amr_id: str, amr_str: str, sentence: str, unaligned_tolerance: float, max_no_parent_vectors: int):
    # TODO: pre processing steps
    amr = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr(amr_id, amr, unaligned_tolerance)
    if identified_concepts.ordered_concepts is None:
        return None
    # # if I don't have some parents in my ordered concepts
    # if len(identified_concepts.ordered_concepts) != len(custom_amr.parent_dict.keys()):
    #     return None
    # empty AMR, don't care about it, should not be many:
    if len(identified_concepts.ordered_concepts) == 0:
        return None
    add_false_root(identified_concepts)
    parent_vectors = generate_parent_vectors(amr, identified_concepts, max_no_parent_vectors)
    if parent_vectors is None:
        return None
    logging_info = 'AMR with id ' + amr_id + '\n' + sentence + '\n' + amr_str + \
                   str(identified_concepts) + '\n' + str(parent_vectors) + '\n\n'
    return ArcsTrainingEntry(identified_concepts, parent_vectors, logging_info, amr_str)


# TODO: cache them to a file (to not always generate them)
def generate_arcs_training_data_per_file(file_path, unaligned_tolerance, max_sentence_len, max_no_parent_vectors):
    no_of_parent_vectors_histogram = {}
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    entries: List[ArcsTrainingEntry] = []
    no_entries_not_processed = 0
    for sentence, amr_str, amr_id in sentence_amr_triples:
        # filter out AMRs with more than max_sentence_len
        # TODO: extract tokens better
        sentence_len = len(sentence.split(" "))
        if sentence_len <= max_sentence_len:
            entry: ArcsTrainingEntry = generate_dataset_entry(amr_id,
                                                              amr_str,
                                                              sentence,
                                                              unaligned_tolerance,
                                                              max_no_parent_vectors)
            if entry is not None:
                entries.append(entry)
                no_of_parent_vectors = len(entry.parent_vectors)
                if no_of_parent_vectors not in no_of_parent_vectors_histogram.keys():
                    no_of_parent_vectors_histogram[no_of_parent_vectors] = 0
                no_of_parent_vectors_histogram[no_of_parent_vectors] += 1
            else:
                no_entries_not_processed += 1
    return entries, no_entries_not_processed, no_of_parent_vectors_histogram


def generate_arcs_training_data(file_paths: List[str],
                                unaligned_tolerance: float,
                                max_sentence_len: int,
                                max_no_parent_vectors: int):
    all_entries = []
    no_all_entries_not_processed = 0
    no_of_parent_vectors_histogram = {}
    for file_path in file_paths:
        entries, \
        no_entries_not_processed, \
        no_of_parent_vectors_histogram_dataset = generate_arcs_training_data_per_file(
            file_path, unaligned_tolerance,
            max_sentence_len,
            max_no_parent_vectors)
        all_entries = all_entries + entries
        no_all_entries_not_processed += no_entries_not_processed
        for key, value in no_of_parent_vectors_histogram_dataset.items():
            if key in no_of_parent_vectors_histogram.keys():
                no_of_parent_vectors_histogram[key] += no_of_parent_vectors_histogram_dataset[key]
            else:
                no_of_parent_vectors_histogram[key] = no_of_parent_vectors_histogram_dataset[key]
    return all_entries, no_all_entries_not_processed, no_of_parent_vectors_histogram


class ArcsTraingAndTestData:

    def __init__(self,
                 train_entries, test_entries,
                 no_train_amrs, no_test_amrs):
        self.train_entries = train_entries
        self.test_entries = test_entries
        self.no_train_amrs = no_train_amrs
        self.no_test_amrs = no_test_amrs


@functools.lru_cache(maxsize=5)
def read_train_test_data(unaligned_tolerance: float, max_sentence_len: int, max_no_parent_vectors: int):
    train_entries, no_train_failed, no_pv_hist_train = generate_arcs_training_data(get_all_paths('training'),
                                                                                   unaligned_tolerance,
                                                                                   max_sentence_len,
                                                                                   max_no_parent_vectors)
    no_train_entries = len(train_entries)
    print(str(no_train_entries) + ' train entries (AMRs) processed ' + str(no_train_failed) + ' train entries failed')
    print('train parent vectors histogram')
    print(OrderedDict(sorted(no_pv_hist_train.items())))
    plt.bar(list(no_pv_hist_train.keys()), no_pv_hist_train.values(), color='g')
    plt.show()
    test_entries, no_test_failed, no_pv_hist_test = generate_arcs_training_data(get_all_paths('dev'),
                                                                                unaligned_tolerance,
                                                                                max_sentence_len,
                                                                                max_no_parent_vectors)
    no_test_entries = len(test_entries)
    print(str(no_test_entries) + ' test entries (AMRs) processed ' + str(no_test_failed) + ' test entries failed')
    print('test parent vectors histogram')
    print(OrderedDict(sorted(no_pv_hist_test.items())))
    plt.bar(list(no_pv_hist_test.keys()), no_pv_hist_test.values(), color='g')
    plt.show()
    return ArcsTraingAndTestData(train_entries, test_entries, no_train_entries, no_test_entries)
