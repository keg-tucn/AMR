from typing import List

from data_extraction import input_file_parser
from data_extraction.dataset_reading_util import get_all_paths
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept

# Already using '' for root variable in custom_amr.parent_dict
ROOT_CONCEPT_VAR = ''
ROOT_CONCEPT_NAME = 'ROOT'


class ArcsTrainingEntry:
    def __init__(self,
                 identified_concepts: IdentifiedConcepts,
                 parent_vector: List[int],
                 logging_info:str,
                 amr_str:str):
        self.identified_concepts = identified_concepts
        self.parent_vector = parent_vector
        self.logging_info = logging_info
        # needed for smatch
        self.amr_str = amr_str


def generate_parent_vector(custom_amr: CustomizedAMR, identified_concepts: IdentifiedConcepts):
    """ go from (AMR) -> (identified concepts , parent vector) """
    index = 0
    variable_to_index = {}
    for concept in identified_concepts.ordered_concepts:
        # TODO: might have an issue for multiple nodes with same variable (same applies to custom amr)
        variable_to_index[concept.variable] = index
        index += 1
    parent_vector = [None] * len(identified_concepts.ordered_concepts)
    # for ROOT node
    parent_vector[0] = -1
    for variable in variable_to_index.keys():
        if variable != ROOT_CONCEPT_VAR:
            parent_vector[variable_to_index[variable]] = variable_to_index[custom_amr.parent_dict[variable]]
    return parent_vector


def add_root(identified_concepts: IdentifiedConcepts):
    """add a ROOT node at pos 0 in the ordered concepts list"""
    root_concept = Concept(ROOT_CONCEPT_VAR, ROOT_CONCEPT_NAME)
    identified_concepts.ordered_concepts.insert(0, root_concept)


def generate_dataset_entry(amr_id: str, amr_str: str, sentence: str):
    # TODO: pre processing steps
    amr = AMR.parse_string(amr_str)
    custom_amr: CustomizedAMR = CustomizedAMR()
    custom_amr.create_custom_AMR(amr)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_custom_amr(amr_id, custom_amr)
    # if I can't put in order all the concepts:
    if len(identified_concepts.ordered_concepts) != len(custom_amr.parent_dict.keys()):
        return None
    # empty AMR, don't care about it, should not be many:
    if len(identified_concepts.ordered_concepts) == 0:
        return None
    add_root(identified_concepts)
    parent_vector = generate_parent_vector(custom_amr, identified_concepts)
    logging_info = 'AMR with id ' + amr_id + '\n' + sentence + '\n' + amr_str +\
                   str(identified_concepts) + '\n' + str(parent_vector) + '\n\n'
    return ArcsTrainingEntry(identified_concepts, parent_vector, logging_info, amr_str)


# TODO: cache them to a file (to not always generate them)
def generate_arcs_training_data_per_file(file_path, max_sentence_len=50):
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    entries: List[ArcsTrainingEntry] = []
    no_entries_not_processed = 0
    for sentence, amr_str, amr_id in sentence_amr_triples:
        # filter out AMRs with more than max_sentence_len
        # TODO: extract tokens better
        sentence_len = len(sentence.split(" "))
        if sentence_len <= max_sentence_len:
            entry = generate_dataset_entry(amr_id, amr_str, sentence)
            if entry is not None:
                entries.append(entry)
            else:
                no_entries_not_processed += 1
    return entries, no_entries_not_processed


def generate_arcs_training_data(file_paths: List[str], max_sentence_len=50):
    all_entries = []
    no_all_entries_not_processed = 0
    for file_path in file_paths:
        entries, no_entries_not_processed = generate_arcs_training_data_per_file(file_path, max_sentence_len)
        all_entries = all_entries + entries
        no_all_entries_not_processed += no_entries_not_processed
    return all_entries, no_all_entries_not_processed


def read_train_test_data():
    train_entries, no_train_failed = generate_arcs_training_data(get_all_paths('training'))
    no_train_entries = len(train_entries)
    print(str(no_train_entries) + ' train entries processed ' + str(no_train_failed) + ' train entries failed')
    test_entries, no_test_failed = generate_arcs_training_data(get_all_paths('dev'))
    no_test_entries = len(test_entries)
    print(str(no_test_entries) + ' test entries processed ' + str(no_test_failed) + ' test entries failed')
    return (train_entries, test_entries)