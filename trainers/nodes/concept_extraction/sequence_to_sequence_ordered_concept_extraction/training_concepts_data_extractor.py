from typing import List

from data_extraction import input_file_parser
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept

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
    logging_info = 'AMR with id ' + amr_id + '\n' + sentence + '\n' +\
                   'Ordered concepts: ' + str(identified_concepts.ordered_concepts) + '\n\n'
    return ConceptsTrainingEntry(identified_concepts, sentence, logging_info, amr_str)


# TODO: cache them to a file (to not always generate them)
def generate_concepts_training_data_per_file(file_path, max_sentence_len=50):
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    entries: List[ConceptsTrainingEntry] = []
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


def generate_concepts_training_data(file_paths: List[str], max_sentence_len=50):
    all_entries = []
    no_all_entries_not_processed = 0
    for file_path in file_paths:
    # for i in range (1):
        entries, no_entries_not_processed = generate_concepts_training_data_per_file(file_path, max_sentence_len)
        all_entries = all_entries + entries
        no_all_entries_not_processed += no_entries_not_processed

    return all_entries, no_all_entries_not_processed


# STATISTICS FOR VERBS VS REST IN CONCEPTS
def generate_entry_verbs_others(entry):
    verbs = []
    others = []

    for concept in entry.identified_concepts.ordered_concepts:
        splitted = concept.name.split('-')
        if splitted[len(splitted) - 1].isdigit():
            verbs.append(concept.name)
        else:
            others.append(concept.name)

    return verbs, others


def generate_statistics_verbs_other_concepts(file_path):
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    entries: List[ConceptsTrainingEntry] = []
    verbs = []
    others = []

    for sentence, amr_str, amr_id in sentence_amr_triples:
        # filter out AMRs with more than max_sentence_len
        # TODO: extract tokens better
        sentence_len = len(sentence.split(" "))
        entry = generate_dataset_entry(amr_id, amr_str, sentence)
        if entry is not None:
            entry_verbs, entry_others = generate_entry_verbs_others(entry)
            verbs = verbs + entry_verbs
            others = others + entry_others
    return verbs, others


def statistics_verbs_other_concepts(file_paths: List[str], max_sentence_len=10000):
    all_verbs = []
    all_distinct_verbs = []
    all_others = []
    all_distinct_others = []
    all_train_verbs = []
    all_distinct_train_verbs = []
    all_train_others = []
    all_distinct_train_others = []
    all_test_verbs = []
    all_distinct_test_verbs = []
    all_test_others = []
    all_distinct_test_others = []

    for file_path in file_paths:
        verbs, others = generate_statistics_verbs_other_concepts(file_path)
        all_verbs = all_verbs + verbs
        all_others = all_others + others
        if 'dev' in file_path:
            all_test_verbs = all_test_verbs + verbs
            all_test_others = all_test_others + others
        else:
            all_train_verbs = all_train_verbs + verbs
            all_train_others = all_train_others + others

    all_distinct_verbs = list(set(all_verbs))
    all_distinct_others = list(set(all_others))
    all_distinct_train_verbs = list(set(all_train_verbs))
    all_distinct_train_others = list(set(all_train_others))
    all_distinct_test_verbs = list(set(all_test_verbs))
    all_distinct_test_others = list(set(all_test_others))

    logs = open("verb-others_statistic_logs", "w")

    logs.write("Total number of verbs: " + str(len(all_verbs)) + '\n')
    logs.write("Total number of other concepts: " + str(len(all_others)) + '\n')
    logs.write("Total number of distinct verbs: " + str(len(all_distinct_verbs)) + '\n')
    logs.write("Total number of distinct other concepts: " + str(len(all_distinct_others)) + '\n')
    logs.write("Total number of verbs in train: " + str(len(all_train_verbs)) + '\n')
    logs.write("Total number of other concepts in train: " + str(len(all_train_others)) + '\n')
    logs.write("Total number of distinct verbs in train: " + str(len(all_distinct_train_verbs)) + '\n')
    logs.write("Total number of distinct other concepts in train: " + str(len(all_distinct_train_others)) + '\n')
    logs.write("Total number of verbs in dev: " + str(len(all_test_verbs)) + '\n')
    logs.write("Total number of other concepts in dev: " + str(len(all_test_others)) + '\n')
    logs.write("Total number of distinct verbs in dev: " + str(len(all_distinct_test_verbs)) + '\n')
    logs.write("Total number of distinct other concepts in dev: " + str(len(all_distinct_test_others)) + '\n')

    logs.close()

    return 0