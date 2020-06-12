from typing import List

from data_extraction import input_file_parser
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept
from pre_post_processing.standford_pre_post_processing import train_pre_processing


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
    amr, new_sentence = train_pre_processing(amr, sentence)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr(amr_id, amr)
    if identified_concepts.ordered_concepts is None:
        return None
    if len(identified_concepts.ordered_concepts) == 0:
        return None

    logging_info = "AMR with id " + amr_id + "\n" + new_sentence + "\n" + \
                   "ORDERED concepts: " + str(identified_concepts.ordered_concepts) + "\n"

    return ConceptsTrainingEntry(identified_concepts, new_sentence, logging_info, amr_str)


# TODO: cache them to a file (to not always generate them)
def generate_concepts_training_data_per_file(file_path, max_sentence_len=50):
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


def generate_concepts_training_data(file_paths: List[str], max_sentence_len=50):
    all_entries = []

    nb_all_entries_not_processed = 0
    for file_path in file_paths:
    # for i in range (1):
        entries, nb_entries_not_processed = generate_concepts_training_data_per_file(file_path, max_sentence_len)
        all_entries = all_entries + entries
        nb_all_entries_not_processed += nb_entries_not_processed

    return all_entries, nb_all_entries_not_processed