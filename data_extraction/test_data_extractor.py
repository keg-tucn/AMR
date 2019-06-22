import logging
from tqdm import tqdm

from data_extraction import input_file_parser
from definitions import AMR_ALIGNMENTS_SPLIT
from models.amr_graph import AMR
from preprocessing import NamedEntitiesReplacer


# Given a file with sentences and aligned amrs,
# it returns a list of preprocessed sentences
def generate_test_data(file_path, verbose=True):
    if verbose is False:
        logging.disable(logging.WARN)

    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    fail_sentences = []
    test_data = []
    named_entity_exceptions = 0

    for i in tqdm(range(0, len(sentence_amr_triples))):
        (sentence, amr_str, amr_id) = sentence_amr_triples[i]
        try:
            logging.warn("Started processing example %d", i)
            concepts_metadata = {}
            amr = AMR.parse_string(amr_str)

            try:
                (new_sentence, named_entities) = NamedEntitiesReplacer.process_sentence(sentence)
                for name_entity in named_entities:
                    concepts_metadata[name_entity[0]] = name_entity[1]
            except Exception as e:
                named_entity_exceptions += 1
                raise e

            test_data.append((new_sentence, concepts_metadata))
        except Exception as e:
            logging.warn(e)
            fail_sentences.append(sentence)
            logging.warn("Failed at: %d", i)
            logging.warn("%s\n", sentence)

    return test_data


print generate_test_data(AMR_ALIGNMENTS_SPLIT + "/test/deft-p2-amr-r1-alignments-test-bolt.txt", False)
