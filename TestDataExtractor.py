import logging

from tqdm import tqdm

from models.AMRGraph import AMR
from preprocessing import NamedEntitiesReplacer
from preprocessing import SentenceAMRPairsExtractor


# Given a file with sentences and aligned amrs,
# it returns a list of preprocessed sentences
def generate_test_data(file_path, verbose=True):
    if verbose is False:
        logging.disable(logging.WARN)

    sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)
    fail_sentences = []
    test_data = []
    named_entity_exceptions = 0

    for i in tqdm(range(0, len(sentence_amr_triples))):
        try:
            logging.warn("Started processing example %d", i)
            concepts_metadata = {}
            (sentence, amr_str, amr_id) = sentence_amr_triples[i]
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

print generate_test_data("/home/iv/Documents/AMR/resources/alignments/split/test/deft-p2-amr-r1-alignments-test-bolt.txt", False)
