from AMRGraph import AMR
from tqdm import tqdm
import AMRData
import ActionSequenceGenerator
import TokensReplacer
import SentenceAMRPairsExtractor
import TrainingDataStats
import logging


# Given a file with sentences and aligned amrs,
# it returns an array of (sentence, action_sequence, amr_string))
def generate_training_data(file_path, verbose=True, withStats=False):
    if verbose is False:
        logging.disable(logging.WARN)

    sentence_amr_pairs = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)
    fail_sentences = []
    unaligned_nodes = {}
    training_data = []
    coreferences_count = 0

    for i in tqdm(range(0, len(sentence_amr_pairs))):
        try:
            logging.warn("Started processing example %d", i)
            (sentence, amr_str) = sentence_amr_pairs[i]
            amr = AMR.parse_string(amr_str)
            TrainingDataStats.get_unaligned_nodes(amr, unaligned_nodes)
            (new_amr, new_sentence, _) = TokensReplacer.replace_named_entities(amr, sentence)
            (new_amr, new_sentence, _) = TokensReplacer.replace_date_entities(new_amr, new_sentence)
            custom_amr = AMRData.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)
            coreferences_count += TrainingDataStats.get_coreferences_count(custom_amr)
            action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_amr, new_sentence)
            training_data.append((new_sentence, action_sequence, amr_str))
        except Exception as e:
            logging.warn(e)
            fail_sentences.append(sentence)
            logging.warn("Failed at: %d", i)
            logging.warn("%s\n", sentence)

    logging.critical("Failed: %d out of %d", len(fail_sentences), len(sentence_amr_pairs))
    if withStats is False:
        return training_data
    else:
        return training_data, unaligned_nodes, coreferences_count

# generate_training_data(
#    "/Users/silvianac/personalprojects/date/LDC2015E86_DEFT_Phase_2_AMR_Annotation_R1/data/alignments/unsplit/deft-p2-amr-r1-alignments-xinhua.txt", False)
