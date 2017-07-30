import logging

from tqdm import tqdm

import AMRData
from AMRGraph import AMR
from amr_util import TrainingDataStats
from preprocessing import SentenceAMRPairsExtractor, ActionSequenceGenerator
from preprocessing import TokensReplacer
from preprocessing.DependencyExtractor import extract_dependencies


# Given a file with sentences and aligned amrs,
# it returns an array of (sentence, action_sequence, amr_string))
def generate_training_data(file_path, verbose=True, withStats=False, withDependencies=False):
    if verbose is False:
        logging.disable(logging.WARN)

    sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)
    fail_sentences = []
    unaligned_nodes = {}
    unaligned_nodes_after = {}
    training_data = []
    coreferences_count = 0
    have_org_role_exceptions = 0
    named_entity_exceptions = 0
    date_entity_exceptions = 0
    temporal_quantity_exceptions = 0
    quantity_exceptions = 0
    processed_sentence_ids = []

    for i in tqdm(range(0, len(sentence_amr_triples))):
        try:
            logging.warn("Started processing example %d", i)
            concepts_metadata = {}
            (sentence, amr_str, amr_id) = sentence_amr_triples[i]
            amr = AMR.parse_string(amr_str)
            TrainingDataStats.get_unaligned_nodes(amr, unaligned_nodes)
            try:
                (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG1")
                (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG2")
            except Exception as e:
                have_org_role_exceptions += 1
                raise e

            try:
                (new_amr, new_sentence, named_entities) = TokensReplacer.replace_named_entities(amr, sentence)
                for name_entity in named_entities:
                    concepts_metadata[name_entity[0]] = name_entity[5]
            except Exception as e:
                named_entity_exceptions += 1
                raise e

            try:
                (new_amr, new_sentence, date_entities) = TokensReplacer.replace_date_entities(new_amr, new_sentence)
                for date_entity in date_entities:
                    concepts_metadata[date_entity[0]] = date_entity[5]
            except Exception as e:
                date_entity_exceptions += 1
                raise e

            try:
                (new_amr, new_sentence, _) = TokensReplacer.replace_temporal_quantities(new_amr, new_sentence)
            except Exception as e:
                temporal_quantity_exceptions += 1
                raise e
            try:
                (new_amr, new_sentence, _) = TokensReplacer.replace_quantities_default(new_amr, new_sentence,
                                                                                       ['monetary-quantity',
                                                                                        'mass-quantity',
                                                                                        'energy-quantity',
                                                                                        'distance-quantity',
                                                                                        'volume-quantity',
                                                                                        'power-quantity'
                                                                                        ])
            except Exception as e:
                quantity_exceptions += 1
                raise e

            TrainingDataStats.get_unaligned_nodes(new_amr, unaligned_nodes_after)
            custom_amr = AMRData.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)
            coreferences_count += TrainingDataStats.get_coreferences_count(custom_amr)
            action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_amr, new_sentence)
            if withDependencies is False:
                training_data.append((new_sentence, action_sequence, amr_str, concepts_metadata, amr_id))
            else:
                try:
                    deps = extract_dependencies(new_sentence)
                except Exception as e:
                    logging.warn("Dependency parsing failed at sentence %s with exception %s.", new_sentence, str(e))
                    deps = {}
                #### For the keras flow, we also attach named_entities, date_entities, to instances
                training_data.append((new_sentence, action_sequence, amr_str, deps, named_entities, date_entities))
            processed_sentence_ids.append(amr_id)
        except Exception as e:
            logging.warn(e)
            fail_sentences.append(sentence)
            logging.warn("Failed at: %d", i)
            logging.warn("%s\n", sentence)

    logging.critical("Failed: %d out of %d", len(fail_sentences), len(sentence_amr_triples))
    # logging.critical("|%s|%d|%d|%d", file_path, len(fail_sentences), len(sentence_amr_pairs), len(sentence_amr_pairs) - len(fail_sentences))
    if withStats is True:
        return training_data, unaligned_nodes, unaligned_nodes_after, coreferences_count, \
               named_entity_exceptions, date_entity_exceptions, temporal_quantity_exceptions, quantity_exceptions, have_org_role_exceptions
    else:
        return training_data

if __name__ == "__main__":
    from os import listdir, path, makedirs
    data = []
    mypath = 'resources/alignments/split/' + "dev"
    print(mypath)
    for f in listdir(mypath):
        if not "dump" in f and "deft" in f:
            mypath_f = mypath + "/" + f
            print(mypath_f)
            new_data = generate_training_data(mypath_f, False)
            data += new_data
            with open(mypath_f) as input_file:
                lines = input_file.readlines()
            audit_f = mypath + "/audit/" + f + ".audit"
            if not path.exists(path.dirname(audit_f)):
                makedirs(path.dirname(audit_f))
            processed_ids = []
            for (_,_,_,_,processed_id) in new_data:
                processed_ids.append(processed_id)
            with open(audit_f, "wb") as audit:
                for processed_id in processed_ids:
                    audit.write("%s" % processed_id)
            amr_inputs = []
            for processed_id in processed_ids:
                i = lines.index(processed_id)
                amr_input = ""
                while i < len(lines) and len(lines[i]) > 1:
                    amr_input += lines[i]
                    i += 1
                amr_input += "\n"
                amr_inputs.append(amr_input)
            with open(audit_f + "_content", "wb") as content:
                for amr_input in amr_inputs:
                    content.write("%s" %amr_input)

    print len(data)
#generate_training_data("/home/andreea/LDC2016E25_DEFT_Phase_2_AMR_Annotation_R2/data/alignments/unsplit/deft-p2-amr-r2-alignments-bolt.txt", False)
