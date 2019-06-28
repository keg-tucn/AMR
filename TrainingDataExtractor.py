import logging

from tqdm import tqdm

import spacy
import neuralcoref

import AMRData
from AMRGraph import AMR
from amr_util import TrainingDataStats
from preprocessing import SentenceAMRPairsExtractor, ActionSequenceGenerator
from preprocessing import TokensReplacer
from preprocessing.DependencyExtractor import extract_dependencies
from collections import namedtuple
from preprocessing.action_sequence_generators.simple_asg__informed_swap import SimpleInformedSwapASG
from preprocessing.action_sequence_generators.simple_asg import SimpleASG

from coreference_detection.coreference_detection import duplicate_concept_data_on_amr_for_coreferenced_nodes, \
    has_AMR_coreferenced_nodes, remove_wiki_and_name_concepts_from_amr

TrainingDataExtraction = namedtuple("TrainingDataExtraction", "data stats")
TrainingData = namedtuple("TrainingData", "sentence, action_sequence, amr_original, dependencies, named_entities, "
                                          "date_entities, concepts_metadata, amr_id")


#for coref handling
from Baseline import baseline

# Given a file with sentences and aligned amrs,
# it returns an array of (TrainingData)
def generate_training_data(file_path, compute_dependencies=False, nlp=None):
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

    exceptions_by_preprocessing_for_found_coref_sentences = 0
    coref_handling = True if nlp is not None else False

    sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)

    if coref_handling:
        # statistics for the co-reference data sets
        coref_statistics = {}
        coref_statistics['nr_of_sentences_with_corefs_which_did_not_raise_exceptions'] = 0
        coref_statistics['nr_of_sentences_with_corefs_which_raised_exceptions'] = 0
        coref_statistics['nr_of_sentences_with_corefs_which_entered_ASG'] = 0
        coref_statistics['nr_of_sentences_in_which_we_found_corefs'] = 0

        sentence_has_corefs = False

    # logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


    for i in tqdm(range(0, len(sentence_amr_triples))):

        try:
            logging.debug("Started processing example %d", i)
            concepts_metadata = {}
            (sentence, amr_str, amr_id) = sentence_amr_triples[i]

            sentence_has_corefs = False

            # coreference handling
            if coref_handling:
                doc = nlp(unicode(sentence, 'utf-8'))

                if doc._.has_coref:                 # if has_AMR_coreferenced_nodes(amr_str):
                    sentence_has_corefs = True
                    coref_statistics['nr_of_sentences_in_which_we_found_corefs'] += 1

                    amr_str = duplicate_concept_data_on_amr_for_coreferenced_nodes(amr_str)

            # amr_str = remove_wiki_and_name_concepts_from_amr(amr_str) # possible fix to named entity bug but it does not improve the system -> needs more analysis

            amr = AMR.parse_string(amr_str)

            try:
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

            except Exception as e:
                if coref_handling and sentence_has_corefs:
                    exceptions_by_preprocessing_for_found_coref_sentences += 1
                raise e

            TrainingDataStats.get_unaligned_nodes(new_amr, unaligned_nodes_after)
            custom_amr = AMRData.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)


            coreferences_count += TrainingDataStats.get_coreferences_count(custom_amr)
            #TODO: put here the new version of the action seq generator

            coref_statistics['nr_of_sentences_with_corefs_which_entered_ASG'] += 1

            asg_implementation = SimpleInformedSwapASG(1, False)
            #asg_implementation = SimpleASG(1,False)
            action_sequence = asg_implementation.generate_action_sequence(custom_amr, new_sentence)
            #action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_amr, new_sentence)

            if compute_dependencies is False:
                # training_data.append(TrainingData(new_sentence, action_sequence, amr_str, concepts_metadata, amr_id))
                named_entities = []
                date_entities = []
                deps = {}
            else:
                try:
                    deps = extract_dependencies(new_sentence)
                except Exception as e:
                    logging.warn("Dependency parsing failed at sentence %s with exception %s.", new_sentence, str(e))
                    deps = {}
                    #### For the keras flow, we also attach named_entities, date_entities, to instances
            training_data.append(
                TrainingData(new_sentence, action_sequence, amr_str, deps, named_entities, date_entities,
                             concepts_metadata, amr_id))
            processed_sentence_ids.append(amr_id)

            if coref_handling and sentence_has_corefs:
                coref_statistics['nr_of_sentences_with_corefs_which_did_not_raise_exceptions'] += 1

        except Exception as e:
            fail_sentences.append(sentence)
            logging.debug("Exception is: '%s'. Failed at: %d with sentence %s.", e, i, sentence)

            if coref_handling and sentence_has_corefs:
                coref_statistics['nr_of_sentences_with_corefs_which_raised_exceptions'] += 1

    logging.info("Failed: %d out of %d", len(fail_sentences), len(sentence_amr_triples))

    if coref_handling:
        print("Dataset %s", file_path)
        print("Sentences with corefs that didn't raise exceptions: ", coref_statistics['nr_of_sentences_with_corefs_which_did_not_raise_exceptions'])
        print("Sentences with corefs that raised exceptions: ", coref_statistics['nr_of_sentences_with_corefs_which_raised_exceptions'])
        print("Sentences in which we found corefs: ",  coref_statistics['nr_of_sentences_in_which_we_found_corefs'], " out of ", len(sentence_amr_triples))
        print("Preprocessing exceptions when we had sentences on which corefs were found: ", exceptions_by_preprocessing_for_found_coref_sentences)
        print("Nr of amrs which entered ASG: ", coref_statistics['nr_of_sentences_with_corefs_which_entered_ASG'])
        print("Exception: have_org_role {}\nnamed_entity {}\ndate_entity {}\ntemporal_quantity {}\nquantity {}".format(
            have_org_role_exceptions,
            named_entity_exceptions,
            date_entity_exceptions,
            temporal_quantity_exceptions,
            quantity_exceptions))
    # logging.critical("|%s|%d|%d|%d", file_path, len(fail_sentences), len(sentence_amr_pairs), len(sentence_amr_pairs) - len(fail_sentences))

    return TrainingDataExtraction(training_data,
                                  TrainingDataStats.TrainingDataStatistics(unaligned_nodes, unaligned_nodes_after,
                                                                           coreferences_count,
                                                                           named_entity_exceptions,
                                                                           date_entity_exceptions,
                                                                           temporal_quantity_exceptions,
                                                                           quantity_exceptions,
                                                                           have_org_role_exceptions)
                                  )


def extract_amr_ids_from_corpus_as_audit_trail():
    from os import listdir, path, makedirs
    data = []
    mypath = 'resources/alignments/split/' + "dev"
    original_path = 'resources/amrs/split/' + "dev"
    print(mypath)
    for f in listdir(mypath):
        if not "dump" in f and "deft" in f:
            mypath_f = mypath + "/" + f
            original_path_f = original_path + "/" + f.replace("alignments", "amrs")
            print(mypath_f)
            new_data = generate_training_data(mypath_f).data
            data += new_data
            with open(original_path_f) as input_file:
                lines = input_file.readlines()
            audit_f = original_path + "/audit/" + f + ".audit"
            if not path.exists(path.dirname(audit_f)):
                makedirs(path.dirname(audit_f))
            processed_ids = []
            for element in new_data:
                processed_ids.append(element.amr_id)
            with open(audit_f, "wb") as audit:
                for processed_id in processed_ids:
                    audit.write("%s\n" % processed_id)
            amr_inputs = []
            for processed_id in processed_ids:
                line = filter(lambda k: processed_id in k, lines)
                i = lines.index(line[0])
                amr_input = ""
                while i < len(lines) and len(lines[i]) > 1:
                    amr_input += lines[i]
                    i += 1
                amr_input += "\n"
                amr_inputs.append(amr_input)
            with open(audit_f + "_content", "wb") as content:
                for amr_input in amr_inputs:
                    content.write("%s" % amr_input)
    print len(data)


if __name__ == "__main__":
    # extract_amr_ids_from_corpus_as_audit_trail()
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)
    generated_data = generate_training_data("resources/alignments/split/dev/deft-p2-amr-r2-alignments-dev-bolt.txt")
    assert isinstance(generated_data, TrainingDataExtraction)
    assert isinstance(generated_data.data, list)
    assert isinstance(generated_data.stats, TrainingDataStats.TrainingDataStatistics)
    data = generated_data.data
    assert len(data) == 20
    for elem in data:
        assert len(elem) == 8
        assert isinstance(elem, TrainingData)
        assert isinstance(elem[0], basestring)
        assert isinstance(elem.sentence, basestring)
        assert isinstance(elem[1], list)
        assert isinstance(elem.action_sequence, list)
        assert isinstance(elem[2], basestring)
        assert isinstance(elem.amr_original, basestring)
        assert isinstance(elem[3], dict)
        assert isinstance(elem.dependencies, dict)
        assert isinstance(elem[4], list)
        assert isinstance(elem.named_entities, list)
        assert isinstance(elem[5], list)
        assert isinstance(elem.date_entities, list)
        assert isinstance(elem[6], dict)
        assert isinstance(elem.concepts_metadata, dict)
        assert isinstance(elem[7], basestring)
        assert isinstance(elem.amr_id, basestring)
