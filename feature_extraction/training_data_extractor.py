import logging
from os import listdir, makedirs
from tqdm import tqdm
from collections import namedtuple

from definitions import PROJECT_ROOT_DIR
from models import amr_data
from models.amr_graph import AMR, ParserError
from models.train_data import TrainData
from amr_util import TrainingDataStats
from preprocessing import SentenceAMRPairsExtractor
from preprocessing import TokensReplacer
from preprocessing.DependencyExtractor import DependencyExtractor
from preprocessing.action_sequence_generators.simple_asg__informed_swap import SimpleInformedSwapASG
from Baseline import baseline

TrainingDataExtraction = namedtuple("TrainingDataExtraction", "data stats")


def generate_training_data(file_path, compute_dependencies=True):
    """
        Return an array of TrainData instances given a file with sentences and aligned AMRs
    """
    sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)

    training_data = []
    processed_sentence_ids = []

    fail_sentences = []
    unaligned_nodes = {}
    unaligned_nodes_after = {}
    coreference_count = 0

    have_org_role_exceptions = 0
    named_entity_exceptions = 0
    date_entity_exceptions = 0
    temporal_quantity_exceptions = 0
    quantity_exceptions = 0

    # set to True for co-reference handling (graph transformed to trees)
    coref_handling = False

    for i in tqdm(range(len(sentence_amr_triples))):

        concepts_metadata = {}
        (sentence, amr_str, amr_id) = sentence_amr_triples[i]

        try:
            logging.debug("Started processing example %d", i)

            amr = AMR.parse_string(amr_str)

            # co-reference handling
            if coref_handling:
                try:
                    new_amr_str = baseline(amr_str)
                    amr = AMR.parse_string(new_amr_str)
                except ParserError:
                    amr = AMR.parse_string(amr_str)

            TrainingDataStats.get_unaligned_nodes(amr, unaligned_nodes)

            #
            # replace have-org-role tokens
            #
            try:
                (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG1")
                (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG2")
            except Exception as e:
                have_org_role_exceptions += 1
                raise e

            #
            # replace named entities tokens
            #
            try:
                (new_amr, new_sentence, named_entities) = TokensReplacer.replace_named_entities(amr, sentence)
                for name_entity in named_entities:
                    concepts_metadata[name_entity[0]] = name_entity[5]
            except Exception as e:
                named_entity_exceptions += 1
                raise e

            #
            # replace date entities tokens
            #
            try:
                (new_amr, new_sentence, date_entities) = TokensReplacer.replace_date_entities(new_amr, new_sentence)
                for date_entity in date_entities:
                    concepts_metadata[date_entity[0]] = date_entity[5]
            except Exception as e:
                date_entity_exceptions += 1
                raise e

            #
            # replace temporal entities tokens
            #
            try:
                (new_amr, new_sentence, _) = TokensReplacer.replace_temporal_quantities(new_amr, new_sentence)
            except Exception as e:
                temporal_quantity_exceptions += 1
                raise e

            #
            # replace quantity entities tokens
            #
            try:
                (new_amr, new_sentence, _) = TokensReplacer.replace_quantities_default(new_amr, new_sentence, [
                    'monetary-quantity', 'mass-quantity', 'energy-quantity', 'distance-quantity', 'volume-quantity',
                    'power-quantity'])
            except Exception as e:
                quantity_exceptions += 1
                raise e

            TrainingDataStats.get_unaligned_nodes(new_amr, unaligned_nodes_after)

            #
            # create CustomAMR data structure
            #
            custom_amr = amr_data.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)

            coreference_count += TrainingDataStats.get_coreference_count(custom_amr)

            # TODO: put here the new version of the action seq generator
            # action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_amr, new_sentence)
            asg_implementation = SimpleInformedSwapASG(1, False)
            action_sequence = asg_implementation.generate_action_sequence(custom_amr, new_sentence)

            #
            # extract dependencies
            #
            if compute_dependencies is False:
                named_entities = []
                date_entities = []
                dependencies = {}
            else:
                try:
                    dependencies = DependencyExtractor.extract_dependencies(new_sentence)
                except Exception as e:
                    logging.warn("Dependency parsing failed at sentence %s with exception %s.", new_sentence, str(e))
                    dependencies = {}

            # For the keras flow, also attach named_entities, date_entities, to instances
            training_data_instance = TrainData(new_sentence, action_sequence, amr_str, dependencies, named_entities,
                                               date_entities, concepts_metadata, amr_id)
            training_data.append(training_data_instance)
            processed_sentence_ids.append(amr_id)

        except Exception as e:
            fail_sentences.append(sentence)
            logging.debug("Exception is: '%s'. Failed at: %d with sentence %s.", e, i, sentence)

    logging.info("Failed: %d out of %d", len(fail_sentences), len(sentence_amr_triples))

    training_data_stats = TrainingDataStats.TrainingDataStatistics(unaligned_nodes, unaligned_nodes_after,
                                                                   coreference_count, named_entity_exceptions,
                                                                   date_entity_exceptions,
                                                                   temporal_quantity_exceptions,
                                                                   quantity_exceptions,
                                                                   have_org_role_exceptions)
    return TrainingDataExtraction(training_data, training_data_stats)


def extract_amr_ids_from_corpus_as_audit_trail():
    data = []
    path = PROJECT_ROOT_DIR + "/resources/alignments/split/" + "dev"
    original_path = 'resources/amrs/split/' + "dev"

    for f in listdir(path):
        if "dump" not in f and "deft" in f:
            path_f = path + "/" + f
            original_path_f = original_path + "/" + f.replace("alignments", "amrs")
            print(path_f)
            new_data = generate_training_data(path_f).data
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
    generated_data = generate_training_data(
        PROJECT_ROOT_DIR + "/resources/alignments/split/dev/deft-p2-amr-r1-alignments-dev-bolt.txt")
    assert isinstance(generated_data, TrainingDataExtraction)
    assert isinstance(generated_data.data, list)
    assert isinstance(generated_data.stats, TrainingDataStats.TrainingDataStatistics)
    data = generated_data.data
    assert len(data) == 23
    for elem in data:
        assert isinstance(elem, TrainData)
        assert isinstance(elem.sentence, basestring)
        assert isinstance(elem.action_sequence, list)
        assert isinstance(elem.original_amr, basestring)
        assert isinstance(elem.dependencies, dict)
        assert isinstance(elem.named_entities, list)
        assert isinstance(elem.date_entities, list)
        assert isinstance(elem.concepts_metadata, dict)
        assert isinstance(elem.amr_id, basestring)
