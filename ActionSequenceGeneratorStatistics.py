import logging
import copy
from tqdm import tqdm

import AMRData
from preprocessing import SentenceAMRPairsExtractor, ActionSequenceGenerator
from preprocessing.ActionSequenceGenerator import SwapException
from preprocessing.ActionSequenceGenerator import TokenOnStackException
from AMRGraph import AMR
from amr_util import TrainingDataStats
from preprocessing import TokensReplacer

success = "success"
coreference = "coreference"
unaligned = "unaligned"
coreference_and_unaligned = "coreference_and_unaligned"
unknown = "unknown"
swap = "swap"
tokens_on_stack = "tokens_on_stack"


class ActionSeqGenStatistics:

    def __init__(self):
        self.histogram_overall = {success: 0, coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_exceptions = {swap: 0, tokens_on_stack: 0}
        self.histogram_swap = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_tokens_on_stack = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.preprocessing_failed = 0
        self.sequence_generation_failed = 0
        self.sentence_failed = 0
        self.amr_parse_fail = 0
        self.amr_pair_extraction_fail = 0

    def preprocessing(self, amr, sentence):
        new_amr = copy.deepcopy(amr)
        new_sentence = copy.deepcopy(sentence)

        # try to perform all preprocessing steps
        concepts_metadata = {}
        try:
            (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG1")
            (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG2")
        except Exception as e:
            # have_org_role_exceptions += 1
             raise e

        try:
            (new_amr, new_sentence, named_entities) = TokensReplacer.replace_named_entities(amr, sentence)
            for name_entity in named_entities:
                concepts_metadata[name_entity[0]] = name_entity[5]
        except Exception as e:
            # named_entity_exceptions += 1
            raise e

        try:
            (new_amr, new_sentence, date_entities) = TokensReplacer.replace_date_entities(new_amr, new_sentence)
            for date_entity in date_entities:
                concepts_metadata[date_entity[0]] = date_entity[5]
        except Exception as e:
            # date_entity_exceptions += 1
            raise e

        try:
            (new_amr, new_sentence, _) = TokensReplacer.replace_temporal_quantities(new_amr, new_sentence)
        except Exception as e:
             # temporal_quantity_exceptions += 1
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
            # quantity_exceptions += 1
            raise e

        return (new_amr, new_sentence)

    def on_swap_exception(self,is_coreference, is_unaligned):
        self.sequence_generation_failed += 1
        self.sentence_failed += 1
        self.histogram_exceptions[swap] += 1
        if is_coreference and not is_unaligned:
            self.histogram_swap[coreference] += 1
        if is_unaligned and (not is_coreference):
            self.histogram_swap[unaligned] += 1
        if is_coreference and is_unaligned:
            self.histogram_swap[coreference_and_unaligned] += 1
        if (not is_unaligned) and (not is_coreference):
            self.histogram_swap[unknown] += 1

    def on_tokens_on_stack_exception(self,is_coreference, is_unaligned):
        self.sequence_generation_failed += 1
        self.sentence_failed += 1
        self.histogram_exceptions[tokens_on_stack] += 1
        if is_coreference and (not is_unaligned):
            self.histogram_tokens_on_stack[coreference] += 1
        if is_unaligned and (not is_coreference):
            self.histogram_tokens_on_stack[unaligned] += 1
        if is_coreference and is_unaligned:
            self.histogram_tokens_on_stack[coreference_and_unaligned] += 1
        if (not is_unaligned) and (not is_coreference):
            self.histogram_tokens_on_stack[unknown] += 1

    def build_overall_histogram(self):
        # besides the success entry, all entries must be constructed
        self.histogram_overall[coreference] = self.histogram_swap[coreference] + self.histogram_tokens_on_stack[coreference]
        self.histogram_overall[unaligned] = self.histogram_swap[unaligned] + self.histogram_tokens_on_stack[unaligned]
        self.histogram_overall[coreference_and_unaligned] = self.histogram_swap[coreference_and_unaligned] + self.histogram_tokens_on_stack[coreference_and_unaligned]
        self.histogram_overall[unknown] = self.histogram_swap[unknown] + self.histogram_tokens_on_stack[unknown]

    def generate_statistics_for_a_sentence(self, i, amr, sentence):
        logging.debug("Started processing example %d", i)

        try:
            (new_amr,new_sentence) = self.preprocessing(amr,sentence)

            unaligned_nodes = {}
            try:
                TrainingDataStats.get_unaligned_nodes(new_amr, unaligned_nodes)
            except:
                print("Exception when getting unaligned nodes\n")

            custom_amr = AMRData.CustomizedAMR()
            try:
                custom_amr.create_custom_AMR(new_amr)
            except:
                print("Exception when creating custom AMR\n") # => twice

            try:
                coreferences_count = TrainingDataStats.get_coreferences_count(custom_amr)
            except:
                print("Exception when getting coreference count\n")

            is_coreference = coreferences_count != 0
            is_unaligned = len(unaligned_nodes) != 0

            try:
                action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_amr, new_sentence)
                self.histogram_overall[success] += 1
            except SwapException as e:
                # swap exception
                self.on_swap_exception(is_coreference, is_unaligned)
            except TokenOnStackException as e:
                # tokens on stack exception
                self.on_tokens_on_stack_exception(is_coreference,is_unaligned)

        except Exception as e:
            self.preprocessing_failed += 1
            self.sentence_failed += 1

    def generate_statistics(self, file_path):
        try:
            sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)
            #for i in tqdm(range(0, len(sentence_amr_triples))):
            for i in range(0, len(sentence_amr_triples)):
                (sentence, amr_str, amr_id) = sentence_amr_triples[i]
                try:
                    amr = AMR.parse_string(amr_str)
                    self.generate_statistics_for_a_sentence(i, amr, sentence)
                except Exception as e:
                    self.amr_parse_fail += 1
                    self.sentence_failed +=1
        except Exception as e:
            self.amr_pair_extraction_fail += 1
            self.sentence_failed +=1
            print(e)

    def print_statistics(self):

        print("Histogram overall:\nSuccess, Coreference, Unaligned, Coreference and unaligned, Unknown\n {0}, {1}, {2}, {3}, {4}\n".format(
            self.histogram_overall[success],
            self.histogram_overall[coreference],
            self.histogram_overall[unaligned],
            self.histogram_overall[coreference_and_unaligned],
            self.histogram_overall[unknown]))
        print("Histogram exceptions:\nSwap, Tokens on stack\n {0}, {1}\n".format(
            self.histogram_exceptions[swap],
            self.histogram_exceptions[tokens_on_stack]))
        print("Histogram swap:\nCoreference, Unaligned, Coreference and unaligned, Unknown\n {0}, {1}, {2}, {3}\n".format(
            self.histogram_swap[coreference],
            self.histogram_swap[unaligned],
            self.histogram_swap[coreference_and_unaligned],
            self.histogram_swap[unknown]))
        print("Histogram tokens on stack:\nCoreference, Unaligned, Coreference and unaligned, Unknown\n {0}, {1}, {2}, {3}\n".format(
            self.histogram_tokens_on_stack[coreference],
            self.histogram_tokens_on_stack[unaligned],
            self.histogram_tokens_on_stack[coreference_and_unaligned],
            self.histogram_tokens_on_stack[unknown]))
        #print("Total no of sentence fails: {0}".format(self.sentence_failed))
        #print("Unparsed amrs: {0}".format(self.amr_parse_fail))
        #print("Amr pairs extraction failed: {0}".format(self.amr_pair_extraction_fail))
        #print("Total no of preprocessings fails: {0}".format(self.preprocessing_failed))
        #print("Total no of action seq gen fails: {0}\n".format(self.sequence_generation_failed))
        print("Histogram sentence fails:\nParse amr str fail, Amr extraction fail, Preprocessing fail, Action sequence generation fail\n {0}, {1}, {2}, {3}\n".format(
            self.amr_parse_fail,
            self.amr_pair_extraction_fail,
            self.preprocessing_failed,
            self.sequence_generation_failed))

    @staticmethod
    def print_histogram( hist_name, column_names, values):
        print("Histogram " + hist_name)
        print(column_names)
        print(','.join([str(elem) for elem in values]))


#splits = ["training", "dev", "test"]
splits = ["training","dev","test"]
data_sets = {"training":["bolt","cctv","dfa","guidelines","mt09sdl","proxy","wb","xinhua"],
             "dev":["bolt","consensus","dfa","proxy","xinhua"],
             "test":["bolt","consensus","dfa","proxy","xinhua"]}


def add_lists(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


# go over all data (training, dev, tests) and construct histograms for eac dataset
for split in splits:
    histogram_overall_split = [0,0,0,0,0]
    histogram_exceptions_split = [0,0]
    histogram_swap_split = [0,0,0,0]
    histogram_tokens_on_stack_split = [0,0,0,0]
    histogram_sentence_fails_split = [0,0,0,0]
    for data_set in data_sets[split]:
        my_file_path = 'resources/alignments/split/'+split+"/"+"deft-p2-amr-r1-alignments-"+split+"-"+data_set+".txt"
        print("Generating statistics for "+my_file_path)
        acgStatistics = ActionSeqGenStatistics()
        acgStatistics.generate_statistics(my_file_path)
        acgStatistics.build_overall_histogram()
        acgStatistics.print_statistics()
        # make statistics per split
        histogram_overall_split = add_lists(histogram_overall_split,[acgStatistics.histogram_overall[success],
                                                                     acgStatistics.histogram_overall[coreference],
                                                                     acgStatistics.histogram_overall[unaligned],
                                                                     acgStatistics.histogram_overall[coreference_and_unaligned],
                                                                     acgStatistics.histogram_overall[unknown]])
        histogram_exceptions_split = add_lists(histogram_exceptions_split, [acgStatistics.histogram_exceptions[swap],
                                                                            acgStatistics.histogram_exceptions[tokens_on_stack]])
        histogram_swap_split = add_lists(histogram_swap_split, [acgStatistics.histogram_swap[k] for k in acgStatistics.histogram_swap])
        histogram_tokens_on_stack_split = add_lists(histogram_tokens_on_stack_split, [acgStatistics.histogram_swap[coreference],
                                                                                      acgStatistics.histogram_swap[unaligned],
                                                                                      acgStatistics.histogram_swap[coreference_and_unaligned],
                                                                                      acgStatistics.histogram_swap[unknown]])
        histogram_sentence_fails_split = add_lists(histogram_sentence_fails_split,
                                                   [acgStatistics.amr_parse_fail,
                                                    acgStatistics.amr_pair_extraction_fail,
                                                    acgStatistics.preprocessing_failed,
                                                    acgStatistics.sequence_generation_failed])
    # print statistics per split
    ActionSeqGenStatistics.print_histogram("overall "+split, "Success, Coreference, Unaligned, Coreference and unaligned, Unknown",histogram_overall_split)
    ActionSeqGenStatistics.print_histogram("exceptions "+split, "Swap, Tokens on stack",histogram_exceptions_split)
    ActionSeqGenStatistics.print_histogram("swap "+split, "Coreference, Unaligned, Coreference and unaligned, Unknown",histogram_swap_split)
    ActionSeqGenStatistics.print_histogram("tokens on stack "+split, "Coreference, Unaligned, Coreference and unaligned, Unknown",histogram_tokens_on_stack_split)