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
from ActionSequenceGeneratorStatisticsPlotter import plot_histogram

success = "success"
coreference = "coreference"
unaligned = "unaligned"
coreference_and_unaligned = "coreference and unaligned"
unknown = "unknown"
swap = "swap"
tokens_on_stack = "tokens on_stack"
preprocessing_failed = "preprocessing failed"
sequence_generation_failed="sequence generation failed"
amr_parse_fail="amr parse fail"
amr_pair_extraction_fail="amr pair extraction fail"

class ActionSeqGenStatistics:

    def __init__(self):
        self.histogram_overall = {success: 0, coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_exceptions = {swap: 0, tokens_on_stack: 0}
        self.histogram_swap = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_tokens_on_stack = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_sentence_fails = {preprocessing_failed: 0, sequence_generation_failed: 0, amr_parse_fail: 0, amr_pair_extraction_fail: 0}
        self.sentence_failed = 0

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
        self.histogram_sentence_fails[sequence_generation_failed] += 1
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
        self.histogram_sentence_fails[sequence_generation_failed] += 1
        #self.sequence_generation_failed += 1
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
            self.histogram_sentence_fails[preprocessing_failed] += 1
            #self.preprocessing_failed += 1
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
                    self.histogram_sentence_fails[amr_parse_fail] += 1
                    self.sentence_failed +=1
        except Exception as e:
            self.histogram_sentence_fails[amr_pair_extraction_fail] += 1
            self.sentence_failed +=1
            print(e)

    def print_statistics(self):

        ActionSeqGenStatistics.print_histograms(["overall","exceptions","swap","tokens on stack","sentence fails"],
                                                [self.histogram_overall,
                                                self.histogram_exceptions,
                                                self.histogram_swap,
                                                self.histogram_tokens_on_stack,
                                                self.histogram_sentence_fails])

    def plot_statistics(self,alg_version,split,data_set):
        histogram_data =[
                        self.histogram_overall,
                        self.histogram_exceptions,
                        self.histogram_swap,
                        self.histogram_tokens_on_stack,
                        self.histogram_sentence_fails]
        ActionSeqGenStatistics.plot_statistics_static(histogram_data,alg_version,split,data_set)

    @staticmethod
    def plot_statistics_static(histogram_data,alg_version,split,data_set):
        histogram_names = ["overall","exceptions","swap","tokens on stack","sentence fails"]
        plot_histogram(histogram_data,histogram_names,alg_version,split,data_set)

    @staticmethod
    def print_histograms(histogram_names, histogram_list):

        h_list_len = len(histogram_list)
        for i in range(0,h_list_len):
            ActionSeqGenStatistics.print_histogram(histogram_names[i], histogram_list[i].keys(), histogram_list[i].values())

    @staticmethod
    def print_histogram(hist_name, column_names, values):
        print("Histogram " + hist_name)
        print(', '.join([str(elem) for elem in column_names]))
        print(', '.join([str(elem) for elem in values]))


splits = ["training", "dev", "test"]
data_sets = {"training":["bolt","cctv","dfa","guidelines","mt09sdl","proxy","wb","xinhua"],
             "dev":["bolt","consensus","dfa","proxy","xinhua"],
             "test":["bolt","consensus","dfa","proxy","xinhua"]}

alg_version = "swap_1"


def add_lists(list1, list2):
    return [a + b for a, b in zip(list1, list2)]

def add_dicts(d1, d2):
    # return a dictionary with keys the union of the sets of keys of d1 and d2
    # and values the addition of the values in d1 and d2
    return {k: d1.get(k, 0) + d2.get(k,0) for k in set(d1.keys()) | set(d2.keys())}


# go over all data (training, dev, tests) and construct histograms for eac dataset
for split in splits:
    # create dictionaries per split
    histogram_overall_split = {}
    histogram_exceptions_split = {}
    histogram_swap_split = {}
    histogram_tokens_on_stack_split = {}
    histogram_sentence_fails_split = {}
    for data_set in data_sets[split]:
        my_file_path = 'resources/alignments/split/'+split+"/"+"deft-p2-amr-r1-alignments-"+split+"-"+data_set+".txt"
        print("Generating statistics for "+my_file_path)
        acgStatistics = ActionSeqGenStatistics()
        acgStatistics.generate_statistics(my_file_path)
        acgStatistics.build_overall_histogram()
        # print statistics per database in split
        acgStatistics.print_statistics()
        # plot statistics per database in split
        acgStatistics.plot_statistics(alg_version,split,data_set)
        # update per split statistics
        histogram_overall_split = add_dicts(histogram_overall_split,acgStatistics.histogram_overall)
        histogram_exceptions_split = add_dicts(histogram_exceptions_split,acgStatistics.histogram_exceptions)
        histogram_swap_split = add_dicts(histogram_swap_split,acgStatistics.histogram_swap)
        histogram_tokens_on_stack_split = add_dicts(histogram_tokens_on_stack_split,acgStatistics.histogram_tokens_on_stack)
        histogram_sentence_fails_split = add_dicts(histogram_sentence_fails_split,acgStatistics.histogram_sentence_fails)
    # print statistics per split
    print("\nStatistics for "+split)
    histogram_data = [
                    histogram_overall_split,
                    histogram_exceptions_split,
                    histogram_swap_split,
                    histogram_tokens_on_stack_split,
                    histogram_sentence_fails_split]
    ActionSeqGenStatistics.print_histograms(["overall","exceptions","swap","tokens on stack","sentence fails"],
                                            histogram_data)
    # plot statistics per split
    ActionSeqGenStatistics.plot_statistics_static(histogram_data,alg_version,split,"all-datasets")