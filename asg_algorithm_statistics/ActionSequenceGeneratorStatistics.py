import logging
import copy

from asg_algorithm_statistics.ActionSequenceGeneratorStatisticsPlotter import plot_2_line_graph, plot_histogram
from definitions import PROJECT_ROOT_DIR
from models import amr_data
from data_extraction import input_file_parser
from models.parameters import ParserParameters
from preprocessing.action_sequence_generators.backtracking_asg import BacktrackingASGFixedReduce
from preprocessing.action_sequence_generators.backtracking_asg import BacktrackingASGInformedSwap
from preprocessing.action_sequence_generators.simple_asg__informed_swap import SimpleInformedSwapASG
from preprocessing.action_sequence_generators.simple_asg_nodes_on_stack import SimpleNodesOnStackASG
from preprocessing.action_sequence_generators.simple_informed_break_nodes_on_stack import \
    SimpleInformedWithBreakNodesOnStackASG
from preprocessing.action_sequence_generators.simple_asg import SimpleASG
from preprocessing.ActionSequenceGenerator import SwapException
from preprocessing.ActionSequenceGenerator import TokenOnStackException
from preprocessing.ActionSequenceGenerator import RotateException
from models.amr_graph import AMR
from amr_util import TrainingDataStats, tokenizer_util
from preprocessing import TokensReplacer
from postprocessing import action_sequence_reconstruction as asr
from smatch import smatch_util
from smatch import smatch_amr

from Baseline import baseline
from Baseline import reentrancy_restoring

success = "success"
coreference = "coreference"
unaligned = "unaligned"
coreference_and_unaligned = "coreference and unaligned"
unknown = "unknown"
swap = "swap"
rotate = "rotate"
tokens_on_stack = "tokens on_stack"
preprocessing_failed = "preprocessing failed"
sequence_generation_failed = "sequence generation failed"
unknown_sequence_generation_error = "unknown sequence generation error"
amr_parse_fail = "amr parse fail"
amr_pair_extraction_fail = "amr pair extraction fail"
smatch_not_1 = "smatch"


class WrongActionSequenceException(Exception):
    pass


MAX_SENTENCE_LEN = 255


class ActionSeqGenStatistics:

    def __init__(self, asg_implementation, min_sentence_len, max_sentence_len, coreference_handling):
        self.histogram_overall = {success: 0, coreference: 0, unaligned: 0, coreference_and_unaligned: 0,
                                  smatch_not_1: 0, unknown: 0}
        self.histogram_exceptions = {swap: 0, tokens_on_stack: 0, rotate: 0, unknown_sequence_generation_error: 0}
        self.histogram_swap = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_tokens_on_stack = {coreference: 0, unaligned: 0, coreference_and_unaligned: 0, unknown: 0}
        self.histogram_sentence_fails = {preprocessing_failed: 0, sequence_generation_failed: 0, amr_parse_fail: 0,
                                         amr_pair_extraction_fail: 0}
        # vector where the sentence lengths are the indices, and the no of sentences with that length is the data
        # initialize described vector with 0
        self.sentence_lengths_all = [0] * MAX_SENTENCE_LEN
        # vector where the sentence lengths are the indices,
        # and the no of successful sentences with that length is the data
        self.sentence_lengths_successes = [0] * MAX_SENTENCE_LEN
        self.sentence_failed = 0
        self.asg_implementation = asg_implementation
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.named_entities_metadata = []
        self.date_entities_metadata = []
        self.coreference_handling = coreference_handling
        self.coref_handling_errors = 0

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

            self.named_entities_metadata = [(n[3], n[2]) for n in named_entities]

            for name_entity in named_entities:
                concepts_metadata[name_entity[0]] = name_entity[5]

        except Exception as e:
            # named_entity_exceptions += 1
            raise e

        try:
            (new_amr, new_sentence, date_entities) = TokensReplacer.replace_date_entities(new_amr, new_sentence)

            self.date_entities_metadata = [(d[3], d[2], d[1]) for d in date_entities]

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
        return new_amr, new_sentence

    def on_swap_exception(self, is_coreference, is_unaligned):
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

    def on_tokens_on_stack_exception(self, is_coreference, is_unaligned):
        self.histogram_sentence_fails[sequence_generation_failed] += 1
        # self.sequence_generation_failed += 1
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
        self.histogram_overall[coreference] = self.histogram_swap[coreference] + self.histogram_tokens_on_stack[
            coreference]
        self.histogram_overall[unaligned] = self.histogram_swap[unaligned] + self.histogram_tokens_on_stack[unaligned]
        self.histogram_overall[coreference_and_unaligned] = self.histogram_swap[coreference_and_unaligned] + \
                                                            self.histogram_tokens_on_stack[coreference_and_unaligned]
        self.histogram_overall[unknown] = self.histogram_swap[unknown] + self.histogram_tokens_on_stack[unknown]

    def generate_statistics_for_a_sentence(self, i, amr_id, amr, sentence, amr_str):
        logging.debug("Started processing example %d", i)

        action_sequence = []

        try:

            if self.coreference_handling:

                try:
                    new_amr_str = baseline(amr_str)
                    amr = AMR.parse_string(new_amr_str)
                except:
                    self.coref_handling_errors += 1

            (new_amr, new_sentence) = self.preprocessing(amr, sentence)

            unaligned_nodes = {}
            try:
                TrainingDataStats.get_unaligned_nodes(new_amr, unaligned_nodes)
            except:
                print("Exception when getting unaligned nodes\n")

            custom_amr = amr_data.CustomizedAMR()

            try:

                custom_amr.create_custom_AMR(new_amr)

            except Exception as ce:
                print("Exception when creating custom AMR\n")  # => twice

            try:
                coreferences_count = TrainingDataStats.get_coreference_count(custom_amr)
            except:
                print("Exception when getting coreference count\n")

            is_coreference = coreferences_count != 0
            is_unaligned = len(unaligned_nodes) != 0

            try:

                action_sequence = self.asg_implementation.generate_action_sequence(custom_amr, new_sentence)

                parser_parameters = ParserParameters(max_len=input_max_sentence_len, with_enhanced_dep_info=False,
                                                     with_target_semantic_labels=False, with_reattach=True,
                                                     with_gold_concept_labels=True, with_gold_relation_labels=True)
                tokens = tokenizer_util.text_to_sequence(sentence)
                generated_amr = asr.reconstruct_all_ne(tokens, action_sequence,
                                                           self.named_entities_metadata,
                                                           self.date_entities_metadata, parser_parameters).amr_print()
                generated_amr_str = generated_amr.amr_print()
                if self.coreference_handling:
                    generated_amr_str = reentrancy_restoring(generated_amr_str)

                smatch_results = smatch_util.SmatchAccumulator()
                # does this not mean the nodes have the allignment info with them>
                original_amr = smatch_amr.AMR.parse_AMR_line(amr_str)
                generated_amr = smatch_amr.AMR.parse_AMR_line(generated_amr_str)
                smatch_f_score = smatch_results.compute_and_add(generated_amr, original_amr)

                if smatch_f_score != 1:
                    self.histogram_overall[smatch_not_1] += 1

                else:
                    self.histogram_overall[success] += 1
                    sentence_len = len(sentence.split(" "))
                    self.sentence_lengths_successes[sentence_len] += 1

            except SwapException as e:
                # swap exception
                self.on_swap_exception(is_coreference, is_unaligned)

            except TokenOnStackException as e:
                # tokens on stack exception
                self.on_tokens_on_stack_exception(is_coreference, is_unaligned)

            except RotateException as e:
                # rotate exception
                self.histogram_sentence_fails[sequence_generation_failed] += 1
                self.sentence_failed += 1
                self.histogram_exceptions[rotate] += 1
            except Exception as e:
                self.histogram_exceptions[unknown_sequence_generation_error] += 1
                self.histogram_sentence_fails[sequence_generation_failed] += 1

        except Exception as e:
            self.histogram_sentence_fails[preprocessing_failed] += 1
            self.sentence_failed += 1

    def generate_statistics(self, file_path):
        try:
            sentence_amr_triples = input_file_parser.extract_data_records(file_path)
            # for i in tqdm(range(0, len(sentence_amr_triples))):
            for i in range(0, len(sentence_amr_triples)):
                (sentence, amr_str, amr_id) = sentence_amr_triples[i]
                sentence_len = len(sentence.split(" "))
                self.sentence_lengths_all[sentence_len] = self.sentence_lengths_all[sentence_len] + 1
                if self.min_sentence_len <= sentence_len < self.max_sentence_len:
                    try:
                        amr = AMR.parse_string(amr_str)
                        self.generate_statistics_for_a_sentence(i, amr_id, amr, sentence, amr_str)
                    except Exception as e:
                        self.histogram_sentence_fails[amr_parse_fail] += 1
                        self.sentence_failed += 1
        except Exception as e:
            # these exceptions maybe shouldn't be counted, I mean at least not added to sentence fails :)
            self.histogram_sentence_fails[amr_pair_extraction_fail] += 1
            self.sentence_failed += 1
            print(e)

    def plot_statistics(self, alg_version, split, data_set):
        histogram_data = [
            self.histogram_overall,
            self.histogram_exceptions,
            self.histogram_swap,
            self.histogram_tokens_on_stack,
            self.histogram_sentence_fails]
        ActionSeqGenStatistics.plot_statistics_static(histogram_data, alg_version, split, data_set)

    @staticmethod
    def plot_statistics_static(histogram_data, alg_version, split, data_set):
        histogram_names = ["overall", "exceptions", "swap", "tokens on stack", "sentence fails"]
        plot_histogram(histogram_data, histogram_names, alg_version, split, data_set)

    @staticmethod
    def plot_sentence_len_graph(split, sentence_lengths_all, sentence_lengths_success):
        graph_relative_path = alg_version + "/" + split + "/" + "sentence_lengths.png"
        plot_2_line_graph(sentence_lengths_all, sentence_lengths_success, graph_relative_path)


def add_lists(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def add_dicts(d1, d2):
    # return a dictionary with keys the union of the sets of keys of d1 and d2
    # and values the addition of the values in d1 and d2
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1.keys()) | set(d2.keys())}


splits = ["training", "dev", "test"]
data_sets = {"training": ["bolt", "cctv", "dfa", "dfb", "guidelines", "mt09sdl", "proxy", "wb", "xinhua"],
             "dev": ["bolt", "consensus", "dfa", "proxy", "xinhua"],
             "test": ["bolt", "consensus", "dfa", "proxy", "xinhua"]}

input_min_sentence_len = 1
input_max_sentence_len = 50
input_no_of_swaps = 1
input_should_rotate = False
alg_version = "simple"

# go over all data (training, dev, tests) and construct histograms for eac datasetl
for split in splits:
    # create dictionaries per split
    histogram_overall_split = {}
    histogram_exceptions_split = {}
    histogram_swap_split = {}
    histogram_tokens_on_stack_split = {}
    histogram_sentence_fails_split = {}
    sentence_lengths_all = [0] * MAX_SENTENCE_LEN
    sentence_lengths_success = [0] * MAX_SENTENCE_LEN
    for data_set in data_sets[split]:
        my_file_path = PROJECT_ROOT_DIR + '/resources/alignments/split/' + split + "/" + "deft-p2-amr-r2-alignments-" + split + "-" + data_set + ".txt"
        print(("Generating statistics for " + my_file_path))

        asg_implementation = SimpleASG(input_no_of_swaps, input_should_rotate)

        max_depth = 4 * input_max_sentence_len

        if alg_version == "backtrack":
            asg_implementation = BacktrackingASGFixedReduce(input_no_of_swaps, max_depth)
        elif alg_version == "informed_swap":
            # doesn't really need rotate, does it?
            asg_implementation = SimpleInformedSwapASG(input_no_of_swaps, should_rotate=input_should_rotate)
        elif alg_version == "backtrack_informed_swap":
            asg_implementation = BacktrackingASGInformedSwap(input_no_of_swaps, max_depth)
        elif alg_version == "simple_nodes_on_stack":
            asg_implementation = SimpleNodesOnStackASG(input_no_of_swaps, input_should_rotate)
        elif alg_version == "simple_nodes_on_stack":
            asg_implementation = SimpleNodesOnStackASG(input_no_of_swaps, input_should_rotate)
        elif alg_version == "simple_informed_break_nodes_on_stack":
            asg_implementation = SimpleInformedWithBreakNodesOnStackASG(input_no_of_swaps, input_should_rotate)

        acgStatistics = ActionSeqGenStatistics(asg_implementation, input_min_sentence_len, input_max_sentence_len,
                                               coreference_handling=False)
        acgStatistics.generate_statistics(my_file_path)
        acgStatistics.build_overall_histogram()
        # plot statistics per database in split
        acgStatistics.plot_statistics(alg_version, split, data_set)
        # update per split statistics
        histogram_overall_split = add_dicts(histogram_overall_split, acgStatistics.histogram_overall)
        histogram_exceptions_split = add_dicts(histogram_exceptions_split, acgStatistics.histogram_exceptions)
        histogram_swap_split = add_dicts(histogram_swap_split, acgStatistics.histogram_swap)
        histogram_tokens_on_stack_split = add_dicts(histogram_tokens_on_stack_split,
                                                    acgStatistics.histogram_tokens_on_stack)
        histogram_sentence_fails_split = add_dicts(histogram_sentence_fails_split,
                                                   acgStatistics.histogram_sentence_fails)
        sentence_lengths_all = add_lists(sentence_lengths_all, acgStatistics.sentence_lengths_all)
        sentence_lengths_success = add_lists(sentence_lengths_success, acgStatistics.sentence_lengths_successes)

    # print statistics per split
    print(("\nStatistics for " + split))
    histogram_data = [
        histogram_overall_split,
        histogram_exceptions_split,
        histogram_swap_split,
        histogram_tokens_on_stack_split,
        histogram_sentence_fails_split]
    # plot statistics per split
    ActionSeqGenStatistics.plot_statistics_static(histogram_data, alg_version, split, "all-datasets")
    ActionSeqGenStatistics.plot_sentence_len_graph(split, sentence_lengths_all, sentence_lengths_success)

print("DONE")
