# test what instances are processed by different asg versions
from preprocessing import SentenceAMRPairsExtractor

from amr_util.demo_util import get_smatch
from postprocessing import ActionSequenceReconstruction as asr
from models.amr_graph import AMR
from models import amr_data
from preprocessing.action_sequence_generators.simple_asg import SimpleASG
from preprocessing.action_sequence_generators.simple_asg__informed_swap import SimpleInformedSwapASG
from preprocessing.action_sequence_generators.simple_informed_break_nodes_on_stack import \
    SimpleInformedWithBreakNodesOnStackASG
from preprocessing.action_sequence_generators.backtracking_asg import BacktrackingASGFixedReduce

bolt_file_path = '/home/andrei/dynet-base/AMR_lic/resources/alignments/split/training/deft-p2-amr-r2-alignments-training-bolt.txt'

datasets = ["bolt", "cctv", "dfa", "dfb", "guidelines", "mt09sdl", "proxy", "wb", "xinhua"]

for i in range(len(datasets)):

    file_path = '/home/andrei/dynet-base/AMR_lic/resources/alignments/split/training/deft-p2-amr-r2-alignments-training-' + \
                datasets[i] + '.txt'

    sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)
    for i in range(0, len(sentence_amr_triples)):
        (sentence, amr_str, amr_id) = sentence_amr_triples[i]
        sentence_len = len(sentence.split(" "))
        if sentence_len < 20:
            try:
                amr = AMR.parse_string(amr_str)
                custom_amr = amr_data.CustomizedAMR()
                custom_amr.create_custom_AMR(amr)

                try:
                    # initial asg
                    asg_implementation = SimpleASG(1, False)
                    action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                    # print(amr_id)
                except:
                    try:
                        # informed swap (1 swap)
                        asg_implementation = SimpleInformedSwapASG(1, should_rotate=False)
                        action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                        # print(action_sequence)
                        # print(amr_id)
                    except:
                        try:
                            # informed swap (2 swap)
                            asg_implementation = SimpleInformedSwapASG(2, should_rotate=False)
                            action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                            # print(action_sequence)
                            # print(amr_id)
                        except:
                            try:
                                # informed swap (3 swaps)
                                asg_implementation = SimpleInformedSwapASG(3, should_rotate=False)
                                action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                                # print(action_sequence)
                                # print(amr_id)
                            except:
                                try:
                                    # informed swap (rotate)
                                    asg_implementation = SimpleInformedSwapASG(3, should_rotate=True)
                                    action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                                    # print(action_sequence)
                                    # print(amr_id)
                                except:
                                    try:
                                        # informed swap with break-token
                                        asg_implementation = SimpleInformedWithBreakNodesOnStackASG(1,
                                                                                                    should_rotate=False)
                                        action_sequence = asg_implementation.generate_action_sequence(custom_amr,
                                                                                                      sentence)
                                        # print(amr_id)
                                        # print(action_sequence)
                                    except:
                                        pass
                        # backtracking
                        if sentence_len < 10:
                            try:
                                max_depth = 4 * sentence_len
                                asg_implementation = BacktrackingASGFixedReduce(1, max_depth)
                                action_sequence = asg_implementation.generate_action_sequence(custom_amr, sentence)
                                generated_amr_str = asr.reconstruct_all(action_sequence)
                                smatch_f_score = get_smatch(generated_amr_str, amr_str)
                                if smatch_f_score == 1:
                                    print(amr_id)
                                    print(amr_str)
                                    print(action_sequence)
                            except:
                                pass
            except:
                pass
