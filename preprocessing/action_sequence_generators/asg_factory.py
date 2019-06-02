from preprocessing.action_sequence_generators.simple_asg import SimpleASG
from preprocessing.action_sequence_generators.simple_asg__informed_swap import SimpleInformedSwapASG
from preprocessing.action_sequence_generators.simple_asg_nodes_on_stack import SimpleNodesOnStackASG
from preprocessing.action_sequence_generators.simple_informed_break_nodes_on_stack import \
    SimpleInformedWithBreakNodesOnStackASG
from preprocessing.action_sequence_generators.backtracking_asg import BacktrackingASGFixedReduce, \
    BacktrackingASGInformedSwap
from models.parameters import ASGParameters

SIMPLE = "simple"
SIMPLE_INFORMED_SWAP = "simple_informed_swap"
NOS_SIMPLE = "nodes_on_stack_simple"
NOS_BREAK_INFORMED_SWAP = "nodes_on_stack_break_informed_swap"
BACKTRACK_INFORMED_SWAP = "backtrack_informed_swap"
BACKTRACK_FIXED_REDUCE = "backtrack_fixed_reduce"


def get_asg_implementation(asg_parameters=ASGParameters()):
    asg_implementation = None

    asg_alg = asg_parameters.asg_alg
    no_swaps = asg_parameters.no_swaps
    swap_distance = asg_parameters.swap_distance
    rotate = asg_parameters.rotate

    if asg_alg == SIMPLE:
        asg_implementation = SimpleASG(no_swaps, rotate)
    elif asg_alg == SIMPLE_INFORMED_SWAP:
        asg_implementation = SimpleInformedSwapASG(swap_distance, rotate)
    elif asg_alg == NOS_SIMPLE:
        asg_implementation = SimpleNodesOnStackASG(swap_distance, rotate)
    elif asg_alg == NOS_BREAK_INFORMED_SWAP:
        asg_implementation = SimpleInformedWithBreakNodesOnStackASG(swap_distance, rotate)
    elif asg_alg == BACKTRACK_INFORMED_SWAP:
        asg_implementation = BacktrackingASGInformedSwap(swap_distance, 256)
    elif asg_alg == BACKTRACK_FIXED_REDUCE:
        asg_implementation = BacktrackingASGFixedReduce(no_swaps, 256)

    return asg_implementation
