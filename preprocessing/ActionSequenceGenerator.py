import logging

import models.Actions as act
from action_sequence_generators.asg_exceptions import *


def generate_action_sequence(amr_graph, sentence, verbose=True):
    return generate_action_sequence_impl(amr_graph, sentence, 1, False, verbose)


def generate_action_sequence_impl(amr_graph, sentence, no_of_swaps, should_rotate, verbose=True):
    if verbose is False:
        logging.disable(logging.INFO)

    buffer = sentence.split(" ")
    # Top of stack is at position 0
    stack = []
    actions = []
    current_token = 0
    # last_action is:
    # 0 for "las action was not a swap"
    # 1 for "last action was a swap_1"
    # 2 for "last action was a swap_2" and so on
    last_action_swap = 0
    last_rotate = False
    while current_token < len(buffer) or len(stack) != 1:

        reduce_succeeded = False
        if len(stack) >= 2:

            # try to reduce the first two terms
            top = len(stack) - 1
            node_var_first = amr_graph.tokens_to_concepts_dict[stack[top]][0]
            node_var_second = amr_graph.tokens_to_concepts_dict[stack[top - 1]][0]
            if (node_var_first, node_var_second) in amr_graph.relations_dict.keys():
                # first is child of second => reduce right
                # check if first has any children left to process
                if len(amr_graph.relations_dict[(node_var_first, node_var_second)][1]) == 0:
                    reduce_right(actions, stack, top, node_var_first, node_var_second, amr_graph)
                    reduce_succeeded = True

            else:
                if (node_var_second, node_var_first) in amr_graph.relations_dict.keys():
                    # second is child of first => reduce left
                    # check if second has any children left to process
                    if len(amr_graph.relations_dict[(node_var_second, node_var_first)][1]) == 0:
                        reduce_left(actions, stack, top, node_var_first, node_var_second, amr_graph)
                        reduce_succeeded = True

        if reduce_succeeded:
            # reset the last_action_swap to 0 to indicate that the last action was not swap
            last_action_swap = 0
            last_rotate = False

        else:
            if last_action_swap == no_of_swaps:
                # try to rotate
                if should_rotate and (not last_rotate) and (len(stack) >= 3):

                    # print("rotate")
                    rotate(actions, stack, top)
                    last_rotate = True

                else:
                    if last_rotate:
                        logging.debug(
                            "Last rotate didn't solve the stack. Tokens left on the stack: %s. Actions %s.",
                            stack,
                            actions)
                        raise RotateException("Could not generate action sequence. Rotate not working")
                    else:
                        logging.debug(
                            "Last swap didn't solve the stack. Tokens left on the stack: %s. Actions %s.",
                            stack, actions)
                        raise SwapException("Could not generate action sequence. Swap not working")

            if current_token >= len(buffer):
                if (len(stack) >= (last_action_swap + 3)) and no_of_swaps != 0 and last_action_swap < no_of_swaps:
                    swap_n(actions, stack, top, last_action_swap + 1)
                    last_action_swap += 1
                    last_rotate = False
                else:
                    logging.debug("Tokens left on the stack: %s. Actions %s.", stack, actions)
                    raise TokenOnStackException("Could not generate action sequence. Tokens left on stack")

            else:
                # try to shift the current token
                if current_token in amr_graph.tokens_to_concepts_dict.keys():
                    stack.append(current_token)
                    tokens_to_concept = amr_graph.tokens_to_concepts_dict[current_token]
                    actions.append(act.AMRAction("SH", tokens_to_concept[1], tokens_to_concept[0]))

                else:
                    actions.append(act.AMRAction.build("DN"))
                    # reset the last_action_swap to 0 to indicate that the last action was not swap
                    last_action_swap = 0

                current_token += 1

    return actions


def reduce_right(actions, stack, top, node_var_first, node_var_second, amr_graph):
    actions.append(
        act.AMRAction.build_labeled("RR", amr_graph.relations_dict[(node_var_first, node_var_second)][0]))
    second_parent = amr_graph.parent_dict[node_var_second]
    # remove node_var_first from the second node's children list
    remove_child(amr_graph, node_var_second, second_parent, node_var_first)
    # remove first from stack
    stack.remove(stack[top])


def reduce_left(actions, stack, top, node_var_first, node_var_second, amr_graph):
    actions.append(act.AMRAction.build_labeled("RL", amr_graph.relations_dict[(node_var_second, node_var_first)][0]))
    first_parent = amr_graph.parent_dict[node_var_first]
    remove_child(amr_graph, node_var_first, first_parent, node_var_second)
    # remove second from stack
    stack.remove(stack[top - 1])


def swap_1(actions, stack, top):
    # we swap the second and third node
    aux = stack[top - 1]
    stack[top - 1] = stack[top - 2]
    stack[top - 2] = aux
    actions.append(act.AMRAction.build("SW"))


def swap_n(actions, stack, top, n):
    # we swap the second and (n+1)th node
    aux = stack[top - 1]
    stack[top - 1] = stack[top - n - 1]
    stack[top - n - 1] = aux
    action_name = "SW"
    if n > 1:
        suffix = "_" + str(n)
        action_name += suffix
    actions.append(act.AMRAction.build(action_name))


def rotate(actions, stack, top):
    # we swap the second and last node
    last_index = len(stack) - 1
    aux = stack[top - 1]
    stack[top - 1] = stack[last_index]
    stack[last_index] = aux
    action_name = "RO"
    actions.append(act.AMRAction.build(action_name))


def remove_child(amr_graph, node, parent, child):
    l = amr_graph.relations_dict[(node, parent)][1]
    try:
        if child in l:
            amr_graph.relations_dict[(node, parent)][1].remove(child)

    except Exception as e:
        raise Exception("Cannot remove child")


def check_relation_dict_consistency(amr_graph):
    is_consistent = True
    for (child, parent) in amr_graph.relations_dict.keys():
        if parent in amr_graph.parent_dict:
            parent_of_parent = amr_graph.parent_dict[parent]
            child_list = amr_graph.relations_dict[(parent, parent_of_parent)][1]
            if child not in child_list:
                is_consistent = False
    return is_consistent
