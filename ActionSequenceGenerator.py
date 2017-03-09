
def generate_action_sequence(amr_graph, sentence):
    buffer = sentence.split(" ")
    # Top of stack is at position 0
    stack = []
    actions = []
    current_token = 0
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
                    actions.append("RR_" + amr_graph.relations_dict[(node_var_first, node_var_second)][0])
                    second_parent = amr_graph.parent_dict[node_var_second]
                    # remove node_var_first from the second node's children list
                    remove_child(amr_graph, node_var_second, second_parent, node_var_first)
                    # remove first from stack
                    stack.remove(stack[top])
                    reduce_succeeded = True

            else:
                if (node_var_second, node_var_first) in amr_graph.relations_dict.keys():
                    # second is child of first => reduce left
                    # check if second has any children left to process
                    if len(amr_graph.relations_dict[(node_var_second, node_var_first)][1]) == 0:
                        actions.append("RL_" + amr_graph.relations_dict[(node_var_second, node_var_first)][0])
                        first_parent = amr_graph.parent_dict[node_var_first]
                        remove_child(amr_graph, node_var_first, first_parent, node_var_second)
                        # remove second from stack
                        stack.remove(stack[top - 1])
                        reduce_succeeded = True

        if not reduce_succeeded:
            if current_token >= len(buffer):
                print "[ERROR]"
                print "\nTokens left on the stack\n"
                print stack
                return actions
            # try to shift the current token
            if current_token in amr_graph.tokens_to_concepts_dict.keys():
                stack.append(current_token)
                actions.append("SH_" + amr_graph.tokens_to_concepts_dict[current_token][1])
            else:
                actions.append("DN")
            current_token += 1
    return actions


def remove_child(amr_graph, node, parent, child):
    amr_graph.relations_dict[(node, parent)][1].remove(child)

