import copy
import amr_util.Actions as act


class ASG:

    def __init__(self, no_of_swaps):
        self.no_of_swaps = no_of_swaps
        self.amr_graph = {}
        self.buffer = []
        self.buffer_indices = []
        self.stack = []
        self.actions = []
        self.removed_indices = []
        self.current_token = 0

    def initialize_fields(self, amr_graph, sentence):
        self.amr_graph = copy.deepcopy(amr_graph)
        self.buffer = sentence.split(" ")
        self.buffer_indices = range(len(self.buffer))
        self.stack = []
        self.actions = []
        self.removed_indices = []
        self.current_token = 0

    def is_buffer_empty(self):
        return len(self.buffer_indices) == 0

    def is_done(self):
        return (self.is_buffer_empty()) and (len(self.stack) == 1)

    def can_reduce_right(self):
        if len(self.stack) >= 2:
            top = len(self.stack) - 1
            return self.can_reduce(top - 1, top)
        return False

    def can_reduce_left(self):
        if len(self.stack) >= 2:
            top = len(self.stack) - 1
            return self.can_reduce(top, top - 1)
        return False

    def can_reduce(self, parent, child):
        node_parent = ASG._get_concept_for_index(self.amr_graph, self.stack[parent])
        node_child = ASG._get_concept_for_index(self.amr_graph, self.stack[child])
        if (node_child, node_parent) in self.amr_graph.relations_dict.keys():
            if len(self.amr_graph.relations_dict[(node_child, node_parent)][1]) == 0:
                return True
        return False

    def can_swap_n(self, n):
        if len(self.actions) > 0:
            last_added_action = self.actions[-1]
            action_name = "SW"
            if n > 1:
                suffix = "_" + str(n)
                action_name += suffix
            # if I'm trying to perform the same swap
            if last_added_action.action == action_name:
                return False
        return len(self.stack) >= n + 2 and self.no_of_swaps != 0

    def can_shift(self):
        return (not self.is_buffer_empty()) and self.current_token in self.amr_graph.tokens_to_concepts_dict.keys()

    def can_delete(self):
        return (not self.is_buffer_empty()) and (self.current_token not in self.amr_graph.tokens_to_concepts_dict.keys())

    def reduce_right(self):
        top = len(self.stack) - 1
        node_var_first = ASG._get_concept_for_index(self.amr_graph, self.stack[top])
        node_var_second = ASG._get_concept_for_index(self.amr_graph, self.stack[top - 1])
        self.actions.append(
            act.AMRAction.build_labeled("RR",
                                        self.amr_graph.relations_dict[(node_var_first, node_var_second)][0]))
        second_parent = self.amr_graph.parent_dict[node_var_second]
        # remove node_var_first from the second node's children list
        ASG._remove_child(self.amr_graph, node_var_second, second_parent, node_var_first)
        # save the index to be removed from stack in an ara (well, basically a stack)
        self.removed_indices.append(self.stack[top])
        # remove first from stack
        self.stack.remove(self.stack[top])

    def reduce_left(self):
        top = len(self.stack) - 1
        node_var_first = ASG._get_concept_for_index(self.amr_graph, self.stack[top])
        node_var_second = ASG._get_concept_for_index(self.amr_graph, self.stack[top - 1])
        self.actions.append(
            act.AMRAction.build_labeled("RL",
                                        self.amr_graph.relations_dict[(node_var_second, node_var_first)][0]))
        first_parent = self.amr_graph.parent_dict[node_var_first]
        ASG._remove_child(self.amr_graph, node_var_first, first_parent, node_var_second)
        # save the index to be removed from stack in an ara (well, basically a stack)
        self.removed_indices.append(self.stack[top - 1])
        # remove second from stack
        self.stack.remove(self.stack[top - 1])

    def swap_n(self, n):

        top = len(self.stack) - 1
        # we swap the second and (n+1)th node
        aux = self.stack[top - 1]
        self.stack[top - 1] = self.stack[top - n - 1]
        self.stack[top - n - 1] = aux
        action_name = "SW"
        if n > 1:
            suffix = "_" + str(n)
            action_name += suffix
        self.actions.append(act.AMRAction.build(action_name))

    def shift(self):
        self.stack.append(self.current_token)
        tokens_to_concept = self.amr_graph.tokens_to_concepts_dict[self.current_token]
        self.actions.append(act.AMRAction("SH", tokens_to_concept[1], tokens_to_concept[0]))
        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token = self.buffer_indices[0]

    #TODO: analyze this method (I have the vague impression there should be a tokens_to_concept_list)
    def brk(self):
        self.stack.append(self.current_token)
        tokens_to_concept = self.amr_graph.tokens_to_concepts_dict[self.current_token]
        self.actions.append(act.AMRAction("BRK", tokens_to_concept[1], tokens_to_concept[0]))
        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token = self.buffer_indices[0]

    def delete(self):
        self.actions.append(act.AMRAction.build("DN"))
        i = self.buffer_indices.pop(0)
        self.removed_indices.append(i)
        if len(self.buffer_indices) != 0:
            self.current_token = self.buffer_indices[0]

    def rotate(self):
        top = len(self.stack) - 1
        # we swap the second and last node
        last_index = len(self.stack) - 1
        aux = self.stack[top - 1]
        self.stack[top - 1] = self.stack[last_index]
        self.stack[last_index] = aux
        action_name = "RO"
        self.actions.append(act.AMRAction.build(action_name))

    @staticmethod
    def _get_concept_for_index(amr_graph, index):
        return amr_graph.tokens_to_concepts_dict[index][0]

    @staticmethod
    def _remove_child(amr_graph, node, parent, child):
        list_of_children = amr_graph.relations_dict[(node, parent)][1]
        try:
            if child in list_of_children:
                amr_graph.relations_dict[(node, parent)][1].remove(child)

        except Exception as e:
            raise Exception("Cannot remove child")
