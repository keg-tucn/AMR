from asg_nodes_on_stack import NodesOnStackASG
import logging
import models.Actions as act

""" Added break action to simple informed alg
"""
from preprocessing.ActionSequenceGenerator import TokenOnStackException


class SimpleInformedWithBreakNodesOnStackASG(NodesOnStackASG):

    def __init__(self, no_of_swaps, should_rotate):
        NodesOnStackASG.__init__(self,no_of_swaps)
        self.should_rotate = should_rotate

    def generate_action_sequence(self, amr_graph, sentence):

        NodesOnStackASG.initialize_fields(self, amr_graph, sentence)

        swapped = False

        last_action_swap = 0
        while not self.is_done():
            reduce_succeeded = False

            if self.can_reduce_right():
                self.reduce_right()
                reduce_succeeded = True

            else:
                if self.can_reduce_left():
                    self.reduce_left()
                    reduce_succeeded = True

            if reduce_succeeded:
                # reset the last_action_swap to 0 to indicate that the last action was not swap
                last_action_swap = 0
                swapped = False
            else:

                for i in range(1,self.no_of_swaps+1):
                    if self.can_swap_n(i):
                        self.swap_n(i)
                        swapped = True
                        break
                        last_action_swap = i
                        # I can still shift or delete
                if self.should_rotate and self.can_rotate():
                    self.rotate()
                    swapped = True
                if not swapped:
                    if not self.is_buffer_empty():
                        # try to "break" the token
                        if self.can_break(2):
                            self.brk()
                        # try to shift the current token
                        elif self.can_shift():
                            self.shift()
                        else:
                            self.delete()
                            # if not self.can_delete():
                            #     print("not good")
                    else:
                        logging.debug("Tokens left on the stack: %s. Actions %s.", self.stack, self.actions)
                        raise TokenOnStackException(
                            "Could not generate action sequence. Tokens left on stack")

        return self.actions

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

    def can_swap_n(self, n):

        if not NodesOnStackASG.can_swap_n(self, n):
            return False

            # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, top - n - 1) or self.can_reduce(top - n - 1, top):
            return True
        else:
            return False

    def can_break(self, no_of_nodes):
        if self.is_buffer_empty():
            return False
        if self.current_token in self.amr_graph.tokens_to_concept_list_dict.keys():
            aligned_nodes = self.amr_graph.tokens_to_concept_list_dict[self.current_token]
            return len(aligned_nodes) == no_of_nodes
        return False

    def can_shift(self):
        return self.can_break(1)

    def shift(self):
        tokens_to_concept = self.amr_graph.tokens_to_concept_list_dict[self.current_token][0]
        node = tokens_to_concept[0]
        concept = tokens_to_concept[1]
        node_token_pair = (node, self.current_token)
        self.stack.append(node_token_pair)
        self.actions.append(act.AMRAction("SH", concept, node))
        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token = self.buffer_indices[0]

    def brk(self):
        tokens_to_concept_list = self.amr_graph.tokens_to_concept_list_dict[self.current_token]
        node1 = tokens_to_concept_list[0][0]
        concept1 = tokens_to_concept_list[0][1]
        node2 = tokens_to_concept_list[1][0]
        concept2 = tokens_to_concept_list[1][1]
        node_token_pair_1 = (node1, self.current_token)
        node_token_pair_2 = (node2, self.current_token)
        self.stack.append(node_token_pair_1)
        self.stack.append(node_token_pair_2)
        self.actions.append(act.AMRAction("BRK", concept1, node1, concept2, node2))
        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token = self.buffer_indices[0]

    #
    # def can_delete(self):
    #     if self.is_buffer_empty():
    #         return False
    #     return self.current_token not in self.amr_graph.tokens_to_concepts_dict.keys()

    # def can_shift(self):
    #     if self.is_buffer_empty():
    #         return False
    #     return self.current_token in self.amr_graph.tokens_to_concepts_dict.keys()

    def can_delete(self):
        if self.is_buffer_empty():
            return False
        return self.current_token not in self.amr_graph.tokens_to_concepts_dict.keys()

    def can_rotate(self):

        if len(self.stack) < 3:
            return False

        # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, 0) or self.can_reduce(0, top):
            return True
        else:
            return False