from asg import ASG
import amr_util.Actions as act
import copy
import sys
import logging
"""
ASG: Action sequence generator
"""


class BacktrackingASG(ASG):

    def __init__(self, no_of_swaps, max_depth):
        ASG.__init__(self,no_of_swaps)
        self.max_depth = max_depth

    def generate_action_sequence(self, amr_graph, sentence):

        ASG.initialize_fields(self, amr_graph, sentence)

        could_generate = self.generate_action_sequence_recursive(0)
        if not could_generate:
            raise Exception("action sequence not found")
        return self.actions

    def generate_action_sequence_recursive(self, depth):
        depth += 1
        if depth > self.max_depth:
            return False

        done = False
        # conditions are checked before entering recursion, but just to make sure
        if self.is_done():
            return True
        # try to delete
        if self.can_delete():
            self.delete()
            # the delete brought the alg into a final state
            if self.is_done():
                return True
            done = self.generate_action_sequence_recursive(depth)
            if done:
                return True
            # this branch doesn't lead to a solution => backtrack
            self.undo_delete()

        else:

            # try to reduce right
            if self.can_reduce_right():
                self.reduce_right()
                # the reduce brought the alg into a final state
                if self.is_done():
                    return True
                # go deeper
                done = self.generate_action_sequence_recursive(depth)
                # the branch led to a solution
                if done:
                    return True
                # no solution on this branch => backtrack
                self.undo_reduce_right()

            # try to reduce left
            if self.can_reduce_left():
                self.reduce_left()
                # the reduce brought the alg into a final state
                if self.is_done():
                    return True
                # go deeper
                done = self.generate_action_sequence_recursive(depth)
                # the branch led to a solution
                if done:
                    return True
                # no solution on this branch => backtrack
                self.undo_reduce_left()

            # try to shift
            if self.can_shift():
                self.shift()
                # won't be done by simply shifting => no need to check we're in a final state
                # go deeper
                done = self.generate_action_sequence_recursive(depth)
                # a solution was found
                if done:
                    return True
                # no sol on this branch
                self.undo_shift()

            # try to swap
            for i in range(1, self.no_of_swaps+1):
                if self.can_swap_n(i):
                    self.swap_n(i)
                    done = self.generate_action_sequence_recursive(depth)
                    if done:
                        return True
                    self.undo_swap_n(i)
        return False

    def can_swap_n(self, n):

        no_of_actions_done = len(self.actions)

        # do I wanna swap
        if self.no_of_swaps == 0:
            return False

        # check I haven't already performed this swap (like, just before)
        if no_of_actions_done>0:
            last_added_action = self.actions[-1]
            action_to_be_performed = BacktrackingASG._get_swap_action_name(n)
            if last_added_action.action == action_to_be_performed:
                return False

        # if no_of_swaps is 3, only allow up to 3 consecutive swap actions
        if no_of_actions_done >= self.no_of_swaps:
            top_actions = len(self.actions) - 1
            all_swaps = True
            for i in range(top_actions, top_actions - self.no_of_swaps,-1):
                action = self.actions[i]
                action_name = self.actions[i].action
                all_swaps = all_swaps and BacktrackingASG._is_swap_action(action_name)
            if all_swaps:
                return False

        # check I have enough tokens on stack
        return len(self.stack) >= n + 2

    def undo_reduce_right(self):

        # add the indice back on stack
        index = self.removed_indices.pop()
        self.stack.append(index)

        # add back the kid (node_var_first in the list of children of node_var_second)
        top = len(self.stack) - 1
        node_var_first = BacktrackingASG._get_concept_for_index(self.amr_graph, self.stack[top])
        node_var_second = BacktrackingASG._get_concept_for_index(self.amr_graph, self.stack[top - 1])
        second_parent = self.amr_graph.parent_dict[node_var_second]
        BacktrackingASG._add_child(self.amr_graph, node_var_second, second_parent, node_var_first)

        #delete action from actions
        self.actions.pop()

    def undo_reduce_left(self):

        # add the indice back on stack
        index = self.removed_indices.pop()
        stack_length = len(self.stack)
        self.stack.insert(stack_length-1,index)

        # add back the kid
        top = len(self.stack) - 1
        node_var_first = BacktrackingASG._get_concept_for_index(self.amr_graph, self.stack[top])
        node_var_second = BacktrackingASG._get_concept_for_index(self.amr_graph, self.stack[top - 1])
        first_parent = self.amr_graph.parent_dict[node_var_first]
        BacktrackingASG._add_child(self.amr_graph, node_var_first, first_parent, node_var_second)

        # delete action from actions
        self.actions.pop()

    def undo_swap_n(self, n):
        top = len(self.stack) - 1
        # we swap the second and (n+1)th node
        aux = self.stack[top - 1]
        self.stack[top - 1] = self.stack[top - n - 1]
        self.stack[top - n - 1] = aux
        # delete the swap action
        self.actions.pop()

    def undo_shift(self):
        self.stack.pop()
        self.actions.pop()
        self.current_token -= 1

    def undo_delete(self):
        self.actions.pop()
        self.current_token -= 1

    @staticmethod
    def _add_child(amr_graph, node, parent, child):
        amr_graph.relations_dict[(node, parent)][1].append(child)

    @staticmethod
    def _is_swap_action(action_name):
        return "SW" in action_name

    @staticmethod
    def _get_swap_action_name(n):
        action_name = "SW"
        if n > 1:
            suffix = "_" + str(n)
            action_name += suffix
        return action_name


"""
In this algorithm implementation, whenever a RL or RR can be performed, it is (same as with delete in the base version)
The only choice at each step is between shifts and swaps
Performing reduce and deletes is similar to the deterministic version (reducing has priority over delete)
"""


class BacktrackingASGFixedReduce(BacktrackingASG):

    def __init__(self, no_of_swaps, max_depth):
        BacktrackingASG.__init__(self, no_of_swaps, max_depth)

    def generate_action_sequence_recursive(self, depth):
        depth += 1
        if depth > self.max_depth:
            return False

        done = False
        # conditions are checked before entering recursion, but just to make sure
        if self.is_done():
            return True

        # try to reduce right
        if self.can_reduce_right():
            self.reduce_right()
            # the reduce brought the alg into a final state
            if self.is_done():
                return True
            # go deeper
            done = self.generate_action_sequence_recursive(depth)
            # the branch led to a solution
            if done:
                return True
            # no solution on this branch => backtrack
            self.undo_reduce_right()

        else:

            # try to reduce left
            if self.can_reduce_left():
                self.reduce_left()
                # the reduce brought the alg into a final state
                if self.is_done():
                    return True
                # go deeper
                done = self.generate_action_sequence_recursive(depth)
                # the branch led to a solution
                if done:
                    return True
                # no solution on this branch => backtrack
                self.undo_reduce_left()

            else:

                # try to delete
                if self.can_delete():
                    self.delete()
                    # the delete brought the alg into a final state
                    if self.is_done():
                        return True
                    done = self.generate_action_sequence_recursive(depth)
                    if done:
                        return True
                    # this branch doesn't lead to a solution => backtrack
                    self.undo_delete()

                else:

                    # try to shift
                    if self.can_shift():
                        self.shift()
                        # won't be done by simply shifting => no need to check we're in a final state
                        # go deeper
                        done = self.generate_action_sequence_recursive(depth)
                        # a solution was found
                        if done:
                            return True
                        # no sol on this branch
                        self.undo_shift()

                    # try to swap
                    for i in range(1, self.no_of_swaps + 1):
                        if self.can_swap_n(i):
                            self.swap_n(i)
                            done = self.generate_action_sequence_recursive(depth)
                            if done:
                                return True
                            self.undo_swap_n(i)

        return False

"""
This is a "merge" between BacktrackingASGInformedSwap and SimpleInformedSwapASG
The alg doesn't choose between shift and swaps only through backtracks, it will
do swaps whenever they lead to a reduce
"""


class BacktrackingASGInformedSwap(BacktrackingASG):

    def __init__(self, no_of_swaps, max_depth):
        BacktrackingASG.__init__(self, no_of_swaps, max_depth)

    def generate_action_sequence_recursive(self, depth):
        depth += 1
        if depth > self.max_depth:
            return False

        done = False
        # conditions are checked before entering recursion, but just to make sure
        if self.is_done():
            return True

        # try to reduce right
        if self.can_reduce_right():
            self.reduce_right()
            # the reduce brought the alg into a final state
            if self.is_done():
                return True
            # go deeper
            done = self.generate_action_sequence_recursive(depth)
            # the branch led to a solution
            if done:
                return True
            # no solution on this branch => backtrack
            self.undo_reduce_right()

        else:

            # try to reduce left
            if self.can_reduce_left():
                self.reduce_left()
                # the reduce brought the alg into a final state
                if self.is_done():
                    return True
                # go deeper
                done = self.generate_action_sequence_recursive(depth)
                # the branch led to a solution
                if done:
                    return True
                # no solution on this branch => backtrack
                self.undo_reduce_left()

            else:

                # try to delete
                if self.can_delete():
                    self.delete()
                    # the delete brought the alg into a final state
                    if self.is_done():
                        return True
                    done = self.generate_action_sequence_recursive(depth)
                    if done:
                        return True
                    # this branch doesn't lead to a solution => backtrack
                    self.undo_delete()

                else:

                    # try to swap
                    for i in range(1, self.no_of_swaps + 1):
                        # when I must do the swap, I do it
                        if self.must_swap_n(i):
                            self.swap_n(i)
                            done = self.generate_action_sequence_recursive(depth)
                            if done:
                                return True
                            self.undo_swap_n(i)
                        else:
                            # I don't have to do the swap, but I might if I can
                            if self.can_swap_n(i):
                                self.swap_n(i)
                                done = self.generate_action_sequence_recursive(depth)
                                if done:
                                    return True
                                self.undo_swap_n(i)

                            # try to shift
                            if self.can_shift():
                                self.shift()
                                # won't be done by simply shifting => no need to check we're in a final state
                                # go deeper
                                done = self.generate_action_sequence_recursive(depth)
                                # a solution was found
                                if done:
                                    return True
                                # no sol on this branch
                                self.undo_shift()

        return False

    def must_swap_n(self, n):

        if not self.can_swap_n(n):
            return False

        # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, top - n - 1) or self.can_reduce(top - n - 1, top):
            return True
        else:
            return False

