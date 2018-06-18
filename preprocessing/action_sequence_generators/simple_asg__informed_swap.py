from asg import ASG
import logging
import amr_util.Actions as act

# should def move the exceptions somewhere else
from preprocessing.ActionSequenceGenerator import SwapException
from preprocessing.ActionSequenceGenerator import TokenOnStackException
"""
    This is the simple deterministic alg (as designed by Silviana)
    but swaps are done only if they can help
    (for ex., swap_1 if reduce can be performed between the stack top and the third element)
"""


class SimpleInformedSwapASG(ASG):

    def __init__(self, no_of_swaps):
        ASG.__init__(self,no_of_swaps)

    def generate_action_sequence(self, amr_graph, sentence):

        ASG.initialize_fields(self, amr_graph, sentence)

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
            else:
                if last_action_swap == self.no_of_swaps:

                    logging.debug(
                        "Last swap didn't solve the stack. Tokens left on the stack: %s. Actions %s.",
                        self.stack, self.actions)
                    raise SwapException("Could not generate action sequence. Swap not working")

                else:
                    if last_action_swap < self.no_of_swaps:
                        if self.can_swap_n(last_action_swap + 1):
                            self.swap_n(last_action_swap + 1)
                            last_action_swap += 1
                        else:
                            # I can still shift or delete
                            if not self.is_buffer_empty():
                                # try to shift the current token
                                if self.can_shift():
                                    self.shift()
                                else:
                                    self.delete()
                            else:
                                # I still have swaps to perform, but the length is not enough
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

        if not ASG.can_swap_n(self, n):
            return False

            # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, top - n - 1) or self.can_reduce(top - n - 1, top):
            return True
        else:
            return False
