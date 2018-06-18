from asg import ASG
import logging
import amr_util.Actions as act

# should def move the exceptions somewhere else
from preprocessing.ActionSequenceGenerator import SwapException
from preprocessing.ActionSequenceGenerator import TokenOnStackException
from preprocessing.ActionSequenceGenerator import RotateException

""" This is the deterministic version of the algorithm for action sequence generation
    as described in Silviana Campean's thesis
    (the code being refactored to use the methods in the ASG class)
"""


class SimpleASG(ASG):

    def __init__(self, no_of_swaps, should_rotate):
        ASG.__init__(self,no_of_swaps)
        self.should_rotate = should_rotate

    def generate_action_sequence(self, amr_graph, sentence):

        ASG.initialize_fields(self, amr_graph, sentence)

        last_action_swap = 0
        last_rotate = False
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
                last_rotate = False
            else:
                # def should make sure only either swap or rotate happens
                if self.is_buffer_empty():  # this means I can no longer shift or delete

                    if last_action_swap == self.no_of_swaps:

                        # try to rotate
                        if self.should_rotate and (not last_rotate) and (len(self.stack) >= 3):
                            self.rotate()
                            last_rotate = True
                        else:
                            if last_rotate:
                                logging.debug(
                                    "Last rotate didn't solve the stack. Tokens left on the stack: %s. Actions %s.",
                                    self.stack, self.actions)
                                raise RotateException("Could not generate action sequence. Rotate not working")
                            else:
                                logging.debug(
                                    "Last swap didn't solve the stack. Tokens left on the stack: %s. Actions %s.",
                                    self.stack, self.actions)
                                raise SwapException("Could not generate action sequence. Swap not working")

                    else:
                        if last_action_swap < self.no_of_swaps:
                            if self.can_swap_n(last_action_swap + 1):
                                self.swap_n(last_action_swap + 1)
                                last_action_swap += 1
                                last_rotate = False
                            else:
                                # I still have swaps to perform, but the length is not enough
                                logging.debug("Tokens left on the stack: %s. Actions %s.", self.stack, self.actions)
                                raise TokenOnStackException(
                                    "Could not generate action sequence. Tokens left on stack")
                else:
                    # try to shift the current token
                    if self.can_shift():
                        self.shift()
                    else:
                        self.delete()
        return self.actions

