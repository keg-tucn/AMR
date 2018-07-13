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

    def __init__(self, no_of_swaps,should_rotate):
        ASG.__init__(self,no_of_swaps)
        self.should_rotate = should_rotate

    def generate_action_sequence(self, amr_graph, sentence):

        ASG.initialize_fields(self, amr_graph, sentence)

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
                #try to rotate, which is still a swapping operation
                if self.should_rotate and self.can_rotate():
                    self.rotate()
                    swapped = True
                if not swapped:
                    if not self.is_buffer_empty():
                        # try to shift the current token
                        if self.can_shift():
                            self.shift()
                        else:
                            self.delete()
                    else:
                        logging.debug("Tokens left on the stack: %s. Actions %s.", self.stack, self.actions)
                        raise TokenOnStackException(
                            "Could not generate action sequence. Tokens left on stack")

        return self.actions

    def can_swap_n(self, n):

        if not ASG.can_swap_n(self, n):
            return False

            # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, top - n - 1) or self.can_reduce(top - n - 1, top):
            return True
        else:
            return False

    def can_rotate(self):

        if len(self.stack) < 3:
            return False

        # check if my swap leads to a reduce
        top = len(self.stack) - 1
        if self.can_reduce(top, 0) or self.can_reduce(0, top):
            return True
        else:
            return False