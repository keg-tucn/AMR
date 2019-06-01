class ASGParameters:
    def __init__(self,
                 asg_alg="nodes_on_stack_break_informed_swap",
                 no_swaps=1,
                 swap_distance=1,
                 rotate=False
                 ):
        self.asg_alg = asg_alg
        self.no_swaps = no_swaps
        self.swap_distance = swap_distance
        self.rotate = rotate
