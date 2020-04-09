class ASGParameters:
    def __init__(self,
                 asg_alg="simple_informed_swap",
                 no_swaps=1,
                 swap_distance=1,
                 rotate=False
                 ):
        self.asg_alg = asg_alg
        self.no_swaps = no_swaps
        self.swap_distance = swap_distance
        self.rotate = rotate

    @classmethod
    def init_from_dict(cls, params_dict):
        parameters = ASGParameters()
        for key in params_dict:
            setattr(parameters, key, params_dict[key])

        return parameters

# TODO: Find where are these configured? (Should they not be attached to the action seq generators?)
SIMPLE_ACTION_SET = ["SH", "RL", "RR", "DN", "SW"]
SIMPLE_WITH_SWAPS_ACTION_SET = ["SH", "RL", "RR", "DN", "SW", "SW_2", "SW_3"]
SIMPLE_WITH_SWAP_BACK_ACTION_SET = ["SH", "RL", "RR", "DN", "SW_BK"]
SIMPLE_WITH_BREAK_ACTION_SET = ["SH", "RL", "RR", "DN", "SW", "BRK"]
SIMPLE_WITH_SWAPS_BREAK_ACTION_SET = ["SH", "RL", "RR", "DN", "SW", "SW_2", "SW_3", "BRK"]
SIMPLE_WITH_SWAP_BACK_BREAK_SET = ["SH", "RL", "RR", "DN", "BRK", "SW_BK"]
FULL_ACTION_SET = ["SH", "RL", "RR", "DN", "SW", "SW_2", "SW_3", "RO", "BRK", "SW_BK"]


class ActionSet:
    actions = SIMPLE_ACTION_SET

    def __init__(self):
        pass

    @classmethod
    def action_set_size(cls):
        return len(cls.actions)

    @classmethod
    def action_index(cls, action):
        if action == "NONE":
            return len(cls.actions)
        else:
            return cls.actions.index(action)

    @classmethod
    def index_action(cls, index):
        if -1 < index < cls.action_set_size():
            return cls.actions[index]
        else:
            return "NONE"


class ModelParameters:
    def __init__(self,
                 no_stack_tokens=3,
                 no_buffer_tokens=1,
                 no_dep_features=6,
                 embeddings_dim=200,
                 train_epochs=50,
                 hidden_layer_size=1024,
                 learning_rate=0.01,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 ):
        self.no_stack_tokens = no_stack_tokens
        self.no_buffer_tokens = no_buffer_tokens
        self.no_dep_features = no_dep_features
        self.embeddings_dim = embeddings_dim
        self.train_epochs = train_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    @classmethod
    def init_from_dict(cls, params_dict):
        parameters = ModelParameters()
        for key in params_dict:
            setattr(parameters, key, params_dict[key])

        return parameters


class ParserParameters:
    def __init__(self,
                 asg_parameters=ASGParameters(),
                 model_parameters=ModelParameters(),
                 max_len=50,
                 shuffle_data=True,
                 deps_source="stanford",
                 with_enhanced_dep_info=False,
                 with_target_semantic_labels=False,
                 with_reattach=False,
                 with_gold_concept_labels=True,
                 with_gold_relation_labels=True):
        self.asg_parameters = asg_parameters
        self.model_parameters = model_parameters
        self.max_len = max_len
        self.shuffle_data = shuffle_data
        self.deps_source = deps_source
        self.with_enhanced_dep_info = with_enhanced_dep_info
        self.with_target_semantic_labels = with_target_semantic_labels
        self.with_reattach = with_reattach
        self.with_gold_concept_labels = with_gold_concept_labels
        self.with_gold_relation_labels = with_gold_relation_labels

    @classmethod
    def init_from_dict(cls, params_dict):
        parameters = ParserParameters()
        for key in params_dict:
            setattr(parameters, key, params_dict[key])

        parameters.asg_parameters = ASGParameters.init_from_dict(dict(parameters.asg_parameters))
        parameters.model_parameters = ModelParameters.init_from_dict(dict(parameters.model_parameters))

        return parameters
