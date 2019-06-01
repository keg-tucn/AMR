from asg_parameters import ASGParameters
from model_parameters import ModelParameters


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
