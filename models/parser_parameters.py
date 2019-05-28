class ParserParameters:
    def __init__(self,
                 max_len=50,
                 with_enhanced_dep_info=False,
                 with_target_semantic_labels=False,
                 with_reattach=False,
                 with_gold_concept_labels=True,
                 with_gold_relation_labels=True):
        self.max_len = max_len
        self.with_enhanced_dep_info = with_enhanced_dep_info
        self.with_target_semantic_labels = with_target_semantic_labels
        self.with_reattach = with_reattach
        self.with_gold_concept_labels = with_gold_concept_labels
        self.with_gold_relation_labels = with_gold_relation_labels
