from typing import List

from models.amr_data import CustomizedAMR


class Concept:

    def __init__(self, variable, name):
        self.variable = variable
        self.name = name

    def __repr__(self):
        return '(' + self.variable + ' , ' + self.name + ')'

    def __str__(self):
        return '(' + self.variable + ' , ' + self.name + ')'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class IdentifiedConcepts:

    def __init__(self):
        self.amr_id: str = ''
        self.ordered_concepts: List[Concept] = []

    def __repr__(self):
        return '(' + self.amr_id + ' , ' + str(self.ordered_concepts) + ')'

    def __str__(self):
        return '(' + self.amr_id + ' , ' + str(self.ordered_concepts) + ')'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def create_from_custom_amr(self, amr_id: str, custom_amr: CustomizedAMR):
        self.amr_id = amr_id
        tokens = list(custom_amr.tokens_to_concept_list_dict.keys())
        tokens.sort()
        for token in tokens:
            for aligned_concept in custom_amr.tokens_to_concept_list_dict[token]:
                concept = Concept(aligned_concept[0], aligned_concept[1])
                if concept not in self.ordered_concepts:
                    self.ordered_concepts.append(concept)
