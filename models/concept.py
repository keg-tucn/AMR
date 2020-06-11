import re
from typing import List

from models.amr_data import CustomizedAMR
import random

from models.amr_graph import AMR


class Concept:

    def __init__(self, variable, name, no=None):
        self.variable = variable
        self.name = name
        # for concepts with no variable (eg. -), in case there are multiple in the AMR, to differentiate them
        # the first one will have no=0, the second one will have no=1
        self.no = no

    def __repr__(self):
        return '(' + self.variable + ' , ' + self.name + ' , ' + str(self.no) + ')'

    def __str__(self):
        return '(' + self.variable + ' , ' + self.name + ' , ' + str(self.no) + ')'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.variable, self.name, self.no))

    @staticmethod
    def strip_concept_sense(concept_name: str):
        """
        It strips the concept of the sense number
        Eg. recommend-01 => recommend
        Needed for extracting embeddings or pos tags for the concept
        """
        splits = re.split("-([0-9])+", concept_name)
        return splits[0]


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

    # TODO: delete
    def create_from_custom_amr(self, amr_id: str, custom_amr: CustomizedAMR, unaligned_tolerance=0):
        self.amr_id = amr_id
        tokens = list(custom_amr.tokens_to_concept_list_dict.keys())
        tokens.sort()
        for token in tokens:
            for aligned_concept in custom_amr.tokens_to_concept_list_dict[token]:
                concept = Concept(aligned_concept[0], aligned_concept[1])
                if concept not in self.ordered_concepts:
                    self.ordered_concepts.append(concept)
        # put the unaligned concepts in there as well (if allowed by the unaligned_tolerance)
        all_concepts_vars = custom_amr.amr_graph.keys()
        no_unaligned_concepts = len(all_concepts_vars) - len(self.ordered_concepts)
        unaligned_concepts_percentage = no_unaligned_concepts / len(all_concepts_vars)
        if unaligned_concepts_percentage <= unaligned_tolerance:
            # put unaligned concepts at random indexes
            self.__add_unaligned_concepts_randomly(custom_amr, all_concepts_vars)
        else:
            self.ordered_concepts = None

    def create_from_amr(self, amr_id: str, amr: AMR, unaligned_tolerance=0):
        self.amr_id = amr_id
        # create a concept -> token list dict (concept will be obj of type concept)
        concepts_to_tokens = IdentifiedConcepts.__create_concept_to_tokens_dict(amr)
        # create a list of (concept,token) pairs - for now the first token will be taken
        # concepts with no tokens will not be added
        concept_firsttoken_list = []
        for concept, aligned_tokens in concepts_to_tokens.items():
            if len(aligned_tokens) > 0:
                concept_firsttoken_list.append((concept, int(aligned_tokens[0])))
        ordered_concept_token_pairs = sorted(concept_firsttoken_list, key=self.__key_func_concept_token_pair)
        self.ordered_concepts = [concept_tok[0] for concept_tok in ordered_concept_token_pairs]
        # deal with unaligned concepts
        all_concepts = concepts_to_tokens.keys()
        unaligned_concepts_set = set(all_concepts) - set(self.ordered_concepts)
        unaligned_concepts_percentage = len(unaligned_concepts_set) / len(all_concepts)
        if unaligned_concepts_percentage <= unaligned_tolerance:
            # put unaligned concepts at random indexes
            for unaligned_concept in unaligned_concepts_set:
                random_index = random.randint(0, len(self.ordered_concepts) - 1)
                self.ordered_concepts.insert(random_index, unaligned_concept)
        else:
            self.ordered_concepts = None

    @staticmethod
    def __create_concept_to_tokens_dict(amr: AMR):
        concepts_to_tokens = {}
        for var in amr.keys():
            # get aligned tokens if possible
            tokens = []
            if var in amr.node_to_tokens.keys():
                tokens = amr.node_to_tokens[var]
            # differentiate between concepts with vars and without
            if var in amr.node_to_concepts.keys():
                concept = Concept(var, amr.node_to_concepts[var])
                # need to check for tuples because there are some constants with the same values as variables
                # eg. literal "a" in graph which also has variable a
                tokens_list = [t[0] if isinstance(t, tuple) else t for t in tokens]
                concepts_to_tokens[concept] = tokens_list
            else:
                c_t = IdentifiedConcepts.__get_concepts_tokens_list_for_no_var_node(var, tokens)
                concepts_to_tokens.update(c_t)
        return concepts_to_tokens

    @staticmethod
    def __get_concepts_tokens_list_for_no_var_node(var: str, amr_node_to_tokens_entry):
        """
            Returns a list of concepts (with no var) with associated tokens - retrieved from amr.node_to_tokens entry
            eg:
                in amr.node_to_tokens '-': [('10', 's2'), ('21', 's3')],
                need two concepts with associated numbers (bolt-eng-DF-170-181103-8889109_0085.26)

                in amr.node_to_tokens '-': [('2', 'e2'), ('11', 'e2'), ('2', 'e'), ('11', 'e')],
                need two concepts with associated numbers (bolt-eng-DF-170-181103-8889109_0077.18)
        """
        # if node not aligned
        if not amr_node_to_tokens_entry:
            concept = Concept(var, var, 0)
            return {concept: []}
        result_dict = {}
        parent_token_dict = IdentifiedConcepts.get_parents_tokens_list_for_no_var_node(amr_node_to_tokens_entry)
        no = 0
        for parent, token_str_list in parent_token_dict.items():
            concept = Concept(var, var, no)
            no += 1
            result_dict[concept] = token_str_list
        return result_dict

    @staticmethod
    def get_parents_tokens_list_for_no_var_node(amr_node_to_tokens_entry):
        """
            Returns a list of concepts (with no var) with associated tokens - retrieved from amr.node_to_tokens entry
            eg:
                in amr.node_to_tokens '-': [('10', 's2'), ('21', 's3')],
                returns: {'s2':['10'],'s3':['21']

                in amr.node_to_tokens '-': [('2', 'e2'), ('11', 'e2'), ('2', 'e'), ('11', 'e')],
                returns: {'e2':['2','11'],'e':['2','11']}
        """
        parent_token_dict = {}
        for token_str_parent_tuple in amr_node_to_tokens_entry:
            if isinstance(token_str_parent_tuple,tuple):
                # most cases it should be a tuple, these is some weird behaviour for AMRs with "li" relations
                token_str, parent = token_str_parent_tuple
                if parent in parent_token_dict.keys():
                    parent_token_dict[parent].append(token_str)
                else:
                    parent_token_dict[parent] = [token_str]
            else:
                # TODO: treat this better
                return {}
        return parent_token_dict

    def __key_func_concept_token_pair(self, concept_token_pair):
        return concept_token_pair[1]

    def __add_unaligned_concepts_randomly(self, custom_amr: CustomizedAMR, all_concepts_vars):
        all_concepts_vars_set = set(all_concepts_vars)
        ordered_concepts_vars = set([c.variable for c in self.ordered_concepts])
        unaligned_concept_vars = all_concepts_vars_set.difference(ordered_concepts_vars)
        for unaligned_concept_var in unaligned_concept_vars:
            concept_name = custom_amr.get_concept_for_var(unaligned_concept_var)
            concept = Concept(unaligned_concept_var, concept_name)
            random_index = random.randint(0, len(self.ordered_concepts) - 1)
            self.ordered_concepts.insert(random_index, concept)
