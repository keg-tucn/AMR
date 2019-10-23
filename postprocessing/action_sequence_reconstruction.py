from amr_util import tokenizer_util
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from models.node import Node
from models.parameters import ParserParameters
from postprocessing import concept_identification, relation_identification
from postprocessing.action_concept_transfer import ActionConceptTransfer
from preprocessing import ActionSequenceGenerator
from preprocessing import TokensReplacer


def reconstruct_all_ne(tokens, action_sequence, named_entities_metadata, date_entities_metadata, parser_parameters):
    rec_obj = MetadataReconstructionState(tokens, named_entities_metadata, date_entities_metadata, parser_parameters)

    for action in action_sequence:
        rec_obj.process_action(action)

    top = rec_obj.finalize()

    if not parser_parameters.with_gold_concept_labels:
        concept_identification.annotate_node_concepts(top)

    if not parser_parameters.with_gold_relation_labels:
        relation_identification.annotate_node_relations(top)

    return top


class MetadataReconstructionState:
    def __init__(self, tokens, _named_entity_metadata, _date_entity_metadata, parser_parameters):
        self.tokens = tokens
        self.named_entity_metadata = _named_entity_metadata
        self.date_entity_metadata = _date_entity_metadata
        self.parser_parameters = parser_parameters
        self.buffer_indices = range(256)
        self.current_token_index = 0
        self.stack = []
        self.index_word_map = tokenizer_util.get_index_word_map()

    def process_action(self, action):
        # execute the action to update the parser state
        # TODO: split named/date entities replacement of concept after we reconstruct the graph
        # TODO: reference named/date entitites by concept instead of by index
        if action.action == "SH":
            self._process_shift(action)
        elif action.action == "RL":
            self._process_reduce_left(action)
        elif action.action == "RR":
            self._process_reduce_right(action)
        elif action.action == "DN":
            self._process_delete(action)
        elif action.action == "BRK":
            self._process_break(action)
        elif action.action == "SW":
            self._process_swap(action)
        elif action.action == "SW_2":
            self._process_swap_2(action)
        elif action.action == "SW_3":
            self._process_swap_3(action)
        elif action.action == "RO":
            self._process_rotate(action)
        elif action.action == "SW_BK":
            self._process_swap_back(action)

    def finalize(self):
        top = self.stack.pop()

        return top

    def _process_shift(self, action):
        current_token = self.tokens[self.current_token_index]

        if self.parser_parameters.with_reattach and self._is_named_entity():
            if self.parser_parameters.with_gold_concept_labels:
                node = self._make_named_entity(action.label, self.named_entity_metadata[0][1])
            else:
                node = self._make_named_entity(self.index_word_map[current_token], self.named_entity_metadata[0][1])
            self.named_entity_metadata.pop(0)
        elif self.parser_parameters.with_reattach and self._is_date_entity():
            node = self._make_date_entity(self.date_entity_metadata[0][1], self.date_entity_metadata[0][2])
            self.date_entity_metadata.pop(0)
        else:
            if self.parser_parameters.with_gold_concept_labels:
                node = Node(action.label)
            else:
                node = Node(self.index_word_map[current_token])

        self.stack.append(node)

        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token_index = self.buffer_indices[0]

    def _process_reduce_left(self, action):
        right = self.stack.pop()
        left = self.stack.pop()
        head, modifier = right, left
        if self.parser_parameters.with_gold_relation_labels:
            head.add_child(modifier, action.label)
        else:
            head.add_child(modifier, "unk")
        self.stack.append(head)

    def _process_reduce_right(self, action):
        right = self.stack.pop()
        left = self.stack.pop()
        head, modifier = left, right
        if self.parser_parameters.with_gold_relation_labels:
            head.add_child(modifier, action.label)
        else:
            head.add_child(modifier, "unk")
        self.stack.append(head)

    def _process_delete(self, action):
        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token_index = self.buffer_indices[0]

    def _process_break(self, action):
        node1 = Node(action.label)
        node2 = Node(action.label2)
        self.stack.append(node1)
        self.stack.append(node2)

        self.buffer_indices.pop(0)
        if len(self.buffer_indices) != 0:
            self.current_token_index = self.buffer_indices[0]

    def _process_swap(self, action):
        top = self.stack.pop()
        mid = self.stack.pop()
        lower = self.stack.pop()
        self.stack.append(mid)
        self.stack.append(lower)
        self.stack.append(top)

    def _process_swap_2(self, action):
        top = self.stack.pop()
        mid = self.stack.pop()
        mid2 = self.stack.pop()
        lower = self.stack.pop()
        self.stack.append(mid)
        self.stack.append(mid2)
        self.stack.append(lower)
        self.stack.append(top)

    def _process_swap_3(self, action):
        top = self.stack.pop()
        mid = self.stack.pop()
        mid2 = self.stack.pop()
        mid3 = self.stack.pop()
        lower = self.stack.pop()
        self.stack.append(mid)
        self.stack.append(mid3)
        self.stack.append(mid2)
        self.stack.append(lower)
        self.stack.append(top)

    def _process_rotate(self, action):
        top = self.stack.pop()
        second = self.stack.pop()
        bottom = self.stack.pop(0)
        self.stack.insert(0, second)
        self.stack.append(bottom)
        self.stack.append(top)

    def _process_swap_back(self, action):
        top = len(self.stack) - 1
        j = self.stack.pop(top - 1)
        self.buffer_indices.insert(0, j)

    def _is_named_entity(self):
        return len(self.named_entity_metadata) > 0 and self.current_token_index == self.named_entity_metadata[0][0]

    def _is_date_entity(self):
        return len(self.date_entity_metadata) > 0 and self.current_token_index == self.date_entity_metadata[0][0]

    def _make_named_entity(self, concept, literals):
        wiki_tag = "_".join(literals)
        wiki_node = Node(wiki_tag, tag=wiki_tag)
        name_node = Node("name")
        for i, literal in enumerate(literals):
            literal_node = Node(literal, tag=literal)
            name_node.add_child(literal_node, "op{}".format(i + 1))
        concept_node = Node(concept)
        concept_node.add_child(wiki_node, "wiki")
        concept_node.add_child(name_node, "name")
        return concept_node

    def _make_date_entity(self, date_relations, quantities):
        date_entity_node = Node("date-entity")
        for date_relation, quantity in zip(date_relations, quantities):
            date_entity_node.add_child(Node(quantity, quantity), date_relation)
        return date_entity_node


if __name__ == "__main__":

    parser_parameters = ParserParameters(max_len=50, with_enhanced_dep_info=False,
                                         with_target_semantic_labels=False, with_reattach=True,
                                         with_gold_concept_labels=True, with_gold_relation_labels=True)

    sentence = "It looks like we will also bring in whales ."
    amr = AMR.parse_string("""
    (l / look-02~e.1
          :ARG1~e.2 (b / bring-01~e.6
                :ARG0 (w / we~e.3)
                :ARG1~e.7 (w2 / whale~e.8)
                :mod (a / also~e.5)))
    """)

    custom_AMR = CustomizedAMR()
    custom_AMR.create_custom_AMR(amr)

    actions = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence)
    acts_i = [a.index for a in actions]

    act = ActionConceptTransfer()
    act.load_from_action_objects(actions)
    actions_re = act.populate_new_actions(acts_i)

    tokens = tokenizer_util.text_to_sequence(sentence)

    print "Original actions:"
    for act in actions:
        print act
    print "Reconstructed actions:"
    for act in actions_re:
        print act

    amr_re = reconstruct_all_ne(tokens, actions, [], [], parser_parameters=parser_parameters)
    print amr_re.amr_print()

    sentence = "upgrade fire control systems of Indian tanks ."

    amr_str = """(u / upgrade-02~e.0
          :ARG1 (s / system~e.3
                :ARG0-of (c / control-01~e.2
                      :ARG1 (f / fire-01~e.1))
                :poss~e.4 (t / tank~e.6
                      :mod (c2 / country :wiki "India"
                            :name (n / name :op1 "India"~e.5)))))"""
    amr = AMR.parse_string(amr_str)
    amr_new, sentence_new, _ = TokensReplacer.replace_named_entities(amr, sentence)
    custom_AMR = CustomizedAMR()
    custom_AMR.create_custom_AMR(amr_new)

    actions = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence_new)
    acts_i = [a.index for a in actions]

    act = ActionConceptTransfer()
    act.load_from_action_objects(actions)
    actions_re = act.populate_new_actions(acts_i)

    tokens_to_concepts_dict = custom_AMR.tokens_to_concepts_dict

    print actions
    print actions_re

    amr_re = reconstruct_all_ne(tokens, actions, [(5, ["Indian"])], [], parser_parameters)
    print amr_re.amr_print()

    amr_str = """(d / difficult~e.5
          :domain~e.4 (r / reach-01~e.7
                :ARG1 (c / consensus~e.0
                      :topic~e.1 (c2 / country :wiki "India"
                            :name (n / name :op1 "India"~e.2)))
                :time~e.8 (m / meet-03~e.11
                      :ARG0 (o / organization :wiki "Nuclear_Suppliers_Group"
                            :name (n2 / name :op1 "NSG"~e.10))
                      :time~e.12 (d2 / date-entity :year 2007~e.14 :month~e.13 11~e.13))))"""
    sentence = """Consensus on India will be difficult to reach when the NSG meets in November 2017 ."""

    amr = AMR.parse_string(amr_str)
    amr_new, sentence_new, named_entities = TokensReplacer.replace_named_entities(amr, sentence)
    amr_new, sentence_new, date_entities = TokensReplacer.replace_date_entities(amr_new, sentence_new)

    custom_AMR = CustomizedAMR()
    custom_AMR.create_custom_AMR(amr_new)

    actions = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence_new)
    acts_i = [a.index for a in actions]

    act = ActionConceptTransfer()
    act.load_from_action_objects(actions)
    actions_re = act.populate_new_actions(acts_i)

    tokens = tokenizer_util.text_to_sequence(sentence)

    print actions
    print actions_re
    print reconstruct_all_ne(tokens, actions, [], [], parser_parameters)

    print reconstruct_all_ne(tokens, actions, [(2, ["India"]), (10, ["NSG"])], [(13, ["month", "year"], [2017, 11])],
                             parser_parameters)
