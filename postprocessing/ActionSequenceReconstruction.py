from collections import deque

from AMRData import CustomizedAMR
from AMRGraph import AMR
from amr_util.Node import Node
from preprocessing import ActionSequenceGenerator
from preprocessing import TokensReplacer

VOCAB_ACTS = ['SH', 'RL', 'RR', 'DN', 'SW']
SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NUM_ACTIONS = len(VOCAB_ACTS)


def action_name(index):
    return VOCAB_ACTS[index]


class ActionConceptTransfer:
    def __init__(self):
        self.node_concepts = deque()
        self.relation_concepts = deque()

    def load_from_verbose(self, original_actions):
        for action_concept in original_actions:
            action = action_concept[:2]
            if action == 'SH':
                self.node_concepts.append(action_concept[3:])
            elif action == 'RR' or action == 'RL':
                self.relation_concepts.append(action_concept[3:])

    def load_from_action_and_label(self, action_i, label):
        for i in range(len(action_i)):
            if action_i[i] == SH:
                self.node_concepts.append(label[i])
            elif action_i[i] == RR or action_i[i] == RL:
                self.relation_concepts.append(label[i])

    def populate_new_actions(self, new_actions):
        result = []
        for action in new_actions:
            if action == SH:
                if len(self.node_concepts) > 0:
                    concept = self.node_concepts.popleft()
                else:
                    concept = 'unk'
                result.append(action_name(action) + '_' + concept)
            elif action == RR or action == RL:
                if len(self.relation_concepts) > 0:
                    concept = self.relation_concepts.popleft()
                else:
                    concept = 'unk'
                result.append(action_name(action) + '_' + concept)
            else:
                result.append(action_name(action))
        return result


def reconstruct_all(action_sequence):
    rec_obj = ReconstructionState()
    for action in action_sequence:
        rec_obj.process_action(action[:2], action[3:])
    top = rec_obj.finalize()
    return top.amr_print()


def reconstruct_all_ne(action_sequence, named_entities_metadata):
    rec_obj = MetadataReconstructionState(named_entities_metadata)
    for action in action_sequence:
        rec_obj.process_action(action[:2], action[3:])
    top = rec_obj.finalize()
    return top.amr_print()


class MetadataReconstructionState:
    def __init__(self, _named_entity_metadata):
        self.named_entity_metadata = _named_entity_metadata
        self.current_token_index = 0
        self.stack = []

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

    def _process_action_ne(self, action, concept):
        current_ne_index = self.named_entity_metadata[0][0]

        if action == 'SH':
            if self.current_token_index == current_ne_index:
                node = self._make_named_entity(concept, self.named_entity_metadata[0][1])
                self.named_entity_metadata.pop()
            else:
                node = Node(concept)
            self.stack.append(node)
            self.current_token_index += 1
        elif action == 'DN':
            self.current_token_index += 1
            pass
        elif action == 'SW':
            top = self.stack.pop()
            mid = self.stack.pop()
            lower = self.stack.pop()
            self.stack.append(mid)
            self.stack.append(lower)
            self.stack.append(top)
        else:  # one of the reduce actions
            right = self.stack.pop()
            left = self.stack.pop()
            head, modifier = (left, right) if action == 'RR' else (right, left)
            head.add_child(modifier, concept)
            self.stack.append(head)

    def process_action(self, action, concept):
        # execute the action to update the parser state
        if len(self.named_entity_metadata) > 0:
            self._process_action_ne(action, concept)
        else:
            if action == 'SH':
                node = Node(concept)
                self.stack.append(node)
                self.current_token_index += 1
            elif action == 'DN':
                self.current_token_index += 1
                pass
            elif action == 'SW':
                top = self.stack.pop()
                mid = self.stack.pop()
                lower = self.stack.pop()
                self.stack.append(mid)
                self.stack.append(lower)
                self.stack.append(top)
            else:  # one of the reduce actions
                right = self.stack.pop()
                left = self.stack.pop()
                head, modifier = (left, right) if action == 'RR' else (right, left)
                head.add_child(modifier, concept)
                self.stack.append(head)

    def finalize(self):
        top = self.stack.pop()
        return top


class ReconstructionState:
    def __init__(self):

        self.stack = []

    def process_action(self, action, concept):
        # execute the action to update the parser state
        if action == 'SH':
            node = Node(concept)
            self.stack.append(node)
        elif action == 'DN':
            pass
        elif action == 'SW':
            top = self.stack.pop()
            mid = self.stack.pop()
            lower = self.stack.pop()
            self.stack.append(mid)
            self.stack.append(lower)
            self.stack.append(top)
        else:  # one of the reduce actions
            right = self.stack.pop()
            left = self.stack.pop()
            head, modifier = (left, right) if action == 'RR' else (right, left)
            head.add_child(modifier, concept)
            self.stack.append(head)

    def finalize(self):
        top = self.stack.pop()
        return top


if __name__ == "__main__":
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
    acts_short = [a[:2] for a in actions]
    acts_i = [VOCAB_ACTS.index(a) for a in acts_short]

    act = ActionConceptTransfer()
    act.load_from_verbose(actions)
    actions_re = act.populate_new_actions(acts_i)

    tokens_to_concepts_dict = custom_AMR.tokens_to_concepts_dict

    print actions
    print acts_short
    print actions_re

    print reconstruct_all(actions)

    sentence = "upgrade fire control systems of Indian tanks ."

    amr_str = """(u / upgrade-02~e.0
          :ARG1 (s / system~e.3
                :ARG0-of (c / control-01~e.2
                      :ARG1 (f / fire-01~e.1))
                :poss~e.4 (t / tank~e.6
                      :mod (c2 / country :wiki "India"
                            :name (n / name :op1 "India"~e.5)))))"""
    amr = AMR.parse_string(amr_str)
    amr_new, sentence_new, named_entities = TokensReplacer.replace_named_entities(amr, sentence)
    custom_AMR = CustomizedAMR()
    custom_AMR.create_custom_AMR(amr_new)

    actions = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence)
    acts_short = [a[:2] for a in actions]
    acts_i = [VOCAB_ACTS.index(a) for a in acts_short]

    act = ActionConceptTransfer()
    act.load_from_verbose(actions)
    actions_re = act.populate_new_actions(acts_i)

    tokens_to_concepts_dict = custom_AMR.tokens_to_concepts_dict

    print actions
    print acts_short
    print actions_re

    print reconstruct_all_ne(actions, [(5, ["Indian"])])
