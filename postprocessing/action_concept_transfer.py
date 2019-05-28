from collections import deque

from models.actions import AMRAction

VOCAB_ACTS = ["SH", "RL", "RR", "DN", "SW"]
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

    def load_from_action_objects(self, original_actions):
        """
        Load AMR concept and relation labels from the gold AMR actions into the internal dictionaries
        :param original_actions: list of AMRAction instances
        :return: none
        """
        for action_concept in original_actions:
            action = action_concept.action
            if action == "SH":
                self.node_concepts.append(action_concept.label)
            elif action == "RR" or action == "RL":
                self.relation_concepts.append(action_concept.label)

    def load_from_action_indices_and_labels(self, action_i, label):
        """
        Load AMR concepts and relation labels from the gold AMR labels into the internal dictionaries
        :param action_i: list of action indices
        :param label: list of action labels
        :return: none
        """
        for i in range(len(action_i)):
            if action_i[i] == SH:
                self.node_concepts.append(label[i])
            elif action_i[i] == RR or action_i[i] == RL:
                self.relation_concepts.append(label[i])

    def populate_new_actions(self, new_actions):
        """
        Populate the action indices given as input with the concepts and labels from the dictionaries
        :param new_actions: list of action indices to be populated with labels
        :return: list of AMRAction instances
        """
        result = []
        for action in new_actions:
            if action == SH:
                if len(self.node_concepts) > 0:
                    concept = self.node_concepts.popleft()
                else:
                    concept = "unk"
                predicted_act = AMRAction.build_labeled(action_name(action), concept)
                result.append(predicted_act)
            elif action == RR or action == RL:
                if len(self.relation_concepts) > 0:
                    concept = self.relation_concepts.popleft()
                else:
                    concept = "unk"
                predicted_act = AMRAction.build_labeled(action_name(action), concept)
                result.append(predicted_act)
            else:
                predicted_act = AMRAction.build(action_name(action))
                result.append(predicted_act)
        return result
