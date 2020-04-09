from .parameters import ActionSet


class AMRAction:
    def __init__(self, action, label, key, label2=None, key2=None):
        self.action = action
        self.label = label
        self.label2 = label2
        self.key = key
        self.key2 = key2
        self.index = ActionSet.action_index(self.action)

    def __repr__(self):
        return "action: %s label: %s label2: %s index: %s key: %s key2: %s" % (
            self.action, self.label, self.label2, self.index, self.key, self.key2)

    def to_string(self):
        if not self.label:
            return self.action
        if not self.label2:
            return self.action + "_" + self.label
        return self.action + "_" + self.label + "_" + self.label2

    @classmethod
    def build(cls, action):
        return AMRAction(action, None, None)

    @classmethod
    def build_labeled(cls, action, label):
        return AMRAction(action, label, None)
