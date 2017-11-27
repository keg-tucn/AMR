acts = ['SH', 'RL', 'RR', 'DN', 'SW']

class AMRAction:
    def __init__(self, action, label, key):
        self.action = action
        self.label = label
        self.key = key
        self.index = acts.index(action)

    def __repr__(self):
        return "action: %s label: %s index: %s key: %s" % (self.action, self.label, self.index, self.key)

    @classmethod
    def build(cls, action):
        AMRAction(action, "n/a", "n/a")

    @classmethod
    def build_labeled(cls, action, label):
        AMRAction(action, label, "n/a")

if __name__ == "__main__":

    index = 0
    for act in acts:
        amr_action = AMRAction(act, "label", "key")
        assert amr_action.index == index
        index += 1
