acts = ['SH', 'RL', 'RR', 'DN', 'SW', 'SW_2', 'SW_3', 'SW_4', 'RO', 'BRK', 'SW_BK']


class AMRAction:

    def __init__(self, action, label, key, label2=None, key2=None):
        self.action = action
        self.label = label
        self.label2 = label2
        self.key = key
        self.key2 = key2
        self.index = acts.index(action)

    def __repr__(self):
        return "action: %s label: %s label2: %s index: %s key: %s key2: %s" % (self.action, self.label, self.label2, self.index, self.key, self.key2)

    @classmethod
    def build(cls, action):
        return AMRAction(action, None, None)

    @classmethod
    def build_labeled(cls, action, label):
        return AMRAction(action, label, None)

if __name__ == "__main__":

    index = 0
    for act in acts:
        amr_action = AMRAction(act, "label", "key")
        assert amr_action.index == index
        index += 1
