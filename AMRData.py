class AMRData():

    def __init__(self):

        self.parent = ""
        self.edgeToParent = ""
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class CustomizedAMR():

    def __init__(self):
        self.dict ={}

    def add_parent(self, key, parent):
        if key not in self.dict.keys():
            data = AMRData()
            data.parent = parent
            self.dict[key] = data
        else:
            self.dict[key].parent = parent

    def createCustomAMR(self, amrgraph):
        for item in amrgraph.items():
            data = AMRData()
            concept = amrgraph.node_to_concepts[item[0]]
            for value in item[1].itervalues():
                data.add_child(value[0][0])
                self.add_parent(value[0][0], concept)














