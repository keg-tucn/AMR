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

    def add_parent_and_edge(self, key, parent, edge):
        if key not in self.dict.keys():
            data = AMRData()
            data.parent = parent
            data.edgeToParent= edge
            self.dict[key] = data
        else:
            self.dict[key].parent = parent
            self.dict[key].edgeToParent = edge

    def createCustomAMR(self, amrgraph):
        for item in amrgraph.items():
            data = AMRData()
            concept = item[0]
            for key, value in item[1].iteritems():
                data.add_child(value[0][0])
                self.add_parent_and_edge(value[0][0], concept, key)














