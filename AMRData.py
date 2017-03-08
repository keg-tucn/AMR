class AMRNode():
    def __init__(self):
        self.edgeToParent = ""
        self.children = []
        self.token = ""

    def add_child(self, child):
        self.children.append(child)


class CustomizedAMR():
    def __init__(self):
        self.parent_dict = {}
        self.relations_dict = {}
        self.tokens_to_concepts_dict = {}

    # in the first phase we will have a tokens dict which
    # just takes the first token aligned to a concept and discards all the others
    # It only works on concepts which have variables assigned to them, not on
    # elements such as polarity
    def create_tokens_to_concepts_dict(self, amr_graph):

        for node_variable in amr_graph.node_to_tokens.keys():
            tokens = amr_graph.node_to_tokens[node_variable]
            if node_variable in amr_graph.node_to_concepts.keys():
                concept = amr_graph.node_to_concepts[node_variable]
                self.tokens_to_concepts_dict[int(tokens[0])] = (node_variable, concept)

    def add_parent_and_edge(self, key, parent, edge, data):
        if (key, parent) not in self.relations_dict.keys()and (key, "") not in self.relations_dict.keys():
            data.edgeToParent = edge
            self.relations_dict[(key, parent)] = (edge, "", "")
        else:
            if (key, "") in self.relations_dict.keys():
                t = list(self.relations_dict[(key, "")])
                t[0] = edge
                self.relations_dict[(key, parent)] = tuple(t)
                self.relations_dict.__delitem__((key, ""))
            else:
                t = list(self.relations_dict[(key, parent)])
                t[0] = edge
                self.relations_dict[(key, parent)] = tuple(t)

    def add_child(self, child, concept):
        if(concept) not in self.parent_dict.keys() and (concept, "") not in self.relations_dict.keys():
            self.relations_dict[(concept, "")] = ("",[child],"")
        else:
            if (concept, "") in self.relations_dict.keys():
                t = list(self.relations_dict[(concept, "")])
                t[1].append(child)
                self.relations_dict[(concept, "")] = tuple(t)
            else:
                t = list(self.relations_dict[(concept, self.parent_dict[concept])])
                t[1].append(child)
                self.relations_dict[(concept, self.parent_dict[concept])] = tuple(t)

    def create_custom_AMR(self, amr_graph):
        self.create_tokens_to_concepts_dict(amr_graph)

        for item in amr_graph.items():
            data = AMRNode()
            concept = item[0]
            for key, value in item[1].iteritems():
                self.add_child(value[0][0], concept)
                self.add_parent_and_edge(value[0][0], concept, key, data)
                self.parent_dict[value[0][0]] = concept
        for key, value in self.relations_dict.iteritems():
            if key[1] == "":
                self.parent_dict[key[0]] = ""

        for item in self.relations_dict.items():
            concept = item[0][0]
            for key, value in amr_graph.node_to_tokens.iteritems():
                if key is concept:
                    t = list(self.relations_dict[item[0]])
                    if type(value[0]) is tuple:
                        i = 0
                        while value[i][1] != item[0][1]:
                            i += 1
                        t[2] = [value[i][0]]
                    else:
                        t[2] = value
                    self.relations_dict[(key, item[0][1])] = tuple(t)
