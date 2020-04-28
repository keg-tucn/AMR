class AMRNode:

    def __init__(self):
        self.edgeToParent = ""
        self.children = []
        self.token = ""

    def add_child(self, child):
        self.children.append(child)


class CustomizedAMR:

    def __init__(self):
        self.amr_graph = None
        self.parent_dict = {}
        self.relations_dict = {}
        self.tokens_to_concepts_dict = {}
        self.tokens_to_concept_list_dict = {}

    def __printing__representation(self):
        printing_repr = ''
        printing_repr += 'parent_dict: ' + str(self.parent_dict) + '\n'
        printing_repr += 'relations_dict: '+str(self.relations_dict) + '\n'
        printing_repr += 'tokens_to_concepts_dict: '+str(self.tokens_to_concepts_dict) + '\n'
        printing_repr += 'tokens_to_concept_list_dict: '+str(self.tokens_to_concept_list_dict) + '\n'
        printing_repr += 'amr_graph:\n' + str(self.amr_graph) + '\n'
        return printing_repr

    def __str__(self):
        return self.__printing__representation()

    def __repr__(self):
        return self.__printing__representation()

    # In the first phase we will have a tokens dict which
    # just takes the first token aligned to a concept and discards all the others
    # It only works on concepts which have variables assigned to them, not on
    # elements such as polarity
    def create_tokens_to_concepts_dict(self, amr_graph):
        exceptions = ["-", "interrogative"]
        for node_variable in list(amr_graph.node_to_tokens.keys()):
            tokens = amr_graph.node_to_tokens[node_variable]
            if node_variable in list(amr_graph.node_to_concepts.keys()):
                concept = amr_graph.node_to_concepts[node_variable]
                for token in tokens:
                    # condition for the rare case of having a literal among concept variables (quick fix)
                    if type(token) != tuple:
                        self.tokens_to_concepts_dict[int(token)] = (node_variable, concept)
            else:
                # for "-" tokens are given as a (token, node_variable) pair
                # ex: '-': [('5', 'f')
                for token_tuple in tokens:
                    self.tokens_to_concepts_dict[int(token_tuple[0])] = (node_variable, node_variable)

    # TODO: it doesn't take into consideration for example nodes with expressive
    # TODO: it doesn't take into consideration more tokens aligned to the same node
    # (I would need to support some merge action to also need that info)
    def create_tokens_to_concept_list_dict(self, amr_graph):
        exceptions = ["-", "interrogative"]
        for node_variable in list(amr_graph.node_to_tokens.keys()):
            tokens = amr_graph.node_to_tokens[node_variable]
            if node_variable in list(amr_graph.node_to_concepts.keys()):
                for token in tokens:
                    # condition for the rare case of having a literal among concept variables (quick fix)
                    if type(token) != tuple:
                        t = int(token)
                        if t not in list(self.tokens_to_concept_list_dict.keys()):
                            self.tokens_to_concept_list_dict[t] = []
                        concept = amr_graph.node_to_concepts[node_variable]
                        self.tokens_to_concept_list_dict[t].append((node_variable, concept))
            else:
                # exceptions + literals
                for token_tuple in tokens:
                    t = int(token_tuple[0])
                    if t not in list(self.tokens_to_concept_list_dict.keys()):
                        self.tokens_to_concept_list_dict[t] = []
                    self.tokens_to_concept_list_dict[t].append((node_variable, node_variable))


    # This implementation supports having more tokens aligned to the same node
    # def create_tokens_to_concept_list_dict(self, amr_graph):
    #     exceptions = ["-", "interrogative"]
    #     # for each node variable (e.g. 'a', 'b')
    #     for node_variable in amr_graph.node_to_tokens.keys():
    #         tokens = amr_graph.node_to_tokens[node_variable]
    #         # for each token in aligned to node_variable
    #         for token in tokens:
    #             if node_variable in amr_graph.node_to_concepts.keys():
    #                 t = int(token)
    #                 if t not in self.tokens_to_concept_list_dict.keys():
    #                     self.tokens_to_concept_list_dict[t] = []
    #                 concept = amr_graph.node_to_concepts[node_variable]
    #                 self.tokens_to_concept_list_dict[t].append((node_variable, concept))
    #             else:
    #                 if node_variable in exceptions:
    #                     # because token is actually a (token,node_variable) pair in this case
    #                     t = int(token[0])
    #                     if t not in self.tokens_to_concept_list_dict.keys():
    #                         self.tokens_to_concept_list_dict[t] = []
    #                     self.tokens_to_concept_list_dict[t].append((node_variable, node_variable))

    def add_parent_and_edge(self, key, parent, edge, data):
        if (key, parent) not in list(self.relations_dict.keys()) and (key, "") not in list(self.relations_dict.keys()):
            data.edgeToParent = edge
            self.relations_dict[(key, parent)] = (edge, [], "")
        else:
            if (key, "") in list(self.relations_dict.keys()):
                t = list(self.relations_dict[(key, "")])
                t[0] = edge
                self.relations_dict[(key, parent)] = tuple(t)
                self.relations_dict.__delitem__((key, ""))
            else:
                t = list(self.relations_dict[(key, parent)])
                t[0] = edge
                self.relations_dict[(key, parent)] = tuple(t)

    def add_child(self, child, concept):
        if concept not in list(self.parent_dict.keys()) and (concept, "") not in list(self.relations_dict.keys()):
            self.relations_dict[(concept, "")] = ("", [child], "")
        else:
            if (concept, "") in list(self.relations_dict.keys()):
                t = list(self.relations_dict[(concept, "")])
                t[1].append(child)
                self.relations_dict[(concept, "")] = tuple(t)
            else:
                t = list(self.relations_dict[(concept, self.parent_dict[concept])])
                t[1].append(child)
                self.relations_dict[(concept, self.parent_dict[concept])] = tuple(t)

    """
    Each graph item consists of a pair (variable, dictionary of children)
    In the dictionary the edge labels are the keys, and the list of variables connected
    through such a label are the values
    For example, an item would be:
    ('p2', {ARG0-of':[('m',)],'mod':[('p3',),('t',)]})
    In the example above, p2 would be connected to m through the ARG0-of edge,
    p2 would be connected through mod to p3 and t
    """

    def create_custom_AMR(self, amr_graph):
        self.amr_graph = amr_graph
        self.create_tokens_to_concepts_dict(amr_graph)
        self.create_tokens_to_concept_list_dict(amr_graph)

        for item in list(amr_graph.items()):
            data = AMRNode()
            concept = item[0]
            for key, children_list in item[1].items():
                for child in children_list:
                    # because the child is represented as ('p') for example, we get child[0] => 'p'
                    child_var = child[0]
                    self.add_child(child_var, concept)
                    self.add_parent_and_edge(child_var, concept, key, data)
                    self.parent_dict[child_var] = concept
        for key, value in self.relations_dict.items():
            if key[1] == "":
                self.parent_dict[key[0]] = ""

        for item in list(self.relations_dict.items()):
            concept = item[0][0]
            for key, value in amr_graph.node_to_tokens.items():
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
