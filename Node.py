class Node:
    def __init__(self, label, node_index):
        self.label = label
        self.children = []
        self.node_index = node_index

    def add_child(self, obj, relation):
        self.children.append((obj, relation))

    def amr_print(self, depth=1):
        str = "( d%s / %s " % (self.node_index, self.label)
        for (child, relation) in self.children:
            if relation == "polarity" or relation == "mode":
                child_representation = ":%s %s" % (relation, child.label)
            else:
                child_representation = ":%s  %s" % (relation, child.amr_print(depth + 1))
            str += "\n".ljust(depth + 1, "\t") + child_representation

        if self.children:
            str += "\n".ljust(depth, "\t")
        str += ")"
        return str
