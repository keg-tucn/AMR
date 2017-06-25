class Node:
    def __init__(self, label, tag=None):
        self.label = label
        # adds tag to accommodate "opX" and "wiki" inline quoted information
        self.tag = tag
        self.children = []

    def add_child(self, obj, relation):
        self.children.append((obj, relation))

    def amr_print(self, depth=1, idx="1"):
        representation = "( d%s / %s " % (idx, self.label)
        child_cnt = 1
        for (child, relation) in self.children:
            if relation == "polarity" or relation == "mode":
                child_representation = ":%s %s" % (relation, child.label)
            elif child.tag is not None and ("op" in relation or relation == "wiki"):
                child_representation = ":%s \"%s\"" % (relation, child.tag)
            else:
                child_representation = ":%s  %s" % (relation, child.amr_print(depth + 1, idx + "_" + str(child_cnt)))
                child_cnt += 1
            representation += "\n".ljust(depth + 1, "\t") + child_representation

        if self.children:
            representation += "\n".ljust(depth, "\t")
        representation += ")"
        return representation
