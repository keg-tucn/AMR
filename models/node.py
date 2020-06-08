class Node:
    def __init__(self, label, tag=None):
        self.label = label
        # added to accommodate "opX" and "wiki" inline quoted information
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
            elif child.tag is not None:
                child_representation = ":%s %s" % (relation, child.tag)
            else:
                if depth < 50:
                    child_representation = ":%s  %s" % (
                        relation, child.amr_print(depth + 1, idx + "_" + str(child_cnt)))
                else:
                    child_representation = ":%s %s" % (relation, "too_deep")
                child_cnt += 1
            representation += "\n".ljust(depth + 1, "\t") + child_representation

        if self.children:
            representation += "\n".ljust(depth, "\t")
        representation += ")"
        return representation

    def amr_print_with_reentrancy(self):
        seen = []
        seen_variables = []

        def amr_print_with_reentrancy_rec(root, depth=1, idx="1"):
            representation = "( d%s / %s " % (idx, root.label)
            seen.append(root)
            seen_variables.append("d" + idx)
            child_cnt = 1
            # children repr
            for (child, relation) in root.children:
                if child in seen:
                    seen_idx = seen.index(child)
                    child_representation = ":%s %s" % (relation, seen_variables[seen_idx])
                # if child not already visited
                else:
                    if relation == "polarity" or relation == "mode":
                        child_representation = ":%s %s" % (relation, child.label)
                    elif child.tag is not None:
                        child_representation = ":%s %s" % (relation, child.tag)
                    else:
                        if depth < 50:
                            child_representation = ":%s  %s" % (
                                relation, amr_print_with_reentrancy_rec(child, depth + 1, idx + "_" + str(child_cnt)))
                        else:
                            child_representation = ":%s %s" % (relation, "too_deep")
                        child_cnt += 1
                representation += "\n".ljust(depth + 1, "\t") + child_representation
            if root.children:
                representation += "\n".ljust(depth, "\t")
            representation += ")"
            return representation

        amr_str = amr_print_with_reentrancy_rec(self)
        return amr_str
