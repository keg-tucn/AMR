class AmrNotPerfectlyAlignedTreeException(Exception):
    pass


class ChildrenListRepresentation:
    def __init__(self):
        self.children_dict = {}
        self.root = 0


def get_children_list_repr(amr, amr_id):
    """
    :param amr: given as a set of dictionaries (instance of CustomAMR)
    :param amr_id
    :return: a children_list representation for the amr
    """
    c = ChildrenListRepresentation()

    for key in list(amr.relations_dict.keys()):
        # key is a pair (node_variable, parent_variable)
        node = key[0]
        # a tuple of the form (relation, children_list, token_list)
        node_rel = amr.relations_dict[key]
        node_token = int(node_rel[2][0])

        # see if the current node is the root
        parent = key[1]
        if parent == '':
            c.root = node_token

        # add all the children of node (add the tokens they're assocaiated to)
        for child in node_rel[1]:
            child_rel = amr.relations_dict[(child, node)]
            child_token = int(child_rel[2][0])
            if node_token not in c.children_dict:
                c.children_dict[node_token] = []
            c.children_dict[node_token].append(child_token)

    # make sure children list is ordered
    for child_list in list(c.children_dict.values()):
        child_list.sort()

    return c


def get_descendants(node, children_dict):
    # print("descendants from {0}".format(node))

    if node not in list(children_dict.keys()):
        return []

    descendents = []

    for c in children_dict[node]:
        descendents.append(c)
        descendents = descendents + get_descendants(c, children_dict)

    return descendents


def has_smaller_descendents(node, descendents, added):
    """

    :param node: current node
    :param descendents: descendents of current node
    :param added: flags (added[v]==True <=> v was added to the projective order)
    :return: true or false (are there any descendents smaller than node that were not yet added to proj order)
    """

    for d in descendents:
        if d < node and added[d] == False:
            return True
    return False


def inorder(node, children_dict, added, traversal):
    descendants = get_descendants(node, children_dict)
    if not has_smaller_descendents(node, descendants, added):
        traversal.append(node)
        added[node] = True

    if node in list(children_dict.keys()):

        for child in children_dict[node]:
            inorder(child, children_dict, added, traversal)
            # now that we processed some more of the descendents, try to add the node again
            # (in case it wasn't already added)
            if not added[node] and not has_smaller_descendents(node, descendants, added):
                traversal.append(node)
                added[node] = True


def is_tree(amr):
    nodes = []
    for key in list(amr.relations_dict.keys()):
        node = key[0]
        if node not in nodes:
            nodes.append(node)
        elif node in ['-', 'interogative', 'expressive']:
            nodes.append(node)
    no_nodes = len(nodes)
    no_edges = len(amr.relations_dict) - 1
    return no_edges == no_nodes - 1


# test that each token is either aligned to a node or none at all
# and each node is aligned to 1 token
def is_perfectly_aligned(amr):
    all_aligned_tokens = []
    # check each node has tokens aligned
    for key in list(amr.relations_dict.keys()):
        aligned_tokens = amr.relations_dict[key][2]
        if aligned_tokens == '':
            return False
        if len(aligned_tokens) != 1:
            return False
        all_aligned_tokens.append(int(aligned_tokens[0]))

    # check that the same token doesn't appear for more than one node
    all_aligned_tokens_set = set(all_aligned_tokens)
    return len(all_aligned_tokens) == len(all_aligned_tokens_set)


def get_projective_order(amr, amr_id):
    """

    :param amr: given as a set of dictionaries
    :return: list of tokens in projective order
    :description: returns a list of token indexes that corresponds to the
    projective order of the tree (obtained by an inorder traversal)
    """

    if (not is_tree(amr)) or (not is_perfectly_aligned(amr)):
        raise AmrNotPerfectlyAlignedTreeException(amr_id)

    # first, construct a children_list representation for the amr
    c = get_children_list_repr(amr, amr_id)

    # initialize flag vector
    max_sen_length = 300
    added = [False for i in range(max_sen_length)]

    # inorder traversal of the tree
    # (take a node as long as it has no descendants smaller than it)
    projective_order = []
    inorder(c.root, c.children_dict, added, projective_order)

    return projective_order
