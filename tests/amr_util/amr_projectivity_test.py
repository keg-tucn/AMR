from amr_util.amr_projectivity import get_children_list_repr, ChildrenListRepresentation, get_descendants, inorder
from models.amr_data import CustomizedAMR


# TODO: such cases should be better treated in the AMR and CustomizedAMR representation
# ::id bolt-eng-DF-170-181104-8733073_0021.35 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok a
# ::alignments 0-1.1
# (a / amr-unintelligible :value "a"~e.0)
def test_get_children_list_repr():
    custom_amr = CustomizedAMR()
    custom_amr.tokens_to_concepts_dict = {}
    custom_amr.tokens_to_concept_list_dict = {}
    # (child,parent) : (relation, children of child, token aligned to child)
    custom_amr.relations_dict = {('a', 'a'): ('value', ['a'], ['0'])}
    custom_amr.parent_dict = {'a': 'a'}
    generated_list_repr = get_children_list_repr(custom_amr, 'amr_id')
    expected_list_repr = ChildrenListRepresentation()
    expected_list_repr.root = 0
    expected_list_repr.children_dict = {0: [0]}
    assert expected_list_repr == generated_list_repr, \
        'expected ' + str(expected_list_repr) + ' got ' + str(generated_list_repr)


# ::id bolt-eng-DF-170-181104-8733073_0021.35 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok a
# ::alignments 0-1.1
# (a / amr-unintelligible :value "a"~e.0)
def test_get_descendants():
    children_list_repr: ChildrenListRepresentation = ChildrenListRepresentation()
    children_list_repr.root = 0
    children_list_repr.children_dict = {0: [0]}
    generated_descendents = get_descendants(children_list_repr.root,
                                            children_list_repr.children_dict)
    expected_descendents = [0]
    assert generated_descendents == expected_descendents


# ::id bolt-eng-DF-170-181104-8733073_0021.35 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok a
# ::alignments 0-1.1
# (a / amr-unintelligible :value "a"~e.0)
def test_inorder():
    children_list_repr: ChildrenListRepresentation = ChildrenListRepresentation()
    children_list_repr.root = 0
    children_list_repr.children_dict = {0: [0]}
    traversal = []
    inorder(children_list_repr.root, children_list_repr.children_dict, [False, False], traversal)
    expected_traversal = [0]
    assert traversal==expected_traversal


if __name__ == "__main__":
    test_get_children_list_repr()
    test_get_descendants()
    test_inorder()
