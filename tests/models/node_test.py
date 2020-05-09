from models.node import Node


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_amr_print_simple():
    r: Node = Node('recommend-01')
    a: Node = Node('advocate-01')
    i: Node = Node('it')
    v: Node = Node('vigorous')
    r.add_child(a, 'ARG1')
    a.add_child(i, 'ARG1')
    a.add_child(v, 'manner')
    generated_amr_str = r.amr_print()
    generated_amr_no_spaces = ''.join(generated_amr_str.split())
    expected_amr_str = """( d1 / recommend-01 
                            :ARG1  ( d1_1 / advocate-01 
                                :ARG1  ( d1_1_1 / it )
                                :manner  ( d1_1_2 / vigorous )
                            )
                        )"""
    expected_amr_no_spaces = ''.join(expected_amr_str.split())
    assert generated_amr_no_spaces == expected_amr_no_spaces, \
        'expected \n' + expected_amr_str + '\ngot\n' + generated_amr_str + '\n'


# id DF-200-192451-579_6283.20
# (b / bad-07~e.12 :polarity~e.9 -~e.9
#       :ARG1 (i / imitate-01~e.0
#             :ARG1 (c / country~e.4
#                   :ARG1-of (d / develop-02~e.3
#                         :degree (m / most~e.2))
#                   :location~e.5 (w / world~e.7)))
#       :ARG1-of (r / real-04~e.10))
def test_amr_print_with_polarity():
    b: Node = Node('bad')
    neg: Node = Node('-')
    i: Node = Node('imitate-01')
    c: Node = Node('country')
    d: Node = Node('develop-02')
    m: Node = Node('most')
    w: Node = Node('world')
    r: Node = Node('real-04')
    b.add_child(neg, 'polarity')
    b.add_child(i, 'ARG1')
    b.add_child(r, 'ARG1-of')
    i.add_child(c, 'ARG1')
    c.add_child(d, 'ARG1-of')
    c.add_child(w, 'location')
    d.add_child(m, 'degree')
    generated_amr_str = b.amr_print()
    generated_amr_no_spaces = ''.join(generated_amr_str.split())

    expected_amr_str = """( d1 / bad 
                            :polarity -
                            :ARG1  ( d1_1 / imitate-01 
                                :ARG1  ( d1_1_1 / country 
                                    :ARG1-of  ( d1_1_1_1 / develop-02 
                                        :degree  ( d1_1_1_1_1 / most )
                                    )
                                    :location  ( d1_1_1_2 / world )
                                )
                            )
                            :ARG1-of  ( d1_2 / real-04 )
                        )"""
    expected_amr_no_spaces = ''.join(expected_amr_str.split())
    assert generated_amr_no_spaces == expected_amr_no_spaces, \
        'expected \n' + expected_amr_str + '\ngot\n' + generated_amr_str + '\n'


# id DF-200-192451-579_6300.1
# (r / realize-01 :polarity~e.3 -~e.3
#       :ARG0 (i / i~e.0)
#       :ARG1 (t2 / threaten-01~e.11
#             :ARG0 (c2 / country :wiki "Iran"
#                   :name (n / name :op1 "Iran"~e.6))
#             :mod (h / huge~e.10
#                   :degree (s2 / such~e.8))
#             :ARG1-of (c / cause-01~e.5
#                   :ARG0~e.5 (a / amr-unknown~e.5)))
#       :manner (s / simple~e.1))
def test_amr_print_with_literal():
    r: Node = Node('realize-01')
    neg: Node = Node('-')
    i: Node = Node('it')
    t2: Node = Node('threaten-01')
    c2: Node = Node('country')
    iran1: Node = Node(None, 'Iran')
    n: Node = Node('name')
    iran2: Node = Node(None, 'Iran')
    h: Node = Node('huge')
    s2: Node = Node('such')
    c: Node = Node('cause')
    a: Node = Node('amr-unkown')
    s: Node = Node('simple')
    r.add_child(neg, 'polarity')
    r.add_child(i, 'ARG0')
    r.add_child(t2, 'ARG1')
    r.add_child(s, 'manner')
    t2.add_child(c2, 'ARG0')
    t2.add_child(h, 'mod')
    t2.add_child(c, 'ARG1-of')
    c2.add_child(iran1, 'wiki')
    c2.add_child(n, 'name')
    n.add_child(iran2, 'op1')
    h.add_child(s2, 'degree')
    c.add_child(a, 'ARG0')
    generated_amr_str = r.amr_print()
    print(generated_amr_str)


# TODO: example with reantrancy


if __name__ == "__main__":
    test_amr_print_simple()
    test_amr_print_with_polarity()
    test_amr_print_with_literal()
    print("Everything in node_test passed")
