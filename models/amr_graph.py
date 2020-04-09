#!/usr/bin/python
# -*- coding:utf-8 -*-


# A hypergraph representation for amr.

# author: Chuan Wang
# since: 2013-11-20

from util import *
import sys
import re
from optparse import OptionParser
from .dependency_graph import *


# Error definitions
class LexerError(Exception):
    pass


class ParserError(Exception):
    pass


class Node():
    # node_id = 0     #static counter, unique for each node
    # mapping_table = {}  # old new index mapping table

    def __init__(self, parent, trace, node_label, firsthit, leaf, depth, seqID):
        """
        initialize a node in the graph
        here a node keeps record of trace i.e. from where the node is reached (the edge label)
        so nodes with same other attributes may have different trace
        """
        self.parent = parent
        self.trace = trace
        self.node_label = node_label
        self.firsthit = firsthit
        self.leaf = leaf
        self.depth = depth
        self.children = []
        self.seqID = seqID
        # Node.node_id += 1
        # self.node_id = node_id

    def __str__(self):
        return str((self.trace, self.node_label, self.depth, self.seqID))

    def __repr__(self):
        return str((self.trace, self.node_label, self.depth, self.seqID))


class AMR(defaultdict):
    """
    An abstract meaning representation.
    Basic idea is based on bolinas' hypergraph for amr.

    Here one AMR is a rooted, directed, acyclic graph.
    We also use the edge-label style in bolinas.
    """

    def __init__(self, *args, **kwargs):

        defaultdict.__init__(self, ListMap, *args, **kwargs)
        self.roots = []
        self.external_nodes = {}

        # attributes to be added
        self.node_to_concepts = {}
        self.node_to_tokens = {}
        self.relation_to_tokens = {}
        self.align_to_sentence = None

        self.reentrance_triples = []

    @classmethod
    def parse_string(cls, amr_string, RENAME_NODE=False):

        """
        Parse a Pennman style string representation for amr and return an AMR

        >>>x = AMR.parse_string("(a / and :op1(恶化 :ARG0(它) :ARG1(模式 :msod(开发)) :time (已 经)) :op2(堵塞 :ARG0(它) :ARG1(交通 :mod(局部)) :location(a / around :op1(出口)))))")
        >>>
        .
        """

        def make_compiled_regex(rules):
            regexstr = '|'.join('(?P<%s>%s)' % (name, rule) for name, rule in rules)
            return re.compile(regexstr)

        def rename_node(parentnodelabel, parentconcept):
            if not isinstance(parentnodelabel, (Quantity, Polarity, Interrogative, StrLiteral)):
                # graph node rebuild
                if parentconcept is not None:
                    match = alignment_re.match(parentconcept)
                    if match:
                        amr.node_to_concepts[node_idx] = match.group(1)
                    else:
                        amr.node_to_concepts[node_idx] = parentconcept
                    mapping_table[parentnodelabel] = node_idx
                    parentnodelabel = node_idx
                    node_idx += 1
                else:
                    # not revisiting and concept is None
                    if parentnodelabel not in mapping_table:
                        match = alignment_re.mathc(parentnodelabel)
                        if match:
                            amr.node_to_concepts[node_idx] = match.group(1)
                        else:
                            amr.node_to_concepts[node_idx] = parentnodelabel
                        parentnodelabel = node_idx
                        node_idx += 1
                    else:  # revisiting
                        parentnodelabel = mapping_table[parentnodelabel]

        PNODE = 1
        CNODE = 2
        EDGE = 3
        RCNODE = 4

        amr = cls()
        stack = []
        state = 0
        node_idx = 0;  # sequential new node index
        mapping_table = {};  # old new index mapping table

        lex_rules = [
            ("LPAR", '\('),
            ("RPAR", '\)'),
            ("COMMA", ','),
            ("SLASH", '/'),
            # roxanappop: add rule for alignment
            ("ALIGNMENT", "~e.([0-9]+)((?:,(?:[0-9]+))*)"),
            ("EDGELABEL", ":[^\s()]+?(?=[~\s])"),
            ("STRLITERAL", '("[^"]+"|\u201c[^\u201d]+\u201d)'),
            ("LITERAL", "'[^\s(),]+"),
            ("INTERROGATIVE", "\s(interrogative|imperative|expressive)(?=[~\s\)])"),
            ("QUANTITY", "[0-9][0-9Ee^+\-\.,:]*(?=[~\s\)])"),
            ("IDENTIFIER", "[^\s()]+?(?=[~\s\)])"),  # no blank within characters
            ("POLARITY", "\s(\-|\+)(?=[~\s\)])")
        ]

        token_re = make_compiled_regex(lex_rules)

        # lexer = Lexer(lex_rules)
        # amr.reentrance_triples = []

        # modified the code so that the last element in the tuple pushed on stack is the list of aligned tokens
        # and that an edge node has assocaited a 4 element tuple (3rd element always 0)

        for match in token_re.finditer(amr_string):
            token = match.group()
            type = match.lastgroup

            if state == 0:
                if type == "LPAR":
                    state = 1
                else:
                    raise ParserError("Unexpected token %s" % (token))

            elif state == 1:
                if type == "IDENTIFIER":
                    stack.append((PNODE, token.strip(), None, []))
                    state = 2
                elif type == "QUANTITY":
                    stack.append((PNODE, Quantity(token.strip()), None, []))
                    state = 2
                elif type == "STRLITERAL":
                    stack.append((PNODE, StrLiteral(token.strip()), None, []))
                    state = 2
                else:
                    raise ParserError("Unexpected token %s" % (token.encode('utf8')))

            elif state == 2:
                if type == "SLASH":
                    state = 3
                elif type == "EDGELABEL":
                    stack.append((EDGE, token[1:], None, []))
                    state = 5
                elif type == "RPAR":
                    forgetme, parentnodelabel, parentconcept, aligned_tokens = AMR.popFromStack(stack)
                    assert forgetme == PNODE
                    assert parentconcept == None

                    if RENAME_NODE:
                        rename_node(parentnodelabel, parentconcept)
                    else:
                        if not parentnodelabel in amr.node_to_concepts or parentconcept is not None:
                            amr.node_to_concepts[parentnodelabel] = parentconcept

                    foo = amr[parentnodelabel]

                    if stack:
                        stack.append((CNODE, parentnodelabel, parentconcept, []))
                        state = 6
                    else:
                        amr.roots.append(parentnodelabel)
                        amr._align_root(parentnodelabel, aligned_tokens)
                        state = 0

                else:
                    raise ParserError("Unexpected token %s" % (token))

            elif state == 3:
                if type == "IDENTIFIER" or "QUANTITY":
                    assert stack[-1][0] == PNODE
                    nodelabel = AMR.popFromStack(stack)[1]
                    stack.append((PNODE, nodelabel, token, []))
                    state = 4
                else:
                    raise ParserError("Unexpected token %s" % (token))

            elif state == 4:
                if type == "ALIGNMENT":
                    amr._push_alignment_on_stack(stack, token)
                    state = 4
                elif type == "EDGELABEL":
                    stack.append((EDGE, token[1:], None, []))
                    state = 5
                elif type == "RPAR":
                    forgetme, parentnodelabel, parentconcept, aligned_tokens = AMR.popFromStack(stack)
                    assert forgetme == PNODE
                    foo = amr[parentnodelabel]  # add only the node
                    # print state,parentnodelabel,parentconcept
                    if parentconcept is not None:
                        amr.node_to_concepts[parentnodelabel] = parentconcept

                    if stack:
                        stack.append((CNODE, parentnodelabel, parentconcept, aligned_tokens))
                        state = 6
                    else:
                        amr.roots.append(parentnodelabel)
                        amr._align_root(parentnodelabel, aligned_tokens)
                        state = 0
                else:
                    raise ParserError("Unexpected token %s" % (token.encode('utf8')))

            elif state == 5:
                if type == "ALIGNMENT":
                    amr._push_alignment_on_stack(stack, token)
                    state = 5
                elif type == "LPAR":
                    state = 1
                elif type == "QUANTITY":
                    stack.append((CNODE, Quantity(token), None, []))
                    state = 6
                elif type == "STRLITERAL":
                    stack.append((CNODE, StrLiteral(token[1:-1]), None, []))
                    state = 6
                elif type == "INTERROGATIVE":
                    stack.append((CNODE, Interrogative(token[1:]), None, []))
                    state = 6
                elif type == "POLARITY":
                    stack.append((CNODE, Polarity(token.strip()), None, []))
                    state = 6
                elif type == "IDENTIFIER":
                    stack.append((RCNODE, token, None, []))
                    state = 6
                elif type == "EDGELABEL":  # Unary edge
                    stack.append((CNODE, None, None, []))
                    stack.append((EDGE, token[1:], None, []))
                    state = 5

                elif type == "RPAR":
                    print("Error: should not have RPAR after EDGELABEL")
                    raise ParserError("Unexpected token %s" % (token.encode('utf8')))

            elif state == 6:
                if type == "ALIGNMENT":
                    amr._push_alignment_on_stack(stack, token)
                    state = 6
                elif type == "RPAR":

                    edges = []
                    reedges = []
                    while stack[-1][0] != PNODE:
                        children = []
                        reentrances = []
                        # one edge may have multiple children/tail nodes
                        while stack[-1][0] == CNODE or stack[-1][0] == RCNODE:
                            CTYPE, childnodelabel, childconcept, child_aligned_tokens = AMR.popFromStack(stack)
                            if CTYPE == RCNODE:
                                reentrances.append((childnodelabel, childconcept))
                            children.append((childnodelabel, childconcept, child_aligned_tokens))

                        assert stack[-1][0] == EDGE
                        forgetme, edgelabel, none, edge_aligned_tokens = AMR.popFromStack(stack)
                        edges.append((edgelabel, children, edge_aligned_tokens))
                        reedges.append((edgelabel, reentrances))
                        # TODO: check for token

                    forgetme, parentnodelabel, parentconcept, parent_aligned_tokens = AMR.popFromStack(stack)

                    # check for annotation error
                    if parentnodelabel in list(amr.node_to_concepts.keys()):
                        # concept has been defined by the children,
                        # then they must have different concepts, otherwise the children's concepts should be None
                        # (coreference)
                        if amr.node_to_concepts[parentnodelabel] == parentconcept:
                            sys.stderr.write(
                                "Wrong annotation format: Revisited concepts %s should be ignored.\n" % parentconcept)
                        else:
                            sys.stderr.write(
                                "Wrong annotation format: Different concepts %s and %s have same node label(index)\n" % (
                                    amr.node_to_concepts[parentnodelabel], parentconcept))
                            parentnodelabel = parentnodelabel + "1"

                    if RENAME_NODE:
                        rename_node(parentnodelabel, parentconcept)
                    else:
                        if parentnodelabel not in amr.node_to_concepts and parentconcept is not None:
                            amr.node_to_concepts[parentnodelabel] = parentconcept

                    for edgelabel, children, edge_aligned_tokens in reversed(edges):
                        hypertarget = []
                        for node, concept, aligned_tokens in children:
                            if node is not None and not isinstance(node, (
                                    Quantity, Polarity, Interrogative,
                                    StrLiteral)) and not node in amr.node_to_concepts:
                                if RENAME_NODE:
                                    rename_node(node, concept)
                                else:
                                    if concept:
                                        amr.node_to_concepts[node] = concept
                            hypertarget.append(node)
                            # add alignemnt info to node_to_tokens
                            # for nodes that don't have a unique variable, will insert (token, parent) elements
                            # => need parent
                            if isinstance(node, (Quantity, Polarity, Interrogative, StrLiteral)):
                                amr._align_non_var_node(node, parentnodelabel, aligned_tokens)
                            else:
                                amr._align_var_node(node, aligned_tokens)
                        # align edge
                        amr._align_edge(edgelabel, parentnodelabel, edge_aligned_tokens)

                        hyperchild = tuple(hypertarget)
                        amr._add_triple(parentnodelabel, edgelabel, hyperchild)

                    for edgelabel, reentrance in reedges:
                        hreent = []
                        for node, concept in reentrance:
                            hreent.append(node)
                        amr._add_reentrance(parentnodelabel, edgelabel, hreent)

                    if stack:  # we have done with current level
                        state = 6
                        stack.append((CNODE, parentnodelabel, parentconcept, parent_aligned_tokens))
                    else:  # we have done with this subgraph
                        amr._align_root(parentnodelabel, parent_aligned_tokens)
                        amr.roots.append(parentnodelabel)
                        state = 0
                elif type == "COMMA":  # to seperate multiple children/tails
                    state = 7
                elif type == "EDGELABEL":
                    stack.append((EDGE, token[1:], None, []))
                    state = 5
                else:
                    raise ParserError("Unexpected token %s" % (token.encode('utf8')))

            elif state == 7:
                if type == "IDENTIFIER":
                    stack.append((CNODE, token, None, []))  # another children
                    state = 6
                elif type == "LPAR":
                    state = 1
                else:
                    raise ParserError("Unexpected token %s" % (token))

        if state != 0 and stack:
            raise ParserError("mismatched parenthesis")

        return amr

    @classmethod
    def popFromStack(cls, stack):
        pop = stack.pop()
        return pop

    @classmethod
    def peekFromStack(cls, stack):
        node = stack[-1]
        return node

    def get_variable(self, posID):
        """return variable given postition ID"""
        reent_var = None
        seq = self.dfs()[0]
        for node in seq:
            if node.seqID == posID:
                return node.node_label
        return None

    # Anda's lovely method :)))
    def get_seqID(self):
        node_seqID = ""
        node_seqIDList = list()
        """return variable given postition ID"""
        reent_var = None
        seq = self.dfs()[0]
        # print(seq)
        for node in seq:
            node_seqID = node.seqID
            node_seqIDList.append(node_seqID)
        return node_seqIDList

    def get_match(self, subgraph):
        """find the subgraph"""

        def is_match(dict1, dict2):
            rel_concept_pairs = []
            for rel, cpt in list(dict2.items()):
                rel_concept_pairs.append(rel + '@' + cpt)
                if not (rel in dict1 and cpt in dict1[rel]):
                    return None
            return rel_concept_pairs

        subroot = list(subgraph.keys())[0]  # sub root's concept
        concepts_on_the_path = []

        for v in self.node_to_concepts:
            if v[0] == subroot[0] and self.node_to_concepts[v] == subroot:
                concepts_on_the_path = [subroot]
                rcp = is_match(self[v], subgraph[subroot])
                if rcp is not None: return v, concepts_on_the_path + rcp
                # for rel, cpt in subgraph[subroot].items():
                #    if rel in self[v] and cpt in self[v][rel]:
                #        concepts_on_the_path.append(rel+'@'+cpt)
        return None, None

    def get_pid(self, var):
        seq = self.dfs()[0]
        for node in seq:
            if node.node_label == var:
                return node.seqID
        return None
        '''
        posn_queue = posID.split('.')
        var_list = self.roots
        past_pos_id = []
        while posn_queue:
            posn = int(posn_queue.pop(0))
            past_pos_id.append(posn)
            print var_list,past_pos_id,posn,visited_var
            variable = var_list[posn]
            var_list = []
            vars = [v[0] for v in self[variable].values()]
            i = 0
            while i < len(vars):
                k = vars[i]
                if k not in visited_var:
                    var_list.append(k)
                elif isinstance(k,(StrLiteral,Quantity)):
                    var_list.append(k)
                else:
                    if visited_var[k] == '.'.join(str(j) for j in past_pos_id+[i]):
                        var_list.append(k)
                    else:
                        vars.pop(i)
                        i -= 1

                i += 1

        '''
        return variable

    def get_ref_graph(self, alignment):
        """return the gold dependency graph based on amr graph"""
        dpg = DepGraph()
        for h in self:
            hstr = self.node_to_concepts[h] if h in self.node_to_concepts else h
            hidx = alignment[h][0]
            if not hidx in list(dpg.nodes.keys()):
                h_node = DNode(hidx, hstr)
                dpg.addNode(h_node)

            for ds in list(self[h].values()):
                d = ds[0]
                dstr = self.node_to_concepts[d] if d in self.node_to_concepts else d
                didx = alignment[d][0]
                if not didx in list(dpg.nodes.keys()):
                    d_node = DNode(didx, dstr)
                    dpg.addNode(d_node)
                dpg.addEdge(hidx, didx)
        # root
        root = DNode(0, 'ROOT')
        dpg.addNode(root)
        dpg.addEdge(0, alignment[self.roots[0]][0])
        return dpg

    '''
    def get_unlabel_arcs(self):
        arc_set = set()
        for h in self:
            for d in self[h].values():
                arc_set.add((h,d[0]))
        return arc_set
    '''

    def _add_reentrance(self, parent, relation, reentrance):
        if reentrance:
            self.reentrance_triples.append((parent, relation, reentrance[0]))

    def _get_aligned_tokens_list(self, token):
        alignment_rule = "~e.([0-9]+)((?:,(?:[0-9]+))*)"
        a_re = re.compile(alignment_rule)
        m = a_re.match(token)
        l = [m.group(1)]
        if m.group(2) is not None:
            extra_tokens = m.group(2).split(',')
            for i in range(1, len(extra_tokens)):
                l.append(extra_tokens[i])
        return l

    # add alignment info to the last node on the stack
    def _push_alignment_on_stack(self, stack, token):
        aligned_tokens = self._get_aligned_tokens_list(token)
        CTYPE, label, concept, old_aligned_tokens = self.popFromStack(stack)
        new_aligned_tokens = old_aligned_tokens + aligned_tokens
        stack.append((CTYPE, label, concept, new_aligned_tokens))

    # this function aligns to a node of type variable/concept
    # a list of tokens
    def _align_var_node(self, node, aligned_tokens):
        if aligned_tokens:
            if node not in list(self.node_to_tokens.keys()):
                self.node_to_tokens[node] = []
            for t in aligned_tokens:
                self.node_to_tokens[node].append(t)

    # this function aligns to a node of type strliteral/interogative/imperative/expressive
    # a list of (token, parent) tuples
    def _align_non_var_node(self, childnode, parentnode, aligned_tokens):
        if aligned_tokens:
            if childnode not in list(self.node_to_tokens.keys()):
                self.node_to_tokens[childnode] = []
            for t in aligned_tokens:
                self.node_to_tokens[childnode].append((t, parentnode))

    def _align_edge(self, edge, parentnode, aligned_tokens):
        if aligned_tokens:
            if edge not in list(self.relation_to_tokens.keys()):
                self.relation_to_tokens[edge] = []
            for t in aligned_tokens:
                self.relation_to_tokens[edge].append((t, parentnode))

    def _align_root(self, root_node, aligned_tokens):
        if isinstance(root_node, (Quantity, Polarity, Interrogative, StrLiteral)):
            self._align_non_var_node(root_node, None, aligned_tokens)
        else:
            self._align_var_node(root_node, aligned_tokens)

    def _add_triple(self, parent, relation, child, warn=None):
        """
        Add a (parent, relation, child) triple to the DAG.
        """
        if type(child) is not tuple:
            child = (child,)
        if parent in child:
            # raise Exception('self edge!')
            # sys.stderr.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            if warn: warn.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            # raise ValueError, "Cannot add self-edge (%s, %s, %s)." % (parent, relation, child)
        for c in child:
            x = self[c]
            for rel, test in list(self[c].items()):
                if parent in test:
                    if warn:
                        warn.write("WARNING: (%s, %s, %s) produces a cycle with (%s, %s, %s)\n" % (
                            parent, relation, child, c, rel, test))
                        # ATTENTION:maybe wrong, test may not have only one element, deal with it later
                        concept1 = self.node_to_concepts[parent]
                        concept2 = self.node_to_concepts[test[0]]
                        # print concept1,concept2
                        if concept1 != concept2:
                            warn.write("ANNOTATION ERROR: concepts %s and %s have same node label %s!" % (
                                concept1, concept2, parent))

                            # raise ValueError,"(%s, %s, %s) would produce a cycle with (%s, %s, %s)" % (parent, relation, child, c, rel, test)

        self[parent].append(relation, child)

    def set_alignment(self, alignment):
        self.align_to_sentence = alignment

    def bfs(self):
        """
        breadth first search for the graph
        return the bfs-ordered triples
        """
        from collections import deque
        visited_nodes = set()
        amr_triples = []
        sequence = []
        nid = 0

        for i, r in enumerate(self.roots):
            seqID = str(i)
            queue = deque([((r,), None, 0, seqID)])  # node, incoming edge and depth
            amr_triples.append(('root', 'ROOT', r))
            while queue:
                next, rel, depth, seqID = queue.popleft()
                for n in next:
                    firsthit = (parent, rel, n) not in self.reentrance_triples
                    leaf = False if self[n] else True

                    node = Node(rel, n, firsthit, leaf, depth, seqID)
                    # nid += 1
                    sequence.append(node)
                    if n in visited_nodes or (parent, rel, n) in self.reentrance_triples:
                        continue
                    visited_nodes.add(n)
                    p = len([child for rel, child in list(self[n].items()) if
                             (n, rel, child[0]) not in self.reentrance_triples]) - 1
                    for rel, child in reversed(list(self[n].items())):
                        if not (rel, n, child[0]) in amr_triples:
                            if (n, rel, child[0]) not in self.reentrance_triples:
                                queue.append((child, rel, depth + 1, seqID + '.' + str(p)))
                                p -= 1
                            else:
                                queue.append((child, rel, depth + 1, None))
                            amr_triples.append((rel, n, child[0]))

        return sequence, amr_triples

    def print_triples(self):
        result = ''
        amr_triples = self.bfs()[1]
        for rel, parent, child in amr_triples:
            if not isinstance(child, (Quantity, Polarity, Interrogative, StrLiteral)):
                result += "%s(%s,%s)\n" % (rel, self.node_to_concepts[parent], self.node_to_concepts[child])
            else:
                result += "%s(%s,%s)\n" % (rel, self.node_to_concepts[parent], child)
        return result

    def dfs(self):
        """
        depth first search for the graph
        return dfs ordered nodes and edges
        TO-DO: this visiting order information can be obtained
        through the reading order of amr strings; modify the class
        to OrderedDefaultDict;
        """
        visited_nodes = set()
        visited_edges = []
        sequence = []

        for i, r in enumerate(self.roots):
            seqID = str(i)
            stack = [
                ((r,), None, None, 0, seqID)]  # record the node, incoming edge, parent, depth and unique identifier

            # all_nodes = []
            while stack:
                next, rel, parent, depth, seqID = AMR.popFromStack(stack)
                for n in next:
                    if self.reentrance_triples:
                        firsthit = (parent, rel, n) not in self.reentrance_triples
                    else:
                        firsthit = n not in visited_nodes
                    leaf = False if self[n] else True

                    node = Node(parent, rel, n, firsthit, leaf, depth, seqID)

                    # print self.node_to_concepts
                    sequence.append(node)

                    # same StrLiteral/Quantity/Polarity should not be revisited
                    if self.reentrance_triples:  # for being the same with the amr string readed in
                        if n in visited_nodes or (parent, rel, n) in self.reentrance_triples:
                            continue
                    else:
                        if n in visited_nodes:
                            continue

                    visited_nodes.add(n)
                    p = len([child for rel, child in list(self[n].items()) if
                             (n, rel, child[0]) not in self.reentrance_triples]) - 1
                    for rel, child in reversed(list(self[n].items())):
                        # print rel,child
                        if not (rel, n, child[0]) in visited_edges:
                            # if child[0] not in visited_nodes or isinstance(child[0],(StrLiteral,Quantity)):
                            visited_edges.append((rel, n, child[0]))
                            if (n, rel, child[0]) not in self.reentrance_triples:
                                stack.append((child, rel, n, depth + 1, seqID + '.' + str(p)))
                                p -= 1
                            else:
                                stack.append((child, rel, n, depth + 1, None))
                        elif isinstance(child[0], (StrLiteral, Quantity)):
                            stack.append((child, rel, n, depth + 1, seqID + '.' + str(p)))
                            p -= 1
                        else:
                            pass

            return (sequence, visited_edges)

    def replace_node(self, h_idx, idx):
        """for coreference, replace all occurrence of node idx to h_idx"""
        visited_nodes = set()
        visited_edges = set()

        for i, r in enumerate(self.roots[:]):
            stack = [((r,), None, None)]  # node,incoming edge and preceding node

            while stack:
                next, rel, previous = AMR.popFromStack(stack)
                for n in next:
                    if n == idx:
                        if previous == None:  # replace root
                            self.roots[i] = h_idx
                            break
                        self[previous].replace(rel, (h_idx,))
                    if n in visited_nodes:
                        continue
                    visited_nodes.add(n)
                    for rel, child in reversed(list(self[n].items())):
                        if not (n, rel, child) in visited_edges:
                            if child in visited_nodes:
                                stack.append((child, rel, n))
                            else:
                                visited_edges.add((n, rel, child))
                                stack.append((child, rel, n))

    def find_rel(self, h_idx, idx):
        """find the relation between head_idx and idx"""
        rels = []
        for rel, child in list(self[h_idx].items()):
            # print child,idx
            if child == (idx,):
                rels.append(rel)
        return rels

    def replace_head(self, old_head, new_head, KEEP_OLD=True):
        """change the focus of current sub graph"""
        for rel, child in list(self[old_head].items()):
            if child != (new_head,):
                self[new_head].append(rel, child)
        del self[old_head]
        if KEEP_OLD:
            foo = self[old_head]
            self[new_head].append('NA', (old_head,))

    def replace_rel(self, h_idx, old_rel, new_rel):
        """replace the h_idx's old_rel to new_rel"""
        for v in self[h_idx].getall(old_rel):
            self[h_idx].append(new_rel, v)
        del self[h_idx][old_rel]

    '''
    def rebuild_index(self, node, sent_index_mapping=None):
        """assign non-literal node a new unique node label; replace the
           original index with the new node id or sentence offset;
           if we have been provided the sentence index mapping, we use the
           sentence offsets as new node label instead of the serialized node id.
        """
        if sent_index_mapping is None:
            if node.node_label in self.node_to_concepts and self.node_to_concepts[node.node_label] is not None:
                #update the node_to_concepts table
                self.node_to_concepts[Node.node_id] = self.node_to_concepts[node.node_label]
                del self.node_to_concepts[node.node_label]
                Node.mapping_table[node.node_label] = Node.node_id
                node.node_label = Node.node_id

            elif self.node_label not in node_to_concepts and self.node_label in Node.mapping_table:
                new_label = Node.mapping_table[self.node_label]
                self.node_label = new_label
            else:
                #print Node.node_id,self.node_label
                node_to_concepts[Node.node_id] = self.node_label
                self.node_label = Node.node_id

    '''

    def to_amr_string(self):

        amr_string = ""

        seq = self.dfs()[0]

        # always begin with root
        assert seq[0].trace == None
        dep_rec = 0
        for node in seq:
            if node.trace == None:
                if node.firsthit and node.node_label in self.node_to_concepts:
                    amr_string += "(%s / %s" % (node.node_label, self.node_to_concepts[node.node_label])
                else:
                    amr_string += "(%s" % (node.node_label)
            else:
                if node.depth >= dep_rec:
                    dep_rec = node.depth
                else:
                    amr_string += "%s" % ((dep_rec - node.depth) * ')')
                    dep_rec = node.depth

                if not node.leaf:
                    if node.firsthit and node.node_label in self.node_to_concepts:
                        amr_string += "\n%s:%s (%s / %s" % (
                            node.depth * "\t", node.trace, node.node_label, self.node_to_concepts[node.node_label])
                    else:
                        amr_string += "\n%s:%s %s" % (node.depth * "\t", node.trace, node.node_label)

                else:
                    if node.firsthit and node.node_label in self.node_to_concepts:
                        amr_string += "\n%s:%s (%s / %s)" % (
                            node.depth * "\t", node.trace, node.node_label, self.node_to_concepts[node.node_label])
                    else:
                        if isinstance(node.node_label, StrLiteral):
                            amr_string += '\n%s:%s "%s"' % (node.depth * "\t", node.trace, node.node_label)
                        else:
                            amr_string += "\n%s:%s %s" % (node.depth * "\t", node.trace, node.node_label)

        if dep_rec != 0:
            amr_string += "%s" % ((dep_rec) * ')')
        else:
            amr_string += ')'

        return amr_string

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) + t[3:]


import copy as copy
if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-v", action="store_true", dest="verbose")

    (options, args) = opt.parse_args()

    s = '''(a / and :op1(恶化 :ARG0(它) :ARG1(模式 :mod(开发)) :time (已经)) :op2(t / 堵塞 :ARG0(它) :ARG1(交通 :mod(局部)) :location(a / around :op1(出口))))'''
    s1 = '''(a  /  and :op1 (c  /  change-01 :ARG0 (i  /  it) :ARG1 (p  /  pattern :mod (d  /  develop-02)) :ARG2 (b  / bad :degree (m  /  more))) :op2 (c2  /  cause-01 :ARG0 i :ARG1 (c3  /  congest-01 :ARG1 (a2  /  around :op1 (e  /  exit :poss i)) :ARG2 (t  /  traffic) :ARG1-of (l2  /  localize-01))) :time (a3  /  already))'''
    # s = s.decode('utf8')
    # amr_ch = AMR.parse_string(s)
    amr_en = AMR.parse_string(s1)
    print(str(amr_en))
    c2 = copy.deepcopy(amr_en)
    print(str(c2))
    if amr_en != c2:
        print("Issue with deep copy")
    c3 = copy.deepcopy(c2)
    print(str(c3))

    # print amr_ch
    # print amr_ch.dfs()
    # print amr_ch.to_amr_string()
