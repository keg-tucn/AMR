import dynet as dy
from operator import itemgetter

WORD_DIM = 64
LSTM_DIM = 64
ACTION_DIM = 32

# actions the parser can take
acts = ['SH', 'RL', 'RR', 'DN', 'SW']
SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NUM_ACTIONS = len(acts)


class TransitionParser:
    def __init__(self, model, vocab):
        self.vocab = vocab
        self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_comp = model.add_parameters((LSTM_DIM,))
        self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
        self.pb_s2h = model.add_parameters((LSTM_DIM,))
        self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
        self.pb_act = model.add_parameters((NUM_ACTIONS,))

        # layers, in-dim, out-dim, model
        self.buffRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
        self.stackRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
        self.pempty_buffer_emb = model.add_parameters((LSTM_DIM,))
        self.pempty_stack_emb = model.add_parameters((LSTM_DIM,))
        nwords = vocab.size()
        self.WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_DIM))

    # returns an expression of the loss for the sequence of actions
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, t, oracle_actions=None):
        dy.renew_cg()
        if oracle_actions:
            oracle_actions = list(oracle_actions)
            oracle_actions.reverse()
        stack_top = self.stackRNN.initial_state()
        toks = list(t)
        toks.reverse()
        stack = []
        cur = self.buffRNN.initial_state()
        buffer = []
        empty_buffer_emb = dy.parameter(self.pempty_buffer_emb)
        empty_stack_emb = dy.parameter(self.pempty_stack_emb)
        weight_comp = dy.parameter(self.pW_comp)
        bias_comp = dy.parameter(self.pb_comp)
        weight_s2h = dy.parameter(self.pW_s2h)
        bias_s2h = dy.parameter(self.pb_s2h)
        weight_act = dy.parameter(self.pW_act)
        bias_act = dy.parameter(self.pb_act)
        losses = []
        good_predictions = []
        node_index = 1;
        for tok in toks:
            tok_embedding = self.WORDS_LOOKUP[tok]
            cur = cur.add_input(tok_embedding)
            buffer.append((cur.output(), tok_embedding, self.vocab.i2w[tok]))

        while not (len(stack) == 1 and len(buffer) == 0):
            # based on parser state, get valid actions
            valid_actions = []
            if len(buffer) > 0:  # can only reduce if elements in buffer
                valid_actions += [SH]
            if len(buffer) >= 1:
                valid_actions += [DN]
            if len(stack) >= 2:  # can only shift if 2 elements on stack
                valid_actions += [RL, RR]
            if len(stack) >= 3:
                valid_actions += [SW]  # can only swap if we have at least 3 elements on the stack

            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            action = valid_actions[0]
            label = None
            log_probs = None
            if len(valid_actions) > 1:
                buffer_embedding = buffer[-1][0] if buffer else empty_buffer_emb
                stack_embedding = stack[-1][0].output() if stack else empty_stack_emb  # the stack has something here
                parser_state = dy.concatenate([buffer_embedding, stack_embedding])
                h = dy.tanh(weight_s2h * parser_state + bias_s2h)
                logits = weight_act * h + bias_act
                log_probs = dy.log_softmax(logits, valid_actions)
                predicted_action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
                if oracle_actions is None:
                    print('no oracle!')
                    action = predicted_action
            if oracle_actions is not None:
                oracle_action = oracle_actions.pop()
                action = oracle_action.index
                label = oracle_action.label
                if log_probs is not None:
                    # append the action-specific loss based on oracle
                    losses.append(dy.pick(log_probs, action))
                    good_predictions.append(1 if action == predicted_action else 0)
            # execute the action to update the parser state
            if action == SH:
                _, tok_embedding, token = buffer.pop()
                stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
                stack_state = stack_state.add_input(tok_embedding)
                stack.append((stack_state, Node(label, token, node_index)))
                node_index += 1
            elif action == DN:
                buffer.pop()
            elif action == SW:
                top = stack.pop()
                mid = stack.pop()
                lower = stack.pop()
                stack.append(mid)
                stack.append(lower)
                stack.append(top)
            else:  # one of the reduce actions
                right = stack.pop()
                left = stack.pop()
                head, modifier = (left, right) if action == RR else (right, left)
                top_stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
                head_rep, head_node = head[0].output(), head[1]
                mod_rep, mod_node = modifier[0].output(), modifier[1]
                composed_rep = dy.rectify(weight_comp * dy.concatenate([head_rep, mod_rep]) + bias_comp)
                top_stack_state = top_stack_state.add_input(composed_rep)
                head_node.add_child(mod_node, label)
                stack.append((top_stack_state, head_node))
                if oracle_actions is None:
                    print('{0} --> {1}'.format(head_node.token, mod_node.token))

        # the head of the tree that remains at the top of the stack is now the root
        head = stack.pop()[1]
        if oracle_actions is None:
            print('ROOT --> {0}'.format(head))
        # print("losses" + str(map(lambda x: x.scalar_value(), losses)))
        # print(head.preety_print())
        return -dy.esum(losses) if losses else None, head, sum(good_predictions), len(good_predictions)


class Node:
    def __init__(self, label, token, node_index):
        self.label = label
        self.token = token
        self.children = []
        self.node_index = node_index

    def add_child(self, obj, relation):
        self.children.append((obj, relation))

    def preety_print(self, depth=1, include_original=True):
        preety = "( %s" % self.label
        if include_original:
            preety += (" orig: %s" % self.token)
        preety += "".join(
            ("\n".ljust(depth + 1, "\t") + "%s  %s" % (relation, child.preety_print(depth=depth + 1, include_original=include_original))) for (child, relation) in self.children)
        if self.children:
            preety += "\n".ljust(depth, "\t")
        preety += ")"
        return preety

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
