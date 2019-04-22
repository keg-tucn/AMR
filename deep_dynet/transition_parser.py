from operator import itemgetter

import dynet as dy

from models.node import Node
import models.actions as act
import logging

WORD_DIM = 64
LSTM_DIM = 64
ACTION_DIM = 32

# actions the parser can take
SH = 0
RL = 1
RR = 2
DN = 3
SW = 4
NUM_ACTIONS = len(act.acts)


# TODO: think of training  a model for each action and have an ensamble decide the next one ?


def conv_action(action):
    return act.acts[action]


def conv_actions(actions):
    return map(lambda x: conv_action(x), actions)


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

    def convert_token(self, tok):
        return self.vocab.i2w[tok]

    def preety_tokens(self, tokens):
        return map(lambda t: self.convert_token(t), tokens)

    # returns an expression of the loss for the sequence of actions
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def parse(self, tokens, oracle_actions=None, concepts_metadata=None, use_model_predictions=False):
        logging.debug("Parsing with model predictions %s: %s with oracle %s concepts_metadata %s",
                      use_model_predictions, self.preety_tokens(tokens), oracle_actions, concepts_metadata)

        dy.renew_cg()

        if oracle_actions:
            oracle_actions = list(oracle_actions)
            oracle_actions.reverse()

        stack_top = self.stackRNN.initial_state()
        tokens = list(tokens)
        tokens.reverse()
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
        predicted_actions = []
        invalid_actions = 0

        for tok in tokens:
            tok_embedding = self.WORDS_LOOKUP[tok]
            cur = cur.add_input(tok_embedding)
            buffer.append((cur.output(), tok_embedding, self.convert_token(tok)))

        while not (len(stack) <= 1 and len(buffer) == 0):

            # based on parser state, get valid actions
            valid_actions = []
            # can only reduce & delete if elements in buffer
            if len(buffer) > 0:
                valid_actions += [SH, DN]
            # can only shift if 2 elements on stack
            if len(stack) >= 2:
                valid_actions += [RL, RR]
            # can only swap if we have at least 3 elements on the stack
            # and the previous action is not also a swap
            if len(stack) >= 3 and (predicted_actions and predicted_actions[-1] != SW):
                valid_actions += [SW]

            logging.info("valid actions %s", conv_actions(valid_actions))
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            if not valid_actions:
                logging.warn("stack" + str(len(stack)) + "buffer" + str(len(buffer)))

            action = valid_actions[0]
            predicted_action = action
            label = None
            concept_key = None
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
                    logging.warn("no oracle! using predicted action %s", predicted_action)
                    action = predicted_action

            if oracle_actions is not None:
                if oracle_actions:
                    oracle_action = oracle_actions.pop()
                    action = oracle_action.index
                    label = oracle_action.label
                    concept_key = oracle_action.key
                    if log_probs is not None:
                        # append the action-specific loss based on oracle
                        pick = dy.pick(log_probs, action)
                        losses.append(pick)
                        # TODO(flo): check if we pick the losses correctly w.r.t the predicted action
                        good_predictions.append(1 if action == predicted_action else 0)
                        logging.debug("Predicted %s vs Oracle %s", conv_action(predicted_action), conv_action(action))
                else:
                    # consumed all oracle actions and elements still in buffer
                    logging.warn("invalid action - outside oracle. Predicted %s", predicted_action)
                    invalid_actions += 1
                    action = -1
                    label = "UNKNOWN"
                    concept_key = "UNNOWN"
                    if log_probs is not None:
                        # TODO(flo): add some huge loss
                        losses.append(losses[-1])
                        good_predictions.append(0)
            if use_model_predictions:
                action = predicted_action
                predicted_actions.append(predicted_action)
                logging.debug("predicted %s", conv_action(predicted_action))
            logging.debug("applying action %s", conv_action(action))
            # execute the action to update the parser state
            if action == SH:
                _, tok_embedding, token = buffer.pop()
                stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
                stack_state = stack_state.add_input(tok_embedding)
                node = Node(label)
                if concept_key in concepts_metadata:
                    node = concepts_metadata[concept_key]
                stack.append((stack_state, node))
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
        head = stack.pop()[1] if stack else Node("unknown")
        if oracle_actions is None:
            logging.info('ROOT --> {0}'.format(head))
        # print("losses" + str(map(lambda x: x.scalar_value(), losses)))
        # print(head.preety_print())
        return -dy.esum(losses) if losses else None, head, sum(good_predictions), len(
            good_predictions), predicted_actions, invalid_actions
