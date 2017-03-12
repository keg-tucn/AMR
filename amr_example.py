from __future__ import print_function
import dynet as dy
import deep_dynet.support as ds
import deep_dynet.transition_parser as tp
import logging
from os import listdir
import TrainingDataExtractor as tde


def read_data(type):
    data = []
    mypath = 'resources/alignments/split/' + type
    print(mypath)
    for f in listdir(mypath):
        mypath_f = mypath + "/" + f
        print(mypath_f)
        data += tde.generate_training_data(mypath_f, False)
    return data


def process_data(data, vocab_words, vocab_acts):
    for d in data:
        sentence = d[0]
        actions = d[1]
        yield (
            ds.word_sentence_to_vocab_index(sentence.split(), vocab_words),
            ds.oracle_actions_to_action_index(actions, vocab_acts),
            sentence,
            actions,
            d[2]
        )


training_data = read_data("training")
# dev_data = read_data("dev")
test_data = read_data("test")
print("Training size %d" % len(training_data))
print("Test size %d" % len(test_data))

vocab_acts = ds.Vocab.from_list(tp.acts)
vocab_words = ds.Vocab.from_file('resources/data/vocab.txt')
# train = list(ds.read_oracle('resources/data/amr-examples.txt', vocab_words, vocab_acts))
# dev = list(ds.read_oracle('resources/data/amr-examples-test.txt', vocab_words, vocab_acts))
train = list(process_data(training_data, vocab_words, vocab_acts))
dev = list(process_data(test_data, vocab_words, vocab_acts))

model = dy.Model()
trainer = dy.AdamTrainer(model)

tp = tp.TransitionParser(model, vocab_words)

log_errors_on_train = False
if(log_errors_on_train):
    logging.disable(logging.NOTSET)
# cmake .. -DEIGEN3_INCLUDE_DIR=/Users/flo/Documents/Doctorat/AMR/dynet-base/eigen -DBOOST_ROOT=/usr/local/opt/boost160/ -DPYTHON=/usr/bin/python
min_loss = 100
rounds = 0
min_epoch = 0
fail_sentences = []
for epoch in range(100):
    for (sentence, actions, original_sentence, original_actions, amr) in train:
        loss = None
        try:
            loss = tp.parse(sentence, actions)[0]
        except Exception as e:
            logging.debug(e)
            fail_sentences.append(original_sentence)
            logging.warn("%s\n with actions %s\n", original_sentence, original_actions)
        if loss is not None:
            # for some weird reason backward throws an failed assertion if there is no scalar value retrievall
            loss.scalar_value()
            loss.backward()
            trainer.update()
    print("Failed sentences: %d" % len(fail_sentences))
    dev_words = 0
    dev_loss = 0.0
    fail_sentences = []
    for (ds, da, original_sentence, original_actions, amr) in dev:
        loss = None
        try:
            loss = tp.parse(ds, da)[0]
            dev_words += len(ds)
        except Exception as e:
            logging.debug(e)
            fail_sentences.append(original_sentence)
            logging.warn("%s\n with actions %s\n", original_sentence, original_actions)
        if loss is not None:
            dev_loss += loss.scalar_value()
    print("Failed sentencs in test: %d" % len(fail_sentences))
    loss_dev_words = dev_loss / dev_words
    print('[validation] epoch {}: per-word loss: {}'.format(epoch, loss_dev_words))
    min_loss = min(min_loss, loss_dev_words)
    if min_loss == loss_dev_words:
        rounds = 0
        min_epoch = epoch
    else:
        rounds += 1
    print("since {} min loss {} for {} rounds.".format(min_epoch, min_loss, rounds))
