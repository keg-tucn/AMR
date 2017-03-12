from __future__ import print_function
import dynet as dy
import deep_dynet.support as ds
import deep_dynet.transition_parser as tp

vocab_acts = ds.Vocab.from_list(tp.acts)
vocab_words = ds.Vocab.from_file('resources/data/vocab.txt')
train = list(ds.read_oracle('resources/data/amr-examples.txt', vocab_words, vocab_acts))
dev = list(ds.read_oracle('resources/data/amr-examples-test.txt', vocab_words, vocab_acts))

model = dy.Model()
trainer = dy.AdamTrainer(model)

tp = tp.TransitionParser(model, vocab_words)

# cmake .. -DEIGEN3_INCLUDE_DIR=/Users/flo/Documents/Doctorat/AMR/dynet-base/eigen -DBOOST_ROOT=/usr/local/opt/boost160/ -DPYTHON=/usr/bin/python
min_loss = 100
rounds = 0
min_epoch = 0
for epoch in range(100):
    for (sentence, actions) in train:
        loss = tp.parse(sentence, actions)[0]
        if loss is not None:
            # for some weird reason backward throws an failed assertion if there is no scalar value retrievall
            loss.scalar_value()
            loss.backward()
            trainer.update()
    dev_words = 0
    dev_loss = 0.0
    for (ds, da) in dev:
        loss = tp.parse(ds, da)[0]
        dev_words += len(ds)
        if loss is not None:
            dev_loss += loss.scalar_value()
    loss_dev_words = dev_loss / dev_words
    print('[validation] epoch {}: per-word loss: {}'.format(epoch, loss_dev_words))
    min_loss = min(min_loss, loss_dev_words)
    if min_loss == loss_dev_words:
        rounds = 0
        min_epoch = epoch
    else:
        rounds += 1
    print("since {} min loss {} for {} rounds.".format(min_epoch, min_loss, rounds))
