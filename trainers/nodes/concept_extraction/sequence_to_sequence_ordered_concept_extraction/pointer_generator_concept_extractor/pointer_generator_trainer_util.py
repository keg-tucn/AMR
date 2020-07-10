from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import get_all_concepts
from deep_dynet import support as ds
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.trainer_util import \
    generate_verbs_nonverbs

EOS = 'EOS'
SOS = 'SOS'
UNK = 'UNK'

SGD_trainer = 'SGD'
ADAM_trainer = 'Adam'
Momentum_trainer = 'Momentum'
Cyclical_trainer = 'Cyclical'

class PointerGeneratorConceptExtractorGraphHyperparams:
    def __init__(self,
                 no_epochs,
                 max_sentence_len,
                 use_preprocessing,
                 alignment,
                 experimental_run,
                 two_classifiers: bool,
                 dropout,
                 trainer = SGD_trainer):
        self.max_sentence_len = max_sentence_len
        self.no_epochs = no_epochs
        self.use_preprocessing = use_preprocessing
        self.alignment = alignment
        self.experimental_run = experimental_run
        self.two_classifiers = two_classifiers
        self.dropout = dropout
        self.trainer = trainer

    def __str__(self):
        repr = self.alignment +\
               '_ep_'+str(self.no_epochs)+\
               '_senlen_'+str(self.max_sentence_len)+\
               '_prep_'+str(self.use_preprocessing) + \
               '_exprun_' + str(self.experimental_run) + \
               '_twoclass_' + str(self.two_classifiers) + \
               '_drop_'+str(self.dropout)
        if self.trainer != SGD_trainer:
            repr = self.trainer +'_'+ repr
        return repr

def create_vocabs_for_pointer_generator_network(train_entries):
    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    all_concepts = get_all_concepts(train_concepts)
    all_concepts.append(EOS)
    all_concepts.append(SOS)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)


    all_verbs, all_nonverbs = generate_verbs_nonverbs(all_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concepts)
    all_verbs_vocab = ds.Vocab.from_list(all_verbs)
    all_nonverbs_vocab = ds.Vocab.from_list(all_nonverbs)

    train_words = []
    for train_entry in train_entries:
        for word in train_entry.sentence_tokens:
            train_words.append(word)
    train_words.append(EOS)
    train_words.append(SOS)
    train_words.append(UNK)
    all_words = list(set(train_words))
    all_words_vocab = ds.Vocab.from_list(all_words)

    return all_concepts_vocab, all_words_vocab, all_verbs_vocab, all_nonverbs_vocab
