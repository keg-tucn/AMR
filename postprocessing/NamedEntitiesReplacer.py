import nltk
import re
import time

contentArray = ['Starbucks is not doing very well lately .',
                'Overall, while it may seem there is already a Starbucks on every corner, Starbucks still has a lot of room to grow because China is here.',
                'They just began expansion into food products, which has been going quite well so far for them.',
                'I can attest that my own expenditure when going to Starbucks has increased, in lieu of these food products.',
                'Starbucks is also indeed expanding their number of stores as well.',
                'Starbucks still sees strong sales growth here in the united states, and intends to actually continue increasing this.',
                'Starbucks also has one of the more successful loyalty programs, which accounts for 30%  of all transactions being loyalty-program-based.',
                'As if news could not get any more positive for the company, Brazilian weather has become ideal for producing coffee beans.',
                'Brazil is the world\'s #1 coffee producer, the source of about 1/3rd of the entire world\'s supply!',
                'Given the dry weather, coffee farmers have amped up production, to take as much of an advantage as possible with the dry weather.',
                'Increase in supply... well you know the rules...', ]


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


##let the fun begin!##
def process_language(content_array):
    try:
        for item in content_array:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            chunked = nltk.ne_chunk(tagged, binary=True)
            named_entities = extract_entity_names(chunked)
            result = ''
            tokenized_copy = tokenized
            for i, token in enumerate(tokenized_copy):
                for entity in named_entities:
                    if token == entity:
                        tokenized_copy[i] = 'NAME'
            for token in tokenized_copy:
                result = result + token + ' '
            print result

    except Exception, e:
        print str(e)


def process_sentence(sentence):
    try:
        tokenized = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized)
        chunked = nltk.ne_chunk(tagged, binary=True)
        named_entities = extract_entity_names(chunked)
        result = ''
        tokenized_copy = tokenized
        for i, token in enumerate(tokenized_copy):
            for entity in named_entities:
                if token == entity:
                    tokenized_copy[i] = 'NAME'
        for token in tokenized_copy:
            result = result + token + ' '
        return result

    except Exception, e:
        print str(e)


if __name__ == "__main__":
    print process_sentence('John Jackson likes apples .')
