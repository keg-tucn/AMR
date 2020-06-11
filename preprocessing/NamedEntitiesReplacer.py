import functools

import nltk
from nltk.tag import StanfordNERTagger

from definitions import STANFORD_NER_MODEL, STANFORD_NER_JAR


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


def process_language(content_array):
    try:
        for item in content_array:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            chunked = nltk.ne_chunk(tagged, binary=True)
            named_entities = extract_entity_names(chunked)
            print(named_entities)
            result = ''
            tokenized_copy = tokenized
            for i, token in enumerate(tokenized_copy):
                for entity in named_entities:
                    if token == entity:
                        tokenized_copy[i] = 'NAME'
            for token in tokenized_copy:
                result = result + token + ' '

    except Exception as e:
        print(str(e))


@functools.lru_cache(maxsize=5)
def get_stanford_ner_tagger():
    return StanfordNERTagger(STANFORD_NER_MODEL, STANFORD_NER_JAR)


def process_sentence(sentence,
                     tags_to_identify=['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'PERCENT', 'DATE', 'TIME']):
    try:
        st = get_stanford_ner_tagger()

        tagged_list = st.tag(sentence.split())
        new_sentence = ""
        new_sentence_list = []
        named_entities_location = []
        location = 0
        for token, tag in tagged_list:
            if tag == 'O' or tag not in tags_to_identify:
                new_sentence_list.append(token)
            else:
                if len(new_sentence_list) == 0:
                    new_sentence_list.append(tag)
                    named_entities_location.append((location, [token]))
                else:
                    if new_sentence_list[len(new_sentence_list) - 1] != tag:
                        new_sentence_list.append(tag)
                        named_entities_location.append((location, [token]))
                    else:
                        location -= 1
                        element = named_entities_location.pop(-1)
                        element[1].append(token)
                        named_entities_location.append(element)

            location += 1
        for item in new_sentence_list:
            new_sentence = new_sentence + item + ' '

        return new_sentence, named_entities_location

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    print(process_sentence('Rami Eid John is studying in San Francisco'))
    print(process_sentence('Rami Eid John is studying in San Francisco in October 2020'))
    print(process_sentence('Establish an innovation fund with a maximum amount of 1,000 U.S. dollars .'))
    print(process_sentence('On Tuesday , Wanke announced that it would make an additional donation of RMB 100 million yuan for the temporary resettlement and post @-@ disaster reconstruction in the next three to five years .'))
    print(process_sentence('so a lot of rich people donated their money to the Church before they died .'))
    print(process_sentence('By the 10 th Century AD , 60 % of Britain \'s territory belonged to the Church'))
    print(process_sentence('I headed straight for the center of activities , but in actual fact traffic was being controlled as early as 4 o\'clock , and they had already started limiting the crowds entering the sports center .'))
    print(process_sentence('Sometime after 6 I went straight back home to watch it on TV . The spectacle was just as astonishing on TV , except that it was just one angle of the whole thing .'))
    print(process_sentence('I think it is not those elites , but that some corrupt officials were bribed'))
    print(process_sentence('On the day of the Tangshan Earthquake , i.e. July 28 th , those on duty at Mao Zedong \'s quarters ' \
               'were Wang Dongxing , Wang Hongwen , and Mao Zedong \'s confidential secretary , Zhang Yufeng . '))