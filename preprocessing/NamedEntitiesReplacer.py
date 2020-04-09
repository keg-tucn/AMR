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


def process_sentence(sentence):
    try:
        st = StanfordNERTagger(STANFORD_NER_MODEL, STANFORD_NER_JAR)

        tagged_list = st.tag(sentence.split())
        new_sentence = ""
        new_sentence_list = []
        named_entities_location = []
        location = 0
        for token, tag in tagged_list:
            if tag == 'O':
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

        print(tagged_list)
        print(new_sentence_list)
        print(new_sentence)
        print(named_entities_location)

        return new_sentence, named_entities_location

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    print(process_sentence('Rami Eid John is studying in San Francisco'))
