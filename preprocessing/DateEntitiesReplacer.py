from nltk.tag import StanfordNERTagger

from preprocessing import STANFORD_NER_MODEL, STANFORD_NER_JAR


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
                if tag == 'DATE':
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
                else:
                    new_sentence_list.append(token)
            location += 1
        for item in new_sentence_list:
            new_sentence = new_sentence + item + ' '

        print tagged_list
        print new_sentence
        print named_entities_location

        return new_sentence, named_entities_location

    except Exception, e:
        print str(e)


if __name__ == "__main__":
    print process_sentence('Andrea leaves on monday and returns in July in New York')
