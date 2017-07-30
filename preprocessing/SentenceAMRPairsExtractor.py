import re


def extract_sentence_amr_pairs(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    token_regex = re.compile('^(?:# ::tok )(.*)$')
    amr_start_indices = [index for index in range(0, len(lines)) if token_regex.match(lines[index])]

    triples = map(lambda i: (token_regex.match(lines[i]).group(1), get_amr(lines, i), get_id(lines, i)), amr_start_indices)
    return triples


def get_amr(lines, sentence_index):
    amr = ""
    i = sentence_index + 2
    while i < len(lines) and len(lines[i]) > 1:
        amr += lines[i]
        i += 1
    return amr


def get_id(lines, sentence_index):
    return lines[sentence_index - 1]
