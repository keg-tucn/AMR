import re


def extract_sentence_amr_pairs(file_path, extract_from_coref_datasets=False):
    with open(file_path) as f:
        lines = f.readlines()

    if extract_from_coref_datasets:
        token_regex = re.compile('^(?:# sentence: )(.*)$') # this is how I find data in my dataset which has only senteces and AMRs with corefs

        amr_start_indices = [index for index in range(0, len(lines)) if token_regex.match(lines[index])]

        triples = map(lambda i: (token_regex.match(lines[i]).group(1), get_amr(lines, i), 1),
                      amr_start_indices)

    else:
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
    id_line = lines[sentence_index - 1]
    id_regex = re.compile("^# ::id ([^ ]+) ::(.*)$")
    if id_regex.match(id_line):
        return id_regex.search(id_line).group(1)
    return id_line
