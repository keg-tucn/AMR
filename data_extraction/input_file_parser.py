import re


def extract_data_records(file_path):
    """
    Extracts tuples of sentences, string AMRs and AMR ids from an input file
    :param file_path: file to be processed
    :return: list of tuples (sentence, AMR string, AMR id)
    """
    print("Extracting data records from %s" % file_path)
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    token_regex = re.compile('^(?:# ::tok )(.*)$')
    amr_start_indices = [index for index in range(0, len(lines)) if token_regex.match(lines[index])]

    triples = [(token_regex.match(lines[i]).group(1), _get_amr(lines, i), _get_id(lines, i)) for i in amr_start_indices]
    return triples


def _get_amr(lines, sentence_index):
    amr = ""
    i = sentence_index + 2
    while i < len(lines) and len(lines[i]) > 1:
        amr += lines[i]
        i += 1
    return amr


def _get_id(lines, sentence_index):
    id_line = lines[sentence_index - 1]
    id_regex = re.compile("^# ::id ([^ ]+) ::(.*)$")
    if id_regex.match(id_line):
        return id_regex.search(id_line).group(1)
    return id_line
