from nlp_helpers import *

json_file = "datasets/movieline_chunks/movielines_378.json"
with open(json_file, 'r') as input_json:
    movielines = json.load(input_json)
    sentences = movielines["sentences"]
    replies = movielines["replies"]
    casual_json = {}
    for i in range(0, len(sentences)):
        tokenized_sent = tokenize_sentence(sentences[i])
        tokenized_reply = tokenize_sentence(sentences[i])
        casual_json[sentences[i]] = replies[i]
    casual_file = "datasets/test.json"
    json_object = json.dumps(casual_json, indent = 4) 
    with open(casual_file, "w") as f:
        f.write(json_object)