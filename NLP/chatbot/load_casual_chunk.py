from nlp_helpers import *


json_file = "datasets/movieline_chunks/movielines_378.json"
with open(json_file, 'r') as input_json:
    movielines = json.load(input_json)
    sentences = movielines["sentences"]
    replies = movielines["replies"]
    casual_json = {}
    for i in range(0, len(sentences)):
        bad_words = ["fuck", "fucking", "bitch", "cock", "tits", "tities", "shit", "bullshit", "penis", "vagina", "fucker", "shithead"]
        tokenized_sent = tokenize_sentence(sentences[i])
        tokenized_reply = tokenize_sentence(sentences[i])
        clean = True
        for token in tokenized_sent:
            if token in bad_words:
                clean=False
        for token in tokenized_reply:
            if token in bad_words:
                clean=False
        if clean:
            casual_json[sentences[i]] = replies[i]
    casual_file = "datasets/test.json"
    json_object = json.dumps(casual_json, indent = 4) 
    with open(casual_file, "w") as f:
        f.write(json_object)