from nlp_helpers import *
"""
CHECKLIST:
- load_classifier_data: PASS
- classify: PASS
- load_chatbot_data: PASS
"""
FILE = "datasets/intents.json"
classifier_data = load_classifier_data(FILE)
chatbot_data = load_chatbot_data(FILE)

classifier_vocab = LxtClassifierVocab()
classifier_vocab.create_vocab(classifier_data)

chatbot_vocab = ChatbotVocab()
chatbot_vocab.create_vocab(chatbot_data)

save_classifier_vocab = "saved_data/classifier/classifier_vocab.pickle"
dump_pickle(classifier_vocab, save_classifier_vocab)
save_chatbot_vocab = "saved_data/seq2seq/chatbot_vocab.pickle"
dump_pickle(chatbot_vocab, save_chatbot_vocab)
