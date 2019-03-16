# -*- coding: utf-8 -*-
from termcolor import colored
from gensim.models import KeyedVectors
from spellchecker import SpellChecker

class WordEmbeddings:

    def __init__(self):
        self.model_path = "./models/twitter/word2vec_twitter_model.bin"
        self.model = KeyedVectors.load_word2vec_format(self.model_path, unicode_errors='ignore', binary=True)
        print('Word2Vec is loaded for  semantic similarity task ...')


    def checkSemanticSimilarity(self, labels, words):
        print(colored('############ WORD SIMILARITY CHECK ################', 'blue'))
        for label in labels:
            tmpSimilarity= 0
            tmpWord = ''
            for word in words:
                if len(word) > 2:
                    spell = SpellChecker()
                    newWord = spell.correction(word)
                    if (label in self.model.vocab) and (newWord in self.model.vocab):
                        similarity = self.model.similarity(label, newWord)
                        if similarity > tmpSimilarity:
                            tmpWord = newWord
                            tmpSimilarity = similarity

            print label + '=>' + tmpWord + ': ' + colored(str(tmpSimilarity), 'green')
