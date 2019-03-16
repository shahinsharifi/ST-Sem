# -*- coding: utf-8 -*-

class WordMatching:

    def __init__(self, spell):
        self.spell = spell

    def checkSceneTextCorrelation(self, scenes, texts):
        result = []
        for text in texts: # checking scene/text labels one by one
            for scene in scenes:
                if (len(text) > 0):
                    text = text[0]
                text = self.spell.correction(text)
                va = self.word2vec(text)
                vb = self.word2vec(scene)
                score = self.cosDist(va, vb)
                if(score > 0.9) and (scene not in result):
                    result.append((scene, score))
        return result


    def word2vec(self, word):
        from collections import Counter
        from math import sqrt

        # count the characters in word
        cw = Counter(word)
        # precomputes a set of the different characters
        sw = set(cw)
        # precomputes the "length" of the word vector
        lw = sqrt(sum(c * c for c in cw.values()))

        # return a tuple
        return cw, sw, lw


    def cosDist(self, v1, v2):
        # which characters are common to the two words?
        common = v1[1].intersection(v2[1])
        # by definition of cosine distance we have
        return sum(v1[0][ch] * v2[0][ch] for ch in common) / v1[2] / v2[2]




