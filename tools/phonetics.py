import nltk
from nltk.corpus import cmudict as cmu

arpabet = cmu.dict()

def get_phonem(text):
    try:
        return arpabet[text][0]
    except:
        return ""

def get_phonem_string(text):
    try:
        return "".join(arpabet[text][0])
    except:
        return ""

def get_phonem_string_spaced(text):
    try:
        return " ".join(arpabet[text][0])
    except:
        return ""

# get ryhme for word
# level represents accuracy of rhyme: take the last {level} phonems to compare
def rhyme(inp, level):
    entries = cmu.entries() # [(word, phonetics),...]
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return list(set(rhymes))