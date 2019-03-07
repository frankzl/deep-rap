import nltk
from nltk.corpus import cmudict as cmu

arpabet = cmu.dict()
unknown_words = set()

def get_phonem(text):
    try:
        return arpabet[text][0]
    except:
        unknown_words.add(text)
        return ""


def get_phonem_string(text):
    try:
        return "".join(arpabet[text][0])
    except:
        unknown_words.add(text)
        return ""


def get_phonem_string_spaced(text):
    try:
        return " ".join(arpabet[text][0])
    except:
        unknown_words.add(text)
        return ""


def text_to_phonem(text):
    default_wt = nltk.word_tokenize
    words = default_wt(text)
    all_phonems = (" ").join([get_phonem_string_spaced(word) for word in words]) 
    return all_phonems


def get_unknown_words():
    return list(unknown_words)


def get_unkowns_as_list():
    if len(unknown_words) > 0:
        return "\n".join(get_unknown_words())
    return []


# use logios lextool to get generated phonetics
# http://www.speech.cs.cmu.edu/tools/lextool.html
def create_unknown_dict_from_text(unknown_list):
    unknown_dict = {}
    splits = unknown_list.split("\n")
    for split in splits:
        try:
            word, phonem = split.split(":")
        except:
            print(split)
        unknown_dict[word.lower()] = [phonem.split(" ")]
    return unknown_dict


def update_arpabet(dictionary):
    global arpabet
    global unknown_words
    arpabet = {**arpabet, **dictionary}
    unknown_words = set()
    

# get ryhme for word
# level represents accuracy of rhyme: take the last {level} phonems to compare
def rhyme(inp, level):
    entries = cmu.entries() # [(word, phonetics),...]
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return list(set(rhymes))