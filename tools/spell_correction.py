import re
import tools.processing as pre

def remove_extra_space(text):
    return re.sub(" +", " ", text)

def replace_linebreak( text ):
    return text.replace( "\n", " ; ")


CONTRACTION_MAP = {
        "'em":"them","y'know": "you know", "'hem": "them",                   
        "c'mon": "come on", "'caine": "cocaine",
        "mo'": "my", "cha'": "ya", "'cha": "ya",
        "whaddya": "what do ya", "nuttin": "nothing",
        "thru": "through", "shoulda": "should have",
        "lets": "let us", "let's": "let us", "f'real": "for real",
        "'til": "until", "i'ma": "i am going to",
        "ima": "i am going to", "'cross": "across",
        " imma ": "i am going to", "tho'": "though",
        "st8": "straight", "til'": "until", "str8": "straight",
        "'sll ": "s will", "withcha": "with ya", "befo'": "before",
        "cuz'": "because","coz'": "because","cuz'": "because","cuz": "because",
        "coz": "because","'im": "him", "'bout": "about","tha'": "the","tu'": "to","'n'": "and",
        " n ": " and ", "'n": "and", "yo'": "your", "witcha": "with ya", "wit'": "with",
        "whaddup": "what is up", "pro'lly": "probably", "prolly": "probably", 
        "'laxin": "relaxing", "tryna": "trying to", "'tack": "attack",
        "'head": "ahead", "lil'": "little", "getcha": "get ya",
        "wit'chu": "with you", "get'cha": "get ya","gon": "going to",
        "sweatcha": "sweat ya", "e'ry": "every", "what'cha": "what ya",
        " aight ": " all right ", "hitcha": "hit ya", "hit'cha": "hit ya",
        "gov'na": "governor", "'fore": "before", "mill'": "million",
        "ain't": "is not", "aren't": "are not","can't": "can not", 
        "can't've": "can not have", "'cause": "because", "could've": "could have", 
        "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
        "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
        "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
        "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
        "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
        "i'll've": "i will have","i'm": "i am", "i've": "i have", 
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
        "i'll've": "i will have","i'm": "i am", "i've": "i have", 
        "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
        "it'll": "it will", "it'll've": "it will have","it's": "it is", 
        "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
        "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
        "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
        "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
        "she's": "she is", "should've": "should have", "shouldn't": "should not", 
        "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
        "this's": "this is",
        "that'd": "that would", "that'd've": "that would have","that's": "that is", 
        "there'd": "there would", "there'd've": "there would have","there's": "there is", 
        "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
        "they'll've": "they will have", "they're": "they are", "they've": "they have", 
        "to've": "to have", "wasn't": "was not", "we'd": "we would", 
"we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", "why'd": "why did",
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have", 
                   "gon'": "going to",
                   " an'": "and", "yall": "ya all", "youre": "you are",
                   " de ": " the ", "en'": "ing",
                   "in'": "ing", "'ll": " will", "'ve": " have", " u ": " you ", 
                   "mutha": "mother", "brotha": "brother", "2pac": "tupac", " nite": " night"
                  } 

def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
            flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

def tokens(text):
    """
        Get all words from corpus
    """
    return re.findall(r'\w+', text.lower())

def edits0(word):
    """
    Return all strings that are zero edits away (i.e. the word itself).
    """
    return{word}

def edits1(word):
    """
    Return all strings that are one edits away.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        """
        return a list of all possible pairs
        that the input word is made of
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a,b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a,b) in pairs if len(b) >1]
    replaces = [a+c+b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a,b) in pairs for c in alphabet]
    return(set(deletes + transposes + replaces + inserts))

def edits2(word):
    """
    return all strings that are two edits away.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words, word_counts):
    return {w for w in words if w in word_counts}

unk = []
def correct(word, word_counts, unknowns):
    candidates = (known(edits0(word), word_counts) or
            known(edits1(word), word_counts) or
            known(edits2(word), word_counts) or
            [word])
    cand = max(candidates, key=word_counts.get)
    if(candidates == known(edits2(word), word_counts)):
        if(cand != word):
            unknowns.append(word)
    return cand

def correct_text(text, word_counts):
    unk = []
    processed = text.replace('\n', ' \n ')
    corrected = [ correct(word, word_counts, unk) for word in processed.split(" ") ]
    
    return " ".join(corrected), unk

def recorrect(word, word_counts):
    candidates = (known(edits0(word), word_counts) or
                 known(edits1(word), word_counts) or
                 known(edits2(word), word_counts) or
                 [word])
    return max(candidates, key=word_counts.get)

def recorrect_text(text, ref_text):
    processed = replace_linebreak( text )
    processed = remove_extra_space( processed )

    expanded = expand_contractions(processed, CONTRACTION_MAP)

    # some contractions cannot be fixed so easily
    # example: Tom's bag is outside VS Tom's outside
    expanded = expanded.replace("'s ", " 's ")
    expanded = expanded.replace("s'", " 's ")

    import collections

    wordlist = ref_text 
    WORDS = tokens(wordlist) + [";"]
    WORD_COUNTS = collections.Counter(WORDS)

    corrected = [ recorrect(word, WORD_COUNTS) for word in processed.split(" ") ]
    
    return " ".join(corrected)   

def correct_text( text, ref_text ):
    processed = replace_linebreak( text )
    processed = remove_extra_space( processed )

    expanded = expand_contractions(processed, CONTRACTION_MAP)

    # some contractions cannot be fixed so easily
    # example: Tom's bag is outside VS Tom's outside
    expanded = expanded.replace("'s ", " 's ")
    expanded = expanded.replace("s'", " 's ")

    import collections

    wordlist = ref_text 
    WORDS = tokens(wordlist) + [";"]
    WORD_COUNTS = collections.Counter(WORDS)

    return correct_text(text, WORD_COUNTS)
