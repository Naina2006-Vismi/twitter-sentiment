import re
import string
import nltk
import emoji
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

SLANG_DICT = {
    "bussin"  : "amazing",
    "slaps"   : "amazing",
    "lit"     : "exciting",
    "goated"  : "greatest",
    "slay"    : "succeed",
    "mid"     : "mediocre",
    "cap"     : "lie",
    "no cap"  : "honestly",
    "fr fr"   : "honestly",
    "ngl"     : "honestly",
    "lowkey"  : "somewhat",
    "highkey"  : "very",
    "vibe"    : "feeling",
    "sus"     : "suspicious",
    "hits different" : "feels special",
    "rent free": "constantly thinking",
    "understood the assignment" : "did well",
    "ate"     : "performed well",
    "periodt"  : "definitely",
    "imo"     : "in my opinion",
    "irl"     : "in real life",
    "tbh"     : "honestly",
    "nvm"     : "never mind",
    "omg"     : "oh my god",
    "lol"     : "laughing",
    "lmao"    : "laughing",
    "smh"     : "disappointed",
    "ugh"     : "frustrated",
    "bruh"    : "disappointed",
    "fam"     : "friend",
    "bro"     : "friend",
}

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_map = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_map.get(tag, wordnet.NOUN)

def handle_negation(text):
    negation_words = [
        "not", "never", "no", "neither", "nor", "barely",
        "hardly", "scarcely",

        "dont", "doesnt", "didnt", "wont", "wouldnt",
        "cant", "couldnt", "shouldnt", "isnt", "arent",
        "wasnt", "werent", "hasnt", "havent", "hadnt",
       
        "don't", "doesn't", "didn't", "won't", "wouldn't",
        "can't", "couldn't", "shouldn't", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't"
    ]
    words = text.split()
    result = []
    negate_next = False
    for word in words:
        if word.lower() in negation_words:
            negate_next = True
           
        elif negate_next:
            result.append(f"not_{word}")
            negate_next = False
        else:
            result.append(word)
    return ' '.join(result)

def replace_slang(text):
    text_lower = text.lower()
    for slang, replacement in sorted(SLANG_DICT.items(),
                                      key=lambda x: len(x[0]),
                                      reverse=True):

        pattern = r'\b' + re.escape(slang) + r'\b'
        text_lower = re.sub(pattern, replacement, text_lower)
    return text_lower

def convert_emoji(text):
    return emoji.demojize(text, delimiters=(' ', ' '))

def detect_language(text: str) -> bool:
    if len(text.split()) < 3:
        return True
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return True
    
def detect_language(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return True 

def clean_text_v2(text):
   
    text = convert_emoji(text)

    text = replace_slang(text)

    text = text.replace("'", "")

    text = handle_negation(text)

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    punct = string.punctuation.replace('_', '')
    text = text.translate(str.maketrans('', '', punct))

    words = text.split()

    negation_stops = {"not", "no", "never", "nor", "neither"}
    words = [w for w in words if w not in stop_words
             or w in negation_stops
             or w.startswith('not_')]

    words = [w for w in words if len(w) > 2 or w.startswith('not_')]

    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w))
             if not w.startswith('not_') else w
             for w in words]

    return ' '.join(words).strip()

def analyze_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if len(s.split()) >= 2]
    return sentences

def is_likely_neutral(text):
    neutral_signals = [
        "meeting at", "scheduled for", "reminder",
        "please note", "fyi", "heads up",

        "what time", "where is", "how do i",
        "when does", "who is",
    ]
    
    text_lower = text.lower()

    for signal in neutral_signals:
        if signal in text_lower:
            return True

    cleaned = clean_text_v2(text)
    if len(cleaned.split()) < 2:
        return True
        
    return False

def get_reliability_score(text: str, confidence: float) -> str:
    if len(text.split()) < 4:
        return "Low (Short text)"
    
    sarcasm_signals = [
        "totally", "just what", "oh great", "oh perfect",
        "thanks for nothing", "because obviously", "wow really",
        "sure totally", "absolutely brilliant", "how wonderful",
        "just great", "oh wonderful", "how lovely",
        "what a surprise", "shocked pikachu"
    ]

    if any(signal in text.lower() for signal in sarcasm_signals):
        return "Low (Sarcasm detected)"
    
    if confidence < 0.65:
        return "Low (Low confidence)"
    
    if confidence < 0.75:
        return "Medium (Moderate confidence)"
    
    if confidence >= 0.85:
        return "High (Good confidence)"
    
    return "Medium (Uncertain confidence)"
