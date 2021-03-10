import re
from keras import backend as K

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline


import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
from nltk.stem.snowball import SnowballStemmer

from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk import ngrams
from nltk import sent_tokenize

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



try:
    import cPickle as pickle
except ImportError:
    import pickle  

with open('Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}



def convert_emojis_to_word(text):
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
    return text


# Function to remove emoji.
def del_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
                           
    return emoji_pattern.sub(r'', string)




def text_clean_ponctuation(text):
    stop_punctuation = [':', '(', ')', '/', '|', ',', ']', ';',
                    '.', '*', '#', '"', '&', '~', '``',
                    '-', '_', '\\', '@','?','!','\'', '[', '<', '>', '£', '$', '”', "\u2063", "•", "'s", "“", "+"]
    stoplist = stopwords.words('english')

    TextCleanPonctuation = [word for word in text if word not in stop_punctuation and word not in stoplist]
    return TextCleanPonctuation

def regroupe_text(list_word):
    return ' '.join(list_word)

def count_text(text):
    cv=CountVectorizer()
    text_count = word_count_vector=cv.fit_transform(text)
    return text_count

def delete_cara_no_useful(elem):
    re.sub('[^a-zA-Z0-9 \n\.]', '',elem)
    return elem

def preprocessing_nlp(singleText):
    lemmatizer = WordNetLemmatizer()
    singleText = del_emoji(singleText)
    LowerSingleText = singleText.lower()
    tokenizeSingletext = word_tokenize(LowerSingleText)
    lemmatizeSingleText = list(map(lemmatizer.lemmatize, tokenizeSingletext))
    TextCleanPonctuation = text_clean_ponctuation(lemmatizeSingleText)
    clean_text = list(map(delete_cara_no_useful, TextCleanPonctuation))
    # text_count_result = count_text( )

    return clean_text

def clean_text(text):
    text_preprocessed = preprocessing_nlp(text)
    return  regroupe_text(text_preprocessed)



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))