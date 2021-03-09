import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

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

array_col = ['skincare', 'hair', 'make-up', 'other']

### data
df_train = pd.read_csv("data/hackathon_loreal_train_set.csv")
df_train = df_train.drop(['Unnamed: 0'], axis = 1)





import re

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
                    '-', '_', '\\', '@','?','!','\'', '[', '<', '>', '£', '$', '”', "\u2063", "•", "'s", "“"]
    stoplist = stopwords.words('english')

    TextCleanPonctuation = [word for word in text if word not in stop_punctuation and word not in stoplist]
    return TextCleanPonctuation

def regroupe_text(list_word):
    return ' '.join(list_word)

def count_text(text):
    cv=CountVectorizer()
    text_count = word_count_vector=cv.fit_transform(text)
    return text_count

def preprocessing_nlp(singleText):
    lemmatizer = WordNetLemmatizer()
    singleText = del_emoji(singleText)
    LowerSingleText = singleText.lower()
    tokenizeSingletext = word_tokenize(LowerSingleText)
    lemmatizeSingleText = list(map(lemmatizer.lemmatize, tokenizeSingletext))
    TextCleanPonctuation = text_clean_ponctuation(lemmatizeSingleText)
    # text_count_result = count_text( )

    return TextCleanPonctuation

def clean_text(text):
    text_preprocessed = preprocessing_nlp(text)
    return  regroupe_text(text_preprocessed)





df_train["clean_content"] = df_train.text.apply(clean_text)

from sklearn import preprocessing

df_train_clean = df_train.drop(["text"], axis = 1)
X = df_train['clean_content']
Y = df_train.drop(['clean_content', 'index', 'text'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


cv=CountVectorizer()
cv.fit(x_train)
x_train_count = cv.fit_transform(x_train).astype('int32').toarray()
x_test_count = cv.transform(x_test).astype('int32').toarray()




from keras import backend as K

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


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(128, input_shape=(70574,), activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [f1_m])

model.fit(x_train_count, y_train, epochs=5)


evaluate = model.evaluate(x_test_count, y_test)