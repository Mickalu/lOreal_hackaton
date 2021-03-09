import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

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

df_train = pd.read_csv("data/hackathon_loreal_train_set.csv")

df_train = df_train.drop(['Unnamed: 0'], axis = 1)



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

def preprocessing(singleText):
    lemmatizer = WordNetLemmatizer()
    singleText = del_emoji(singleText)
    LowerSingleText = singleText.lower()
    tokenizeSingletext = word_tokenize(LowerSingleText)
    lemmatizeSingleText = list(map(lemmatizer.lemmatize, tokenizeSingletext))
    TextCleanPonctuation = text_clean_ponctuation(lemmatizeSingleText)

    return TextCleanPonctuation

def clean_text(text):
    text_preprocessed = preprocessing(text)
    return regroupe_text(text_preprocessed)

df_train["clean_content"] = df_train.text.apply(clean_text)





from sklearn import preprocessing

df_train_clean = df_train.drop(["text"], axis = 1)
X = df_train['clean_content']
Y = df_train.drop(['clean_content', 'index'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

tfidf = TfidfTransformer()

tfidf.fit(x_train)

x_vectorized_train = tfidf.transform(x_train)
x_vectorized_test = tfidf.transform(x_test)

model.fit(x_vectorized_train, y_train)
y_predicted = model.predict(x_vectorized_test)

pipe = make_pipeline(TfidfTransformer())
pipe.fit(x_train)
X_feat_train = pipe.transform(x_train)
X_feat_test = pipe.transform(x_test)

X_feat_train_numpy = np.asarray(X_feat_train)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(128, input_shape=(8457,), activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['F1 score'])

model.fit(X_feat_train_numpy, y_train, epochs=5)