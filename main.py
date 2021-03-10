import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense


from function import clean_text, f1_m


array_col = ['skincare', 'hair', 'make-up', 'other']

### data
df_train = pd.read_csv("data/hackathon_loreal_train_set.csv")
df_train = df_train.drop(['Unnamed: 0'], axis = 1)

df_test = pd.read_csv("data/hackathon_loreal_test_set.csv")


df_train["clean_content"] = df_train.text.apply(clean_text)

df_test["clean_content"] = df_test.text.apply(clean_text)
df_test.drop(["text"], axis=1, inplace=True)

X_test = df_test["clean_content"]

df_train_clean = df_train.drop(["text"], axis = 1)
X_train = df_train['clean_content']
Y_train = df_train.drop(['clean_content', 'index', 'text'], axis = 1)

#x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=4)


cv=CountVectorizer()
#cv.fit(x_train)
x_train_count = cv.fit_transform(X_train).astype('int32').toarray()
x_test_count = cv.transform(X_test).astype('int32').toarray()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
#clf.fit(x_train_count, y_train)

model = Sequential()

model.add(Dense(128, input_shape=(80736,), activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [f1_m])

model.fit(x_train_count, Y_train, epochs=5)


predict = model.predict(x_test_count)
predict = predict.round()

array_index = df_test["index"].to_numpy().tolist()

import json

data = {}
data['hash'] = 'f44bde5ccb28af0d'
data['data'] = []
data['data'].append({
    'index': array_index,
    'label': predict.tolist()
})

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

#print(f1_score(predict, y_test, average='macro'))