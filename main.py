import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense


from function import clean_text, f1_m


array_col = ['skincare', 'hair', 'make-up', 'other']

### data
df_train = pd.read_csv("data/hackathon_loreal_train_set.csv")
df_train = df_train.drop(['Unnamed: 0'], axis = 1)

df_test = pd.read_csv("data/hackathon_loreal_test_set.csv")
df_test = df_test.drop(['Unnamed: 0'], axis = 1)



df_train["clean_content"] = df_train.text.apply(clean_text)
df_test["clean_content"] = df_test.text.apply(clean_text)



df_train_clean = df_train.drop(["text"], axis = 1)
df_test_clean = df_test.drop(["text"], axis = 1)


x_train = df_train_clean["clean_content"]
x_test = df_test_clean["clean_content"]

y_train = df_train_clean.drop(["clean_content", "index"], axis = 1)

cv=CountVectorizer()
cv.fit(x_train)

x_train_count = cv.fit_transform(x_train).astype('int32').toarray()
x_test_count = cv.transform(x_test).astype('int32').toarray()



model = Sequential()

model.add(Dense(128, input_shape=(80736,), activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [f1_m])

model.fit(x_train_count, y_train, epochs=5)


y_predicted = model.predict(x_test_count)