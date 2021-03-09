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




df_train["clean_content"] = df_train.text.apply(clean_text)



df_train_clean = df_train.drop(["text"], axis = 1)
X = df_train['clean_content']
Y = df_train.drop(['clean_content', 'index', 'text'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)


cv=CountVectorizer()
cv.fit(x_train)
x_train_count = cv.fit_transform(x_train).astype('int32').toarray()
x_test_count = cv.transform(x_test).astype('int32').toarray()



model = Sequential()

model.add(Dense(128, input_shape=(70574,), activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [f1_m])

model.fit(x_train_count, y_train, epochs=5)


evaluate = model.evaluate(x_test_count, y_test)