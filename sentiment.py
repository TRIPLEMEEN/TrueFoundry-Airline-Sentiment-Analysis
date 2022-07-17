import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pickle




#read file
df = pd.read_csv('airline_sentiment_analysis.csv')


#clean text
df['text']=df['text'].apply(str)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", str(input_txt))
    return input_txt
df['clean_text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
df['clean_text'] = df['clean_text'].str.replace("[^a-zA-Z#]", " ")
df['clean_text'] = df['clean_text'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

# feature extraction
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(df['clean_text'])

x_train, x_test, y_train, y_test = train_test_split(bow, df['airline_sentiment'], random_state=42, test_size=0.30)

# training
model = LogisticRegression()
model.fit(x_train, y_train)

#ddump model and vectorizer
pickle.dump(model, open('sentiment_model.pickle', 'wb'))
pickle.dump(bow_vectorizer, open("bow_vectorizer.pickle", "wb"))