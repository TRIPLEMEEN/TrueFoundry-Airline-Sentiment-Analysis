import re
from fastapi import FastAPI, Form
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer


def clean_table(t):
        new = pd.DataFrame({'clean_tweet':[t]})
        new['clean_tweet'] = new['clean_tweet'].apply(str)

        def remove_pattern(new, pattern): 
            r = re.findall(pattern, new)
            for word in r:
                new = re.sub(word, "", new)
            return new
        new['clean_tweet'] = np.vectorize(remove_pattern)(new['clean_tweet'], "@[\w]*")
        new['clean_tweet'] = new['clean_tweet'].replace("[^a-zA-Z#]", " ")
        new['clean_tweet'] = new['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
        new['clean_tweet'] = new['clean_tweet'].apply(lambda x: x.split())
        new['clean_tweet'] = new['clean_tweet'].apply(lambda x: ' '.join(x))
        return new['clean_tweet']

file = open('sentiment_model.pickle', 'rb')
model =pickle.load(file)

new_bow_vectorizer = pickle.load(open("bow_vectorizer.pickle", 'rb'))  


app = FastAPI()

@app.post("/sentiment")
async def sentiment(word: str = Form(...)):
    clean_new = clean_table(word)

    new_bow = new_bow_vectorizer.transform(clean_new)
    pred = model.predict(new_bow)
    if pred[0] == 'negative':
        return 'Negative'
    elif pred[0] == 'positive':
        return 'Positive'