# GUI which will interact with user and takes the inpput which we'll later send to our model to process it.

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pk1','rb'))
classes = pickle.load(open('classes.pk1','rb'))
def clean_up_sentance(sentance):
    # splitting words into array
    sentance_words = nltk.word_tokenize(sentance)
    # stemming words
    sentance_words = [lemmatizer.lemmatize(word.lower()) for word in sentance_words]
    return sentance_words
# returns bag of word array
def bag_of_words(sentance, words, show_details=True):
    # tokenizing patterns
    sentance_words = clean_up_sentance(sentance)
    # vocabulary matrix (bag of words)
    bag = [0]*len(words)
    for s in sentance_words:
        for i,word in enumerate(words):
            if word == s:
                # assigning 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))
def predict_class(sentance):
    # filter below threshold predictions
    p = bag_of_words(sentance, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strenght probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

