# Prepare libraries and data
import nltk
import re
import os
import heapq
import pandas as pd
import numpy as np
from string import punctuation
punctuation = punctuation + '\n'
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization import summarize
from nltk.cluster.util import cosine_distance


# Text categories
categories = ['Business', 'Entertainment', 'Politics', 'Sport', 'Technology']

# Load dataset
data = pd.read_csv(r"dataset/bbc_processed.csv")


# Data preprocessing
# Delete links:
def delete_links(input_text):
    pattern  = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    out_text = re.sub(pattern, ' ', input_text)
    return out_text

# Fixing word lengthening:
def delete_repeated_characters(input_text):
    pattern  = r'(.)\1{2,}'
    out_text = re.sub(pattern, r"\1\1", input_text)
    return out_text

# Delete bad symbols:
def clean_text(input_text):
    replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
    out_text = re.sub(replace, " ", input_text)
    words = nltk.word_tokenize(out_text)
    words = [word for word in words if word.isalpha()]
    out_text = ' '.join(words)
    return out_text

# Delete stopwords:
def delete_stopwords(input_text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    wnl = nltk.WordNetLemmatizer()
    lemmatizedTokens =[wnl.lemmatize(t) for t in tokens]
    out_text = [w for w in lemmatizedTokens if not w in stop_words]
    out_text = ' '.join(out_text)
    return out_text

# Text prepare:
def text_prepare(input_text):
    out_text = delete_links(input_text)
    out_text = delete_repeated_characters(out_text)
    out_text = clean_text(out_text)
    out_text = delete_stopwords(out_text)
    out_text = out_text.lower()
    return out_text

# Spliiting the data to train and test
# 80% of the data used for models train
# 20% of the data used for test and validation
train_text, test_text, train_cat, test_cat = train_test_split(data['Processed Text'], data['Category Encoded'], test_size=0.2, random_state=0)

# TF-IDF vectorizer:
def tfidf_features(X_train, X_test, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, ngram_range))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test

# Building the gensim summarizer
def gensim_summarizer(article, number_of_sentence):
  r = 1
  extracts=sent_tokenize(article)
  if(number_of_sentence < len(extracts)):
    r = number_of_sentence / len(extracts)
  
  summary = summarize(article, ratio=r, split=False)
  return summary
  
# Building the nltk summarizer
def nltk_summarizer(input_text, number_of_sentence):
    stopWords = set(nltk.corpus.stopwords.words("english"))
    word_frequencies = {}  
    for word in nltk.word_tokenize(input_text):  
        if word not in stopWords:
            if word not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(input_text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(number_of_sentence, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
     
    return summary

# Building the textrank summarizer
def get_sentences(article):
  extracts=sent_tokenize(article)
  sentences=[]
  for extract in extracts:
    clean_sentence=extract.replace("[^a-zA-Z0-9]"," ")   ## Removing special characters
    obtained=word_tokenize(extract) 
    sentences.append(obtained)
  return sentences, extracts


def get_similarity(sent_1,sent_2,stop_words):
  
  sent_1=[w.lower() for w in sent_1]
  sent_2=[w.lower() for w in sent_2]

  total=list(set(sent_1+sent_2)) ## Removing duplicate words in total set

  vec_1= [0] * len(total)
  vec_2= [0] * len(total)


  ## Count Vectorization of two sentences
  for w in sent_1:
    if w not in stop_words:
      vec_1[total.index(w)]+=1

  for w in sent_2:
    if w not in stop_words:
      vec_2[total.index(w)]+=1

  return 1-cosine_distance(vec_1,vec_2)

def build_matrix(sentences):
  stop_words = set(nltk.corpus.stopwords.words("english"))

  sim_matrix=np.zeros((len(sentences),len(sentences)))
  ## Adjacency matrix

  for id1 in range(len(sentences)):
    for id2 in range(len(sentences)):
      if id1==id2:  #escaping diagonal elements
        continue
      else:
        sim_matrix[id1][id2]=get_similarity(sentences[id1],sentences[id2],stop_words)

  return sim_matrix

def textrank(text, eps=0.000001, d=0.85):
    score_mat = np.ones(len(text)) / len(text)
    delta=1
    while delta>eps:
        score_mat_new = np.ones(len(text)) * (1 - d) / len(text) + d * text.T.dot(score_mat)
        delta = abs(score_mat_new - score_mat).sum()
        score_mat = score_mat_new
    return score_mat_new


def textrank_summarizer(article, number_of_sentence):
  summarized=[]
  clean_sentences, sentences=get_sentences(article)
  sim_matrix=build_matrix(clean_sentences)
  score=textrank(sim_matrix)

  ranked_sentence = sorted(((score[i],s) for i,s in enumerate(sentences)), reverse=True)

  if len(ranked_sentence) < number_of_sentence: number_of_sentence = len(ranked_sentence)
  
  for i in range(number_of_sentence):
      summarized.append(ranked_sentence[i][1])

  return " ".join(summarized)

# Summarize and predict for input text:
def summarize_category(article, statements, model_name, summarizer="gensim"):
	summary_text = ""
	if(summarizer == "nltk"):
		summary_text = nltk_summarizer(article, statements)
	elif(summarizer == "textrank"):
		summary_text = textrank_summarizer(article, statements)
	else:
		summary_text = gensim_summarizer(article, statements)
	input_text_arr = [text_prepare(article)]
	f_train, f_test = tfidf_features(train_text, input_text_arr, 2)
	text_prediction = model_name.predict(f_test.toarray())
	text_category = categories[text_prediction[0]]
	return summary_text, text_category