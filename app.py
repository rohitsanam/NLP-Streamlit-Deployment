import time
#os.chdir('')
#st.write(os.getcwd())
import numpy as np

import pandas as pd
import os
#os.chdir('')
#st.write(os.getcwd())

import streamlit as st

# px a high level package for plotly
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
from urllib.request import urlopen
import json


import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import gutenberg as gt
import os
from nltk.tokenize import word_tokenize, sent_tokenize
#os.chdir('F:/Locker/Sai/SaiHCourseNait/DecBtch/R_Datasets/Corpora/')
import re
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown
import bs4 as bs
import urllib.request
import heapq

import urllib
import bs4 as bs
from gensim.models import Word2Vec
# imports needed and logging
import gzip
import gensim 
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

import lxml
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")


import collections
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud,STOPWORDS
#%matplotlib inline

#os.chdir('F:/Locker/Sai/SaiHCourseNait/DecBtch/R_Datasets/')
from collections import Counter
import sys
# !{sys.executable} -m spacy download en
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
#from gensim.utils import lemmatize, simple_preprocess
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
#stop_words = stopwords.words('english')
#stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

#%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)


import urllib.request

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Polavaram_Project').read()
#source 
soup = bs.BeautifulSoup(source,'lxml')
#soup # is more clear

# Fetching the data
text = ""
for paragraph in soup.find_all('p'):
    #print(paragraph.text)
    text += paragraph.text 
    text += '\n\n'

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
#print(text)


clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)
#clean_text

# Tokenize sentences
sentences = nltk.sent_tokenize(text)
#print(sentences[:5])

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# Convert to list
#data = df.content.values.tolist()
#data_words = list(sent_to_words(data))
#print(data_words[:1])




stop_words = nltk.corpus.stopwords.words('english')
#stop_words

# Word counts 
word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

#print(word2count)


for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())
#print(word2count)


# Product sentence scores    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

print(sent2score)


# Gettings best 5 lines             
best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)
print('--------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)
    st.write(sentence)
print('--------------------------------------------------------')


#type(best_sentences)
lst_best_wrds = []
for sent in best_sentences:
    lst = word_tokenize(sent)
    for wrd in lst:
        lst_best_wrds.append(wrd)
#lst_best_wrds 

from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

st.set_option('deprecation.showPyplotGlobalUse', False)

def cloud(image, text, max_word, max_font, random):
    stopwords = set(STOPWORDS)
    stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
    'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
    'put', 'seem', 'asked', 'made', 'half', 'much',
    'certainly', 'might', 'came'])


    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word, mask=image,
                stopwords=stopwords, max_font_size=max_font, random_state=random)
    # generate word cloud
    wc.generate(text)

    # create coloring from image
    image_colors = ImageColorGenerator(image)

    # show
    plt.figure(figsize=(100,100))
    fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 2]})
    axes[0].imshow(wc, interpolation="bilinear")
    # recolor wordcloud and show
    # we could also give color_func=image_colors directly in the constructor
   # axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    axes[1].imshow(image, cmap=plt.cm.gray, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    
    st.pyplot()
    
   


def main():
    #st.write("# Text Summarization with a WordCloud")
    #st.write("[By Boadzie Daniel](https://boadzie.surge.sh)")
    max_word = st.sidebar.slider("Max words", 200, 3000, 200)
    max_font = st.sidebar.slider("Max Font Size", 50, 350, 60)
    random = st.sidebar.slider("Random State", 30, 100, 42 )
    
	#image = st.file_uploader("Choose a file(preferably a silhouette)")
    #text = st.text_area("Add text ..")
    
    text = best_sentences
    #if image and text is not None:
        #if st.button("Plot"):
            #st.write("### Original image")
            #image = np.array(Image.open(image))
            ## st.image(image, width=100, use_column_width=True)
       
            #st.write("### Word cloud")
            #st.write(cloud(image, text, max_word, max_font, random), use_column_width=True)
	

main()
#*
#st.write('st.write(text)')
#st.write(text)
#st.text_area(text)
#st.write(sent2score.keys())
#df_s2s = pd.DataFrame(sent2score,index=sent2score.keys())
df_s2s = pd.DataFrame(sent2score.items(),columns=['sent','score'])
#st.write(df_s2s['sent'].str.len().hist())



from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()


import seaborn as sns
stopwords = nltk.corpus.stopwords.words('english')
#stop=set(stopwords.words('english'))
stopwords.append('The')
stopwords.append('will')
stopwords.append('In')
stopwords.append('would')
stopwords.append('its')
stopwords.append('i.e')
stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

stop=set(stopwords)
corpus=[]
new= df_s2s['sent'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]
from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1


corp_no_stopwords = []
for w in corpus:
	if w not in stop:
		corp_no_stopwords.append(w)

#st.write(corpus)

#st.write(corp_no_stopwords)
corpus = corp_no_stopwords

show_wordcloud(corpus)
st.pyplot()


counter=collections.Counter(corpus)
most=counter.most_common()

#st.write(most)
#st.write(stop)

x, y= [], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)

#st.write(x)        
st.header('Most Common')
sns.barplot(x=y,y=x)
st.pyplot()




from nltk.util import ngrams
#list(ngrams(['I' ,'went','to','the','river','bank'],2))
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]


#top_n_bigrams=get_top_ngram(df_s2s['sent'],2)[:10] 
#x,y=map(list,zip(*top_n_bigrams)) 
#sns.barplot(x=y,y=x) 
#st.pyplot()

for i in range(2,6):
	top_n_bigrams=get_top_ngram(df_s2s['sent'],i)[:10] 
	x,y=map(list,zip(*top_n_bigrams)) 
	sns.barplot(x=y,y=x) 
	st.pyplot()






#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('gutenberg')

def preprocess_news(df):
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for news in df['sent']:
        words=[w for w in word_tokenize(news) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus

#st.write(corpus)

corpus=preprocess_news(df_s2s)
#st.write(corpus)

dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
#st.write(bow_corpus)

#'''
#lda_model = gensim.models.LdaMulticore(bow_corpus, 
#                                   num_topics = 4, 
#                                   id2word = dic,                                    
#                                   passes = 10,
#                                   workers = 2)
#st.write(lda_model.show_topics())




#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
#vis
#st.pyplot()
#'''




st.header('Sentences Length')
df_s2s['sent'].str.len().hist()
st.pyplot()


st.header('Avg Word Length in sentence')
df_s2s['sent'].str.split().\
   apply(lambda x : [len(i) for i in x]). \
   map(lambda x: np.mean(x)).hist()
st.pyplot()





st.table(df_s2s.head(10))
#st.write('st.write(text)')



from textblob import TextBlob
#TextBlob('100 people killed in Iraq').sentiment

def polarity(text):
    return TextBlob(text).sentiment.polarity

st.header('Polarity of Statements')
news = df_s2s
news['headline_text'] = news['sent']
news['polarity_score']=news['headline_text'].\
   apply(lambda x : polarity(x))
news['polarity_score'].hist()
st.pyplot()







def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'
    
news['polarity']=news['polarity_score'].\
   map(lambda x: sentiment(x))

st.header('Polarity Count')
plt.bar(news.polarity.value_counts().index,
        news.polarity.value_counts())
st.pyplot()



st.header('Positive Statements')
st.table(news[news['polarity']=='pos']['headline_text'].head())



st.header('Negative Statements')
st.table(news[news['polarity']=='neg']['headline_text'].head())

st.header('Neutral Statements')
st.table(news[news['polarity']=='neu']['headline_text'].head())








from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):
    # Polarity score returns dictionary
    ss = sid.polarity_scores(sent)
    #return ss
    return np.argmax(list(ss.values())[:-1])

news['polarity']=news['headline_text'].\
    map(lambda x: get_vader_score(x))
polarity=news['polarity'].replace({0:'neg',1:'neu',2:'pos'})

st.header('Polarity Count')
plt.bar(polarity.value_counts().index,
        polarity.value_counts())
st.pyplot()

# NER
#pip install spacy
#python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load("en_core_web_sm")

doc=nlp('India and Iran have agreed to boost the economic viability \
of the strategic Chabahar port through various measures, \
including larger subsidies to merchant shipping firms using the facility, \
people familiar with the development said on Thursday.')

#st.write([(x.text,x.label_) for x in doc.ents])

from spacy import displacy
displacy.render(doc, style='ent')
st.pyplot()





def ner(text):
    doc=nlp(text)
    return [X.label_ for X in doc.ents]

ent=news['headline_text'].\
    apply(lambda x : ner(x))
ent=[x for sub in ent for x in sub]

counter=collections.Counter(ent)
count=counter.most_common()

st.header('Entity frequencies')
x,y=map(list,zip(*count))
sns.barplot(x=y,y=x)
st.pyplot()




def ner(text,ent="GPE"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]

gpe=news['headline_text'].apply(lambda x: ner(x))
gpe=[i for x in gpe for i in x]
counter=collections.Counter(gpe)

x,y=map(list,zip(*counter.most_common(10)))
st.header('Common tokens per entity')
sns.barplot(y,x)
st.pyplot()





per=news['headline_text'].apply(lambda x: ner(x,"PERSON"))
per=[i for x in per for i in x]
counter=Counter(per)

x,y=map(list,zip(*counter.most_common(10)))
st.header('Common Names in data')
sns.barplot(y,x)
st.pyplot()







def get_adjs(text):
    adj=[]
    pos=nltk.pos_tag(word_tokenize(text))
    for word,tag in pos:
        if tag=='NN':
            adj.append(word)
    return adj


words=news['headline_text'].apply(lambda x : get_adjs(x))
words=[x for l in words for x in l]
counter=collections.Counter(words)

x,y=list(map(list,zip(*counter.most_common(7))))
st.header('Top singular words')
sns.barplot(x=y,y=x)
st.pyplot()