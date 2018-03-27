import numpy as np
import pandas as pd
import time
import re
import string

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook

from nltk.stem.porter import *
from nltk import tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans


# ==========================================================================
# LOAD FILE HERE
# ==========================================================================    
path = "D:/Kaggle/Python/MercariPricing/"
train = pd.read_csv(f'{path}train.tsv', sep='\t')
test = pd.read_csv(f'{path}test.tsv', sep='\t') 
    

# ==========================================================================
# FUNCTIONS HERE
# ==========================================================================    
# SPLIT CATEGORY_NAME INTO 3 SUBCAT
def split_category_name(input):
    try:
        return(input.split("/"))
    except:
        return("no label", "no label", "no label")
            
# CLEAN TEXT, REMOVING STOP-WORDS
def clean_text(text):
    try:
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        words = [w for w in txt.split(" ") if w not in stop_words.ENGLISH_STOP_WORDS and len(w)>2]
        words = " ".join(words)
        return (words)
    except:
        return(0)


# CLEAN TEXT, E.G. LOWER, REMOVE STOP WORDS, NUMBER, PUNTUATION ...
def wordCount(text):
    try:
#        text = text.lower()
#        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
##        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
#        txt = regex.sub(" ", text)
#        # REMOVE STOP WORD
#        words = [w for w in txt.split(" ") if w not in stop_words.ENGLISH_STOP_WORDS and len(w)>2]
        words = clean_text(text)
        return (len(words.split(" ")))
    except:
        return(0)


# CLEAN TEXT, THEN TOKENIZE
def wordTokenize(text):
    try:
#        text = text.lower()
#        regex = re.compile('['+re.escape(string.punctuation)+'0-9\\r\\t\\n]')
#        txt = regex.sub(" ",text)
#        words = [w for w in txt.split(" ") if w not in stop_words.ENGLISH_STOP_WORDS]
#        words = word_tokenize(" ".join(words))
        words = clean_text(text)
        return(word_tokenize(words))
    except:
        return(0)


# ANOTHER WAY TO TOKENIZE TEXT
stop = set(stopwords.words('english'))
def tokenize(text):
    try:
        regex = re.compile('['+re.escape(string.punctuation)+'0-9\\r\\t\\n]')
        text = regex.sub(" ",text)        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []        
        for token_by_sent in tokens_:
            tokens += token_by_sent        
        tokens = list(filter(lambda x: x.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>2]        
        return(filtered_tokens)
    except:
        return(0)    


# ==========================================================================
# EXPLORATION
# ==========================================================================    
print(train.shape)
print(test.shape)
train.head(3)

train.price.describe()
train.shipping.describe()

# PLOT THE PRICE HISTOGRAM
plt.subplot(1,2,1)
train.price.plot.hist(bins=50, edgecolor='white', range=[0,250])
plt.subplot(1,2,2)
np.log1p(train.price).plot.hist(bins=50, edgecolor='white')

# LOOKING AT HOW MANY ITEM NEED TO PAY SHIPPING
train.shipping.value_counts()/len(train.shipping)

price_with_shipping = train.loc[train.shipping==0, 'price']
price_no_shipping = train.price.loc[train.shipping==1]

fig, ax = plt.subplots()
ax.hist(np.log1p(price_with_shipping), color = 'blue', alpha= 0.5, bins=50,
        edgecolor='white')
ax.hist(np.log1p(price_no_shipping), color = 'red', alpha= 0.5, bins=50,
        edgecolor='white')
ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')

# CHECK TOP 5 RAW CATEGORY
train.category_name.value_counts()[:5]
train.category_name.isnull().sum()

# SPLIT CATEGORY_NAME INTO 3 SUB_CAT
train['subcat0'], train['subcat1'], train['subcat2'] = zip(*train.category_name.apply(lambda x: split_category_name(x)))
test['subcat0'], test['subcat1'], test['subcat2'] = zip(*test.category_name.apply(lambda x: split_category_name(x)))
train.subcat0.nunique()
train.subcat0.unique()
train.subcat0.value_counts()
train.subcat1.nunique()
train.subcat2.nunique()

# PLOT SUBCAT0
x = train.subcat0.value_counts().index.values.astype('str')
y = train.subcat0.value_counts().values
pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of Items by Main Category',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'))
fig=dict(data=[trace1], layout=layout)
py.iplot(fig)
py.plot(fig)

# PLOT PRICE DISTRIBUTION BY SUB-CATEGORY 0
subcat0_var = train.subcat0.unique()
trace1 = [go.Box(x=np.log1p(x[i]), name=subcat0_var[i]) for i in range(len(subcat0_var))]
layout = dict(title="Price Distribution by Sub-category 0",
              yaxis=dict(title="category"),
              xaxis=dict(title="log(price + 1)"))
fig = dict(data=trace1, layout=layout)
py.plot(fig)

# PLOTSUBCAT1
x = train.subcat1.value_counts().index.values.astype('str')
y = train.subcat1.value_counts().values
pct = [("%.2f"%(v*100)) + "%" for v in (y/len(train))]
percent = [v for v in y/len(train)*100]

trace1 = go.Bar(x=x[:15], y=y[:15], text=pct,
                marker=dict(color=percent, colorscale='Portland', 
                            showscale=True, reversescale=False))
layout = dict(title = 'Number of items by sub_cat1', 
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'),
              )
fig = dict(data=[trace1], layout=layout)
py.plot(fig)

# ==========================
# BRAND NAME
train.brand_name.nunique()
x = train.brand_name.value_counts().index.values.astype('str')
y = train.brand_name.value_counts().values
pct = [("%.2f"%(v*100)) + "%" for v in (y/len(train))]

trace1 = go.Bar(x=x[:10], y=y[:10], text=pct)
fig = dict(data=[trace1])
py.plot(fig)


# ADD ITEM DESCRIPTION INTO A DATA FRAME
train['item_desc_len'] = train.item_description.apply(lambda x: wordCount(x))
test['item_desc_len'] = test.item_description.apply(lambda x: wordCount(x))

df = train.groupby('item_desc_len')['price'].mean().reset_index()
trace1 = go.Scatter(x = df['item_desc_len'],
                    y = np.log(df['price']+1),
                    mode = 'lines+markers',
                    name = 'lines+markers')
layout = dict(title = "mean log(price+1) by description length",
              xaxis = dict(title="Description Length"),
              yaxis = dict(title="Mean log(price+1)"))
fig = dict(data = [trace1], layout = layout)
py.plot(fig)

# ==========================
# ITEM DESCRIPTION TREATMENT 
train.item_description.isnull().sum()
train.loc[train.item_description.isnull(),'item_description'] = 'No description yet'
test.loc[test.item_description.isnull(),'item_description'] = 'No description yet'

# NUMBER OF WORDS (WORD FREQUENCY) FOR EACH CATEGORY
cat_desc = dict()
for cat in subcat0_var:
    text = " ".join(train.loc[train.subcat0==cat, 'item_description'].values)
    cat_desc[cat] = wordTokenize(text)

#cat_desc1 = [wordTokenize(" ".join(train.loc[train.subcat0==w,'item_description'])) for w in subcat0_var] 

# FLAT LIST OF ALL WORDS COMBINED
flat_list = [item for sublist in list(cat_desc.values()) for item in sublist]
allWordsCount = Counter(flat_list)
all_top10 = allWordsCount.most_common(20)

x = [w[0] for w in all_top10]
y = [w[1] for w in all_top10]
pct = [("%.2f"%(y[i]/len(flat_list)*100)) + "%" for i in range(len(y))]
trace1 = go.Bar(x=x, y=y, text = pct)
layout = dict(title="Word Frequency",
              xaxis = dict(title="Word"),
              yaxis = dict(title="Count"))
fig = dict(data = [trace1], layout = layout)
py.plot(fig)

# ==========================
# TOKENIZE ITEM_DESCRIPTION
train['tokens'] = train.item_description.map(tokenize)
test['tokens'] = test.item_description.map(tokenize)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# DOUBLE CHECK OUR TOKENIZATION
for desc, tokens in zip(train.item_description.head(), train.tokens.head()):
    print(desc)
    print(tokens)
    print(" ")

# ==========================
# WORD-CLOUND
cat_desc = dict()
for cat in subcat0_var:
    text = " ".join(train.loc[train.subcat0==cat, 'item_description'].values)
    cat_desc[cat] = tokenize(text)

women100 = Counter(cat_desc['Women']).most_common(100)
kids100 = Counter(cat_desc['Kids']).most_common(100)

fig, axes = plt.subplots(2,2,figsize=(30,15))
ax = axes[0,0]
ax.imshow(WordCloud().generate(str(women100)), interpolation='bilinear')
ax.set_title("Women Top 100", fontsize=30)

ax = axes[1,0]
ax.imshow(WordCloud().generate(str(kids100)), interpolation='bilinear')
ax.set_title("Kids Top 100", fontsize=30)

# ==========================
vectorizer = TfidfVectorizer(min_df=10, max_features=100000, tokenizer=tokenize,
                             ngram_range=(1,2))
all_desc = np.append(train.item_description.values, test.item_description.values)
item_desc_vectorize = vectorizer.fit_transform(all_desc)

# CREATE DICT MAPPING THE TOKENS TO THEIR TFIDF VALUES
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')

tfidf.columns = ['tfidf']
tfidf.sort_values(by=['tfidf'], ascending=False).head(10)

# ==========================
# USE SVD TO REDUCE THE DATA DIMENSION
train_copy = train.copy()
test_copy = test.copy()
train['is_train'] = 1
test['is_train'] = 0
sample_size = 15000
df_combined = pd.concat([train,test])
sample_combined = df_combined.sample(n=sample_size)
vectorize_sample = vectorizer.fit_transform(sample_combined.item_description)

n_comp= 30
svd = TruncatedSVD(n_components=n_comp, random_state=68)
svd_tfidf = svd.fit_transform(vectorize_sample)

# t-SNE
tsne_model = TSNE(n_components = 2, verbose=1, random_state=68, n_iter=500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

sample_combined.reset_index(inplace=True, drop=True)
df_tfidf = pd.DataFrame(tsne_tfidf, columns=['x','y'])
df_tfidf['description'] = sample_combined['item_description']
df_tfidf['tokens'] = sample_combined['tokens']
df_tfidf['category'] = sample_combined['subcat0']

#output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                       title="tf-idf clustering of the item description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

plot_tfidf.scatter(x='x', y='y', source=df_tfidf, alpha=0.7)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "tokens": "@tokens", "category":"@category"}
show(plot_tfidf)

# ==========================
# K-MEANS
cluster_no = 30
kmeans_model = MiniBatchKMeans(n_clusters = cluster_no, init = 'k-means++',
                               n_init = 1, init_size = 1000, batch_size = 1000,
                               verbose = 1, max_iter = 1000)

kmeans = kmeans_model.fit(item_desc_vectorize)
kmeans_clusters = kmeans.predict(item_desc_vectorize)
kmeans_distances = kmeans.transform(item_desc_vectorize)

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# FOR PLOTTING, PICKING UP 15000 SAMPLES
kmeans = kmeans_model.fit(vectorize_sample)
kmeans_clusters = kmeans.predict(vectorize_sample)
kmeans_distances = kmeans.transform(vectorize_sample)

# THEN USING TSNE TO REDUCE TO 2 DIMENSIONS FOR PLOTTING
tsne_model = TSNE(n_components = 2, verbose=1, random_state=68, n_iter=500)
tsne_kmeans = tsne_model.fit_transform(kmeans_distances)

colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])

df_kmeans = pd.DataFrame(tsne_kmeans, columns=['x','y'])
df_kmeans['clusters'] = kmeans_clusters
df_kmeans['description'] = sample_combined.item_description.values
df_kmeans['category'] = sample_combined.subcat0.values

plot_kmeans = bp.figure(plot_width=700, plot_height=600,
                        title="KMeans clustering of the description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

source = ColumnDataSource(data=dict(x=df_kmeans['x'], y=df_kmeans['y'],
                                    color=colormap[kmeans_clusters],
                                    description=df_kmeans['description'],
                                    category=df_kmeans['category'],
                                    cluster=df_kmeans['clusters']))

plot_kmeans.scatter(x='x', y='y', color='color', source=source)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster" }
show(plot_kmeans)

# ========================================================
# LATENT DIRICHLET ALLOCATION
cvectorizer = CountVectorizer(min_df=4, max_features=180000, tokenizer=tokenize,
                              ngram_range=(1,2))

cvz = cvectorizer.fit_transform(sample_combined.item_description)

lda_model = LatentDirichletAllocation(n_components=20,
                                      learning_method='online',
                                      max_iter=20,
                                      random_state=42)

x_topics = lda_model.fit_transform(cvz)

n_top_words = 10
topic_summaries = []

topic_word = lda_model.components_  # get the topic words
vocab = cvectorizer.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))

tsne_lda = tsne_model.fit_transform(x_topics)
unnormalized = np.matrix(x_topics)
doc_topic = unnormalized/unnormalized.sum(axis=1)

lda_keys = []
for i, tweet in enumerate(sample_combined['item_description']):
    lda_keys += [doc_topic[i].argmax()]

lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['description'] = sample_combined['item_description']
lda_df['category'] = sample_combined['subcat0']
lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)

plot_lda = bp.figure(plot_width=700,
                     plot_height=600,
                     title="LDA topic visualization",
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

source = ColumnDataSource(data=dict(x=lda_df['x'], y=lda_df['y'],
                                    color=colormap[lda_keys],
                                    description=lda_df['description'],
                                    topic=lda_df['topic'],
                                    category=lda_df['category']))

plot_lda.scatter(source=source, x='x', y='y', color='color')
hover = plot_kmeans.select(dict(type=HoverTool))
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description",
                "topic":"@topic", "category":"@category"}
show(plot_lda)





































































