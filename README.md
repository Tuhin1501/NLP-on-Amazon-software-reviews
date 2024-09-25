# NLP-on-Amazon-software-reviews

import numpy as np
import pandas as pd
import plotly as pt
import plotly.express as px
import re
import string
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_json(r"C:\Users\t1u5h\Downloads\Software_5.json.gz", lines=True)
df = df.drop(columns=["image","style","vote"])
df = df.drop(columns="reviewerName")
df.isna().sum()
df['reviewText']=df['reviewText'].fillna('Missing')
df['summary']=df['summary'].fillna('Missing')
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df["year"] = df["reviewTime"].dt.year
df["month"] = df["reviewTime"].dt.month
df["overall"] = df["overall"].astype('category')
df = df.drop(columns="unixReviewTime")
def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['reviewText']=df['reviewText'].apply(lambda x:review_cleaning(x))
df['summary']=df['summary'].apply(lambda x:review_cleaning(x))
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df = df.reset_index()
df = df.drop(columns="index")
df_verification = df.groupby(['verified'])['verified'].count().reset_index(name='count')
fig = px.bar(df_verification, x="verified", y="count", color="verified", title='Number of true vs false verified reviews')
fig.update_traces(dict(marker_line_width=0))
iplot(fig)

df_verification = df.groupby(['verified','overall'])['verified'].count().reset_index(name='count')
fig = px.bar(df_verification, 
             y='verified',
             x='count',
             color='verified',orientation='h',  
             title='Verification based on overall rating',
             barmode='group', facet_col='overall'
             )
iplot(fig)

df_overall = df.groupby(['overall'])['overall'].count().reset_index(name='count')
fig = px.bar(df_overall, x="overall", y="count", color="overall", title='rating stars comparison')
fig.update_traces(dict(marker_line_width=0))
iplot(fig)

df["overall"].value_counts(normalize=True)


df_product = df.groupby(['asin']).size().to_frame().sort_values([0], ascending = False).head(10).reset_index()
df_product.columns = ['asin', 'count']
fig = px.bar(df_product, x='asin', y = 'count', color = "asin", title='Top 10 most reviewed products')
fig.layout.yaxis.title.text = 'frequency'
fig.update_layout(showlegend=False)
iplot(fig)

df_B000EORV8Q=df[(df["asin"] == "B000EORV8Q")]
frequent_words = top_n_ngram(df_B000EORV8Q['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])
df_frequent_words
df_B000EORV8Q['reviewText'] network router

df_B0001FS9NE=df[(df["asin"] == "B0001FS9NE")]
frequent_words = top_n_ngram(df_B0001FS9NE['reviewText'], 10,1)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])
df_frequent_words
df_B0001FS9NE["reviewText"][642]  netgear switch


df_B000050ZRE=df[(df["asin"] == "B000050ZRE")]
frequent_words = top_n_ngram(df_B000050ZRE['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])
df_frequent_words
df_B000050ZRE.head(50)


df_B0000AZJY6=df[(df["asin"] == "B0000AZJY6")]
frequent_words = top_n_ngram(df_B0000AZJY6['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])
df_frequent_words
df_B0000AZJY6 PCI-USB card


df_B0000AZJY6=df[(df["asin"] == "B0000AZJY6")]
frequent_words = top_n_ngram(df_B0000AZJY6['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])
df_frequent_words

df['sentiment'] = pd.cut(x=df['overall'], bins=[0,2,3,5], 
                     labels=['Negative','Average','Positive'])

df_sentiment_year = df.groupby(['sentiment','year'])['sentiment'].count().reset_index(name='count')
df_sentiment_year = df_sentiment_year[(df_sentiment_year["year"]>2010)]
fig = px.line(df_sentiment_year, x="year", y="count", title='Sentiment count by year',labels={'count': 'Sentiment count'}, color="sentiment")
iplot(fig)

df_sentiment_month = df.groupby(['sentiment','month'])['sentiment'].count().reset_index(name='count')
fig = px.bar(df_sentiment_month, x="month", y="count",barmode='group', title='Sentiment count by month',labels={'count': 'Sentiment count'}, color="sentiment")
iplot(fig)

df_month = df.groupby(['month'])['month'].count().reset_index(name='count')
df_month["month"] = df_month["month"].astype('category')
fig = px.bar(df_month, x="month", y="count", color="month", title="Number of reviews classified by months")
fig.update_traces(dict(marker_line_width=0))
fig.update_layout(showlegend=False)
iplot(fig)


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

df_sentiment = df

longest_review = df_sentiment['reviewText'].str.len().idxmax()
longest_review_text = df_sentiment.loc[longest_review, 'reviewText']

print(f"Customer review with the longest length (index): {longest_review}")
print(f"Length of the longest review: {len(longest_review_text)}")
print(f"Longest review text:\n{longest_review_text}")

example = df_sentiment["reviewText"][5210]

tokens = nltk.word_tokenize(example)

tagged = nltk.pos_tag(tokens)

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

sia.polarity_scores(example)

df_sentiment = df_sentiment.reset_index()

res={}
for i, row in tqdm(df_sentiment.iterrows(), total=len(df_sentiment)):
    review = row["reviewText"]
    verification = row["index"]
    res[verification] = sia.polarity_scores(review)

res

results = pd.DataFrame(res).T

results = results.reset_index()

results

df_sentiment_res = pd.merge(results, df_sentiment, on = "index", how = "left")

df_sentiment_res

df_sentiment_res = df_sentiment_res.drop(columns="index")

fig = px.histogram(df_sentiment_res, x = "compound", color_discrete_sequence=px.colors.qualitative.Set3, title="Distribution of Compound scores")
iplot(fig)


df_sentiment_res['vader_sentiment'] = pd.cut(x=df_sentiment_res['compound'], bins=[-1,-0.5,0,1], 
                     labels=['Negative','Average','Positive'])

df_vader_sentiment = df_sentiment_res.groupby(['vader_sentiment'])['vader_sentiment'].count().reset_index(name='count')
fig = px.pie(df_vader_sentiment, values='count', names='vader_sentiment', title='Sentiment based on Vader results')
iplot(fig)

df_overall_sentiment = df_sentiment_res.groupby(['sentiment'])['sentiment'].count().reset_index(name='count')
fig = px.pie(df_overall_sentiment, values='count', names='sentiment', title='Sentiment based on overall results')
iplot(fig)

plt.figure(figsize=(8, 6))  # Adjust figure size as desired

# Create the pie chart using Seaborn
plt.pie(data=df_vader_sentiment,
    x="count",
    labels="vader_sentiment",
    autopct="%1.1f%%",  # Display percentage with one decimal place
    startangle=140, 
        colors=['tomato', 'gold', 'cornflowerblue']
)

# Add a title
plt.title("Sentiment based on Vader results")

# Show the plot
plt.show()

plt.figure(figsize=(8, 6))  # Adjust figure size as desired

# Create the pie chart using Seaborn
plt.pie(data=df_overall_sentiment,
    x="count",
    labels="sentiment",
    autopct="%1.1f%%",  # Display percentage with one decimal place
    startangle=140,  # Rotate starting angle (optional)
        colors=['tomato', 'gold', 'cornflowerblue']
)


# Add a title
plt.title("Sentiment based on overall results")

# Show the plot
plt.show()

from nltk.corpus import stopwords
from collections import Counter
from nltk.util import bigrams


reviews = df_sentiment_res['reviewText']

# Convert the reviews to a list of words
words = []
for review in reviews:
    words.extend(review.split())

# Filter out the stubborn stop words that don't want to disappear
stopwords = nltk.corpus.stopwords.words('english')
new_words=['Like', 'good', 'great', 'we', 'i', 'us', 'amazing', 'taste', 'like','one','product','use','really','would'
                ,'price','dont','buy','get','much','also','ive','used' ,'tried','eat','bag', 'got', 'amazon', 'could'
                , 'didnt', 'im', 'even','using','easy','years','well','need']
stopwords.extend(new_words)
key_words = [word for word in words if word not in stopwords and word not in ['Like', 'good', 'great', 'we', 'i', 'us', 'amazing'
                                                                             , 'taste', 'like','one','product','use','really','would'
                                                                             ,'price','dont','buy','get','much','also','ive','used'
                                                                             'tried','eat','bag']]

# Get the top 5 most used key words
word_counts = Counter(key_words)
top_10_words = word_counts.most_common(10)


# Get the labels and values for the histogram
labels, values = zip(*top_10_words)
labels = labels[::-1]
values = values[::-1]


fig = px.bar(y=labels,x=values,color=labels, title='Top 10 words from reviews')
fig.layout.yaxis.title.text = 'Words'
fig.layout.xaxis.title.text = 'Frequency'
fig.update_layout(showlegend=False)
iplot(fig)


df_filtered_pos = df_sentiment_res[(df_sentiment_res["sentiment"] == "Positive")]
reviews_pos = df_filtered_pos['reviewText']


# Convert the reviews to a list of words
words = []
for review in reviews_pos:
    words.extend(review.split())

# Filter out the stubborn stop words that don't want to disappear
stopwords = nltk.corpus.stopwords.words('english')
new_words=['Like', 'good', 'great', 'we', 'i', 'us', 'amazing', 'taste', 'like','one','product','use','really','would'
                ,'price','dont','buy','get','much','also','ive','used' ,'tried','eat','bag', 'got', 'amazon', 'could'
                , 'didnt', 'im', 'even','find','add','try','years','well','need','want','user','still','make','year','lot','way',"using"]
stopwords.extend(new_words)
key_words = [word for word in words if word not in stopwords and word not in ['Like', 'good', 'great', 'we', 'i', 'us', 'amazing'
                                                                             , 'taste', 'like','one','product','use','really','would'
                                                                             ,'price','dont','buy','get','much','also','ive','used'
                                                                             'tried','eat','bag']]

# Get the top 5 most used key words
word_counts = Counter(key_words)
top_20_words = word_counts.most_common(20)


# Get the labels and values for the histogram
labels, values = zip(*top_20_words)


plt.figure(figsize=(12, 12))
sns.set(color_codes=True)
plt.rcParams.update({'font.size': 14})
sns.barplot(x=values, y=labels, hue=labels, palette="Blues_d")  # Replace with your variable
plt.title('Top 20 positive words from reviews',fontsize=16)
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.tick_params(axis='y', which='major', labelsize=16)
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


df_filtered_neu = df_sentiment_res[(df_sentiment_res["sentiment"] == "Average")]
reviews_neu = df_filtered_neu['reviewText']

# Convert the reviews to a list of words
words = []
for review in reviews_neu:
    words.extend(review.split())

# Filter out the stubborn stop words that don't want to disappear
stopwords = nltk.corpus.stopwords.words('english')
new_words=['Like', 'good', 'great', 'we', 'i', 'us', 'amazing', 'taste', 'like','one','product','use','really','would'
                ,'price','dont','buy','get','much','also','ive','used' ,'tried','eat','bag', 'got', 'amazon', 'could'
                , 'didnt', 'im', 'even','find','add','try','years','well','need','want','user','still','make','year','lot','way','using','file']
stopwords.extend(new_words)
key_words = [word for word in words if word not in stopwords and word not in ['Like', 'good', 'great', 'we', 'i', 'us', 'amazing'
                                                                             , 'taste', 'like','one','product','use','really','would'
                                                                             ,'price','dont','buy','get','much','also','ive','used'
                                                                             'tried','eat','bag']]

# Get the top 5 most used key words
word_counts = Counter(key_words)
top_20_words = word_counts.most_common(20)


# Get the labels and values for the histogram
labels, values = zip(*top_20_words)


plt.figure(figsize=(12, 12))
sns.set(color_codes=True)
plt.rcParams.update({'font.size': 14})
sns.barplot(x=values, y=labels, hue=labels, palette="Purples_d")  # Replace with your variable
plt.title('Top 20 neutral words from reviews',fontsize=16)
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.tick_params(axis='y', which='major', labelsize=16)
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

df_filtered_neg = df_sentiment_res[(df_sentiment_res["sentiment"] == "Negative")]
reviews_neg = df_filtered_neg['reviewText']

# Convert the reviews to a list of words
words = []
for review in reviews_neg:
    words.extend(review.split())

# Filter out the stubborn stop words that don't want to disappear
stopwords = nltk.corpus.stopwords.words('english')
new_words=['Like', 'good', 'great', 'we', 'i', 'us', 'amazing', 'taste', 'like','one','product','use','really','would'
                ,'price','dont','buy','get','much','also','ive','used' ,'tried','eat','bag', 'got', 'amazon', 'could'
                , 'didnt', 'im', 'even','find','add','try','years','well','need','want','user','still','make','year','lot','way',"back",
          "quicken"]
stopwords.extend(new_words)
key_words = [word for word in words if word not in stopwords and word not in ['Like', 'good', 'great', 'we', 'i', 'us', 'amazing'
                                                                             , 'taste', 'like','one','product','use','really','would'
                                                                             ,'price','dont','buy','get','much','also','ive','used'
                                                                             'tried','eat','bag']]

# Get the top 5 most used key words
word_counts = Counter(key_words)
top_20_words = word_counts.most_common(20)


# Get the labels and values for the histogram
labels, values = zip(*top_20_words)


plt.figure(figsize=(12, 12))
sns.set(color_codes=True)
plt.rcParams.update({'font.size': 14})
sns.barplot(x=values, y=labels, hue=labels, palette="Reds_d")  # Replace with your variable
plt.title('Top 20 negative words from reviews',fontsize=16)
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.tick_params(axis='y', which='major', labelsize=16)
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
stopwords = nltk.corpus.stopwords.words('english')

def top_n_ngram(corpus,n = None,ngram = 1):
    vec = CountVectorizer(stop_words = 'english',ngram_range=(ngram,ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus) #Have the count of  all the words for each review
    sum_words = bag_of_words.sum(axis =0) #Calculates the count of all the word in the whole review
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,key = lambda x:x[1],reverse = True)
    return words_freq[:n]

fig = px.bar(df_frequent_words, y="reviewText",x="count",color="reviewText", title='Top 5 bigram words from reviews')
fig.layout.yaxis.title.text = 'Words'
fig.layout.xaxis.title.text = 'Frequency'
fig.update_layout(showlegend=False)
fig.update_layout(width=800, height=800)
iplot(fig)

df_filtered_pos = df_sentiment_res[(df_sentiment_res["sentiment"] == "Positive")]
frequent_words = top_n_ngram(df_filtered_pos['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])

plt.figure(figsize=(12, 12))
sns.set(color_codes=True)

sns.barplot(df_frequent_words,x="count", y="reviewText", hue="reviewText", palette="Blues_d")  # Replace with your variable
plt.title('Top 20 positive words from reviews')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


df_filtered_neu = df_sentiment_res[(df_sentiment_res["sentiment"] == "Average")]
frequent_words = top_n_ngram(df_filtered_neu['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])

plt.figure(figsize=(12, 12))
sns.set(color_codes=True)

sns.barplot(df_frequent_words,x="count", y="reviewText", hue="reviewText", palette="Purples_d")  # Replace with your variable
plt.title('Top 20 neutral words from reviews')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

df_filtered_neg = df_sentiment_res[(df_sentiment_res["sentiment"] == "Negative")]
frequent_words = top_n_ngram(df_filtered_neg['reviewText'], 10,2)
df_frequent_words = pd.DataFrame(frequent_words, columns = ['reviewText' , 'count'])

plt.figure(figsize=(12, 12))
sns.set(color_codes=True)

sns.barplot(df_frequent_words,x="count", y="reviewText", hue="reviewText", palette="Reds_d")  # Replace with your variable
plt.title('Top 20 negative words from reviews')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.xticks(rotation=0)  # Optional: Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

df_review_check = df_sentiment_res[["reviewText","sentiment",'vader_sentiment']]

comparison_column = np.where(df_review_check["sentiment"] == df_review_check["vader_sentiment"], True, False)
df_review_check["result"] = comparison_column
df_review_check.head()

df_check = df_review_check.groupby(['result'])['result'].count().reset_index(name='count')
fig = px.pie(df_check, values='count', names='result', title='Percentage of genuine vs fake reviews')
iplot(fig)

ml_df = df_sentiment_res[["reviewText","sentiment"]]

ml_df["sentiment"].value_counts()

ml_df.shape

y = ml_df[["sentiment"]]

x = ml_df.drop(columns="sentiment")

from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

tf=TfidfVectorizer()
x_updated= tf.fit_transform(x['reviewText'])

x_train,x_test,y_train,y_test = train_test_split(x_updated,y,test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(max_depth=200, max_samples= 0.95, min_samples_split=0.009,
                       random_state=42)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test,y_pred,labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = rf.classes_)
disp.plot()

dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=50, random_state=42)

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

lg = LogisticRegression(solver='newton-cg')

lg.fit(x_train,y_train)

y_pred = lg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test,y_pred,labels=lg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = lg.classes_)
disp.plot()


 
