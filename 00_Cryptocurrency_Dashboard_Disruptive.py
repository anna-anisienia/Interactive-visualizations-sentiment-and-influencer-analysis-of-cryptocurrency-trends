import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go
import re
import numpy as np
import pandas as pd
import requests
import json
from textblob import TextBlob
from sqlalchemy import create_engine
from sqlalchemy.types import String, Integer, Float, Boolean, DateTime
# from dashapp import server as application
app = dash.Dash(__name__)

# if you want to implement USER and PASSWORD, please uncomment the following 3 lines of code:
# from dash.dependencies import Input, Output
# USERNAME_PASSWORD_PAIRS = [['bipm', 'crypto']]
# auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)

#_____________________________________________________________________________________________________
# Get the streamed data from REDDIT and show the sentiment over time + aggregated sentiment + comparison bw. BTC and ETH
#_____________________________________________________________________________________________________
# REDDIT preprocessing
#_____________________________________________________________________________________________________

engine = create_engine('postgresql://username:password@server:port/database_name')
connection = engine.connect()
btc_reddit = pd.read_sql(sql = "select distinct title, created_utc, \"SA_score_grouped\", \"SA_score\" from btc_reddit order by created_utc desc", con = connection, index_col=None)
eth_reddit = pd.read_sql(sql = "select distinct title, created_utc, \"SA_score_grouped\", \"SA_score\" from eth_reddit order by created_utc desc", con = connection, index_col=None)

# to later display the interactive data table
reddit = pd.concat([btc_reddit, eth_reddit], axis=0, join='outer', # to get UNION of rows, instead of intersection
          join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
reddit.SA_score = round(reddit.SA_score,2)

# Aggregated sentiment on Reddit
btc_grouped = btc_reddit[["SA_score", "SA_score_grouped"]].groupby("SA_score_grouped").count()
btc_grouped["sentiment"] = btc_grouped.index
btc_grouped.reset_index(drop=True, inplace=True)
btc_grouped.rename(columns={"SA_score": "nr_of_tweets"}, inplace=True)

eth_grouped = eth_reddit[["SA_score", "SA_score_grouped"]].groupby("SA_score_grouped").count()
eth_grouped["sentiment"] = eth_grouped.index
eth_grouped.reset_index(drop=True, inplace=True)
eth_grouped.rename(columns={"SA_score": "nr_of_tweets"}, inplace=True)

#_____________________________________________________________________________________________________
# BTC and ETH values over time - Preprocessing
#_____________________________________________________________________________________________________

url = 'https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=365'
r = requests.get(url) # Decode the JSON data into a dictionary: json_data
json_data = r.json()
btc_values_df = pd.DataFrame(json_data["Data"]) # dictionary of currency values is stored as a list under the key "Data"
btc_values_df["timestamp"] = pd.to_datetime(btc_values_df["time"], unit='s') # clean the date format: up to seconds, without miliseconds

url = 'https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USD&limit=365'
r = requests.get(url)
json_data = r.json() # Decode the JSON data into a dictionary: json_data
eth_values_df = pd.DataFrame(json_data["Data"])
eth_values_df["timestamp"] = pd.to_datetime(eth_values_df["time"], unit='s') # clean the date format: converts the unix timestamp to pandas date data type

#_____________________________________________________________________________________________________
# Get the streamed data from TWITTER and show the sentiment over time + aggregated sentiment + comparison bw. BTC and ETH
#_____________________________________________________________________________________________________
# Twitter preprocessing
#_____________________________________________________________________________________________________

twitter_df = pd.read_sql(sql = "select distinct text, tweet_created from twitter3 TABLESAMPLE SYSTEM(1) where text ~* '(btc|#eth|ether|bitcoin|ethereum)' order by tweet_created desc", con = connection, index_col=None)

list_of_tweets = twitter_df.text.tolist()

# even though the data has been cleaned directly throgh SQL query, we use RegExp to separate tweets related to BTC and ETH
eth_tweets = [tweet for tweet in list_of_tweets if \
              len(re.findall(r"(ethereum|Ethereum|ETH|ETC|Ethereum Classic|EthereumClassic|ether|eth)", tweet)) > 0]

btc_tweets = [tweet for tweet in list_of_tweets if \
              len(re.findall(r"(bitcoin|Bitcoin|BTC|BitCoin|bitCoin|BitcoinClassic|Bitcoin Classic|bitcoinclassic|bitcoinClassic|XBT)", tweet)) > 0]

# Twitter Sentiment analysis
def get_sentiment(sentence):
    analysis = TextBlob(sentence)
    return(round(analysis.sentiment.polarity, 2)) # > 0 positive, < 0 negative

btc_twitter_sa = [get_sentiment(sentence) for sentence in btc_tweets]
twitter_btc_df = pd.DataFrame({"text":btc_tweets, "SA_score":btc_twitter_sa})
twitter_btc_df = pd.merge(twitter_btc_df, twitter_df, how='inner', on="text")

eth_twitter_sa = [get_sentiment(sentence) for sentence in eth_tweets]
twitter_eth_df = pd.DataFrame({"text":eth_tweets, "SA_score":eth_twitter_sa})
twitter_eth_df = pd.merge(twitter_eth_df, twitter_df, how="inner", on="text")

# to later display the interactive data table
twitter = pd.concat([twitter_btc_df, twitter_eth_df], axis=0, join='outer', # to get UNION of rows, instead of intersection
          join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)

#  Remove duplicates for detailed sentiment plot
twitter_btc_df2 = twitter_btc_df.drop_duplicates(subset = ["text"], keep='first', inplace=False)
twitter_eth_df2 = twitter_eth_df.drop_duplicates(subset = ["text"], keep='first', inplace=False)

# Function for aggregated sentiment
def get_short_sentiment(sentence):
    '''function to classify sentiment of passed SA score'''
    if sentence > 0.05:
        return 'positive'
    elif sentence <= 0.05 and sentence > -0.005:
        return 'neutral'
    else:
        return 'negative'

short_twitter_btc = [get_short_sentiment(t) for t in twitter_btc_df.SA_score] # twitter_btc
short_twitter_eth = [get_short_sentiment(t) for t in twitter_eth_df.SA_score] # twitter_eth

twitter_btc_grouped = pd.DataFrame({"nr_of_tweets":twitter_btc_df.text, "short":short_twitter_btc}).groupby("short")
twitter_eth_grouped = pd.DataFrame({"nr_of_tweets":twitter_eth_df.text, "short":short_twitter_eth}).groupby("short")

twitter_btc_grouped = twitter_btc_grouped.count()
twitter_eth_grouped = twitter_eth_grouped.count()

twitter_btc_grouped["sentiment"] = twitter_btc_grouped.index
twitter_eth_grouped["sentiment"] = twitter_eth_grouped.index

twitter_btc_grouped.reset_index(drop=True, inplace=True)
twitter_eth_grouped.reset_index(drop=True, inplace=True)

#______________________________________________________________________________
# News preprocessing
#_____________________________________________________________________________________________________

ccn = pd.read_sql(sql = "select distinct article, date from ccn_articles order by date desc",
                         con = connection, index_col=None)

ccn_sa = [get_sentiment(sentence) for sentence in ccn.article]
ccn_sa_df = pd.DataFrame({"article":ccn.article, "SA_score":ccn_sa})
ccn_df = pd.merge(ccn_sa_df, ccn, how='inner', on="article")

short_ccn = [get_short_sentiment(t) for t in ccn_df.SA_score] # twitter_btc
ccn_grouped = pd.DataFrame({"nr_of_articles":ccn_df.article, "short":short_ccn}).groupby("short").count()
ccn_grouped["sentiment"] = ccn_grouped.index
ccn_grouped.reset_index(drop=True, inplace=True)

#______________________________________________________________________________
# Simple BoW model
#______________________________________________________________________________

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize

n = 10
def generate_word_list(text_col, nr_words = n):
    tokens = word_tokenize(text_col.to_string()) # tokenize
    lower_tokens = [t.lower() for t in tokens] # Convert the tokens into lowercase: lower_tokens
    alpha_only = [t for t in lower_tokens if t.isalpha()] # Retain alphabetic words: alpha_only
    stopwords = nltk.corpus.stopwords.words('english') # Remove all stop words: no_stops
    newStopWords = ["rt", "bitcoin", "crypto", "cryptocurrency", "blockchain", "blockcha", "btc", "bitcoi", "bitcoins", "daily", "say", "could",
                   "price", "ethereum", "eth", "classic", "exchange", "market", "cryptocurrencie", "one", "first", "short", "check",
                   "cryptocurrencies", "http", "htttp", "hour", "list", "u", "new", "vi", "ccn", "etc", "usd"]
    stopwords.extend(newStopWords)
    no_stops = [t for t in alpha_only if t not in stopwords]
    wordnet_lemmatizer = WordNetLemmatizer() # create instance of the WordNetLemmatizer class
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops if len(t)>1] # Lemmatize all tokens into a new list
    lemmatized = [t for t in lemmatized if t not in stopwords] # remove stopwords again after lemmatization
    bow = Counter(lemmatized) # Create the bag-of-words: bow
    word = []
    word_count = []
    for i in range(nr_words):
        word.append(bow.most_common(nr_words)[i][0])
        word_count.append(bow.most_common(nr_words)[i][1])
    words_and_counts_df = pd.DataFrame({"word":word, "word_count":word_count})
    return(words_and_counts_df) # return the n most common tokens

#______________________________________________________________________________
# Aggregate Sentiment by day
#______________________________________________________________________________
# Reddit
minDate = btc_reddit["created_utc"].min()
maxDate = btc_reddit["created_utc"].max()
ts_btc_reddit = btc_reddit.set_index("created_utc", inplace=False)
ts_btc_reddit = ts_btc_reddit.SA_score.resample('D').mean()
ts_eth_reddit = eth_reddit.set_index("created_utc", inplace=False)
ts_eth_reddit = ts_eth_reddit.SA_score.resample('D').mean()
standardized_reddit_scores = pd.DataFrame({'BTC':ts_btc_reddit,'ETH':ts_eth_reddit})
# Since the server might be down on certain days, we need to ensure that our time series has no discontinuities: interpolate() fills gaps of any size with a straight line
standardized_reddit_scores['BTC'].interpolate(method='linear', inplace=True)
standardized_reddit_scores['ETH'].interpolate(method='linear', inplace=True)

# Twitter
ts_twitter_btc_df = twitter_btc_df.set_index("tweet_created", inplace=False)
ts_twitter_btc_df = ts_twitter_btc_df.SA_score.resample('D').mean()
ts_twitter_eth_df = twitter_eth_df.set_index("tweet_created", inplace=False)
ts_twitter_eth_df = ts_twitter_eth_df.SA_score.resample('D').mean()
standardized_twitter_scores = pd.DataFrame({'BTC':ts_twitter_btc_df, 'ETH':ts_twitter_eth_df})
# Since the server might be down on certain days, we need to ensure that time series has no discontinuities: interpolate() fills gaps of any size with a straight line
standardized_twitter_scores['BTC'].interpolate(method='linear', inplace=True)
standardized_twitter_scores['ETH'].interpolate(method='linear', inplace=True)

#  News
ts_ccn = ccn_df.set_index("date", inplace=False)
ts_ccn = ts_ccn.SA_score.resample('D').mean()
standardized_ccn_scores = pd.DataFrame({'CCN':ts_ccn})

#______________________________________________________________________________
# Preprocessing for aggregated plot:
#______________________________________________________________________________
# BTC
url = 'https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=20' # + str(len(standardized_reddit_scores))
# we want only last few days--> &toTs=1522224000
r = requests.get(url) # Decode the JSON data into a dictionary: json_data
json_data = r.json()
btc_mini = pd.DataFrame(json_data["Data"]) # dictionary of currency values is stored as a list under the key "Data"
btc_mini["timestamp"] = pd.to_datetime(btc_mini["time"], unit='s')

# ETH
url = 'https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USD&limit=20' # we want only last few days
r = requests.get(url) # Decode the JSON data into a dictionary: json_data
json_data = r.json()
eth_mini = pd.DataFrame(json_data["Data"]) # dictionary of currency values is stored as a list under the key "Data"
eth_mini["timestamp"] = pd.to_datetime(eth_mini["time"], unit='s')

btc_mini.set_index("timestamp", inplace=True)
eth_mini.set_index("timestamp", inplace=True)

# Now we scale the "Mini" BTC/ETH values so that we can plot them together with sentiment on the same axis.
# we scale values to be between -1 and 1, i.e. on the same scale as the sentiment values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
btc_scaled = pd.DataFrame(scaler.fit_transform(btc_mini), columns=btc_mini.columns)
eth_scaled = pd.DataFrame(scaler.fit_transform(eth_mini), columns=eth_mini.columns)


#_____________________________________________________________________________________________________
# Define the app layout incl. all plots
#_____________________________________________________________________________________________________
app.layout = html.Div([html.H1('This dashboard shows current trends about Bitcoin and Ethereum in order to help you to make an informed decision for your investment',
                        id='h1-element'),
                html.H3("Is the sentiment in the News and Social Media connected to the price developments over time? Let's have a look!"),
                            dcc.Graph(id='barplot5',
                                figure = {'data':[
                                go.Scatter(
                                x = btc_mini.index,
                                y = btc_scaled.close,
                                name = "BTC in USD (scaled)",
                                visible=True,
                                marker=dict(color='#f2a900'),
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = eth_mini.index,
                                y = eth_scaled.close,
                                name = "ETH in USD (scaled)",
                                visible=True,
                                marker=dict(color='#4d4d4e'),
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = standardized_reddit_scores.index,
                                y = standardized_reddit_scores.BTC,
                                line = dict(color = '#f2a900', dash = 'dot'),
                                name = "BTC Sentiment on Reddit",
                                visible=True,
                                #marker=dict(color='green'),
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = standardized_reddit_scores.index,
                                y = standardized_reddit_scores.ETH,
                                    line = dict(color = '#4d4d4e', dash = 'dot'),
                                name = "ETH Sentiment on Reddit",
                                visible=True,
                                #marker=dict(color='blue'),
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = standardized_ccn_scores.index,
                                y = standardized_ccn_scores.CCN,
                                line = dict(color = 'green', dash = 'dash'),
                                name = "BTC and ETH Sentiment in the News",
                                visible=True,
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = standardized_twitter_scores.index,
                                y = standardized_twitter_scores.BTC,
                                line = dict(color = 'blue', dash = 'solid'),
                                name = "BTC Sentiment on Twitter",
                                visible=True,
                                #marker=dict(color='green'),
                                mode = 'markers+lines'
                                ),
                                go.Scatter(
                                x = standardized_twitter_scores.index,
                                y = standardized_twitter_scores.ETH,
                                    line = dict(color = 'purple', dash = 'solid'),
                                name = "ETH Sentiment on Twitter",
                                visible=True,
                                #marker=dict(color='blue'),
                                mode = 'markers+lines'
                                )],
                                'layout':go.Layout(title = 'BTC and ETH values & sentiment', showlegend=True,
                                                    updatemenus = list([
                                                        dict(active=-1, buttons=list([
                                                                dict(label = 'BTC and ETH Values over time',
                                                                     method = 'update',
                                                                     args = [{'visible': [True, True, False, False, False, False, False]},
                                                                             {'title': 'BTC and ETH values'}]),
                                                                dict(label = 'BTC and ETH Sentiment on Reddit',
                                                                     method = 'update',
                                                                     args = [{'visible': [False, False, True, True, False, False, False]},
                                                                             {'title': 'BTC and ETH sentiment on Reddit'}]),
                                                                dict(label = 'News',
                                                                     method = 'update',
                                                                     args = [{'visible': [False, False, False, False, True, False, False]},
                                                                             {'title': 'BTC and ETH sentiment in the News'}]),
                                                                dict(label = 'Twitter BTC & ETH',
                                                                     method = 'update',
                                                                     args = [{'visible': [False, False, False, False, False, True, True]},
                                                                             {'title': 'BTC and ETH sentiment on Twitter'}]),
                                                                dict(label = 'Reset: show all',
                                                                     method = 'update',
                                                                     args = [{'visible': [True, True, True, True, True, True, True]},
                                                                             {'title': 'BTC and ETH values & sentiment in the News and Social Media'}])
                                                            ])
                                                        )
                                                    ])
                                                    ,
                                                    xaxis = dict(title = 'Time', range = [minDate, maxDate]),
                                                    yaxis = dict(title = 'Sentiment & Values over time')
                                            )}),
                        html.P("In this dashboard, you can analyze the sentiment on social media and in the news regarding the two most popular cryptocurrencies: Bitcoin (BTC) and Ethereum (ETH).\n \
                        You can choose the source you are interested in by selecting from the dropdown-menu on the left. \
                        The sentiment score on the Y axis is a value between -1, denoting a strong negative sentiment, and 1, very positive sentiment."),
                    dcc.Graph(id='scatterplot1',
                    figure = {'data':[
                            go.Scatter(
                            x = btc_reddit.created_utc,
                            y = btc_reddit.SA_score,
                            name = "BTC Sentiment on Reddit",
                            visible=True,
                            marker=dict(color='#f2a900'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = eth_reddit.created_utc,
                            y = eth_reddit.SA_score,
                            name = "ETH Sentiment on Reddit",
                            visible=True,
                            marker=dict(color='#4d4d4e'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = twitter_btc_df2.tweet_created,
                            y = twitter_btc_df2.SA_score,
                            name = "BTC Sentiment on Twitter",
                            visible=False,
                            marker=dict(color='#f2a900'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = twitter_eth_df2.tweet_created,
                            y = twitter_eth_df2.SA_score,
                            name = "ETH Sentiment on Twitter",
                            visible=False,
                            marker=dict(color='#4d4d4e'),
                            mode = 'markers+lines'
                            ),
                            go.Scatter(
                            x = ccn_df.date[np.logical_and(ccn_df.date >= minDate, ccn_df.date <= maxDate)],
                            y = ccn_df.SA_score[np.logical_and(ccn_df.date >= minDate, ccn_df.date <= maxDate)],
                            name = "BTC and ETH Sentiment in the News",
                            visible=False,
                            marker=dict(color='#4d4d4e'),
                            mode = 'markers+lines'
                            )
                    ],
                            'layout':go.Layout(title = 'BTC and ETH sentiment over time', showlegend=True,
                                                updatemenus = list([
                                                    dict(active=-1,
                                                         buttons=list([
                                                            dict(label = 'BTC Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [True, False, False, False, False]},
                                                                         {'title': 'BTC sentiment over time on Reddit'}]),
                                                            dict(label = 'ETH Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, True, False, False, False]},
                                                                         {'title': 'ETH sentiment over time on Reddit'}]),
                                                            dict(label = 'Both: Sentiment on Reddit',
                                                                 method = 'update',
                                                                 args = [{'visible': [True, True, False, False, False]},
                                                                         {'title': 'BTC and ETH sentiment over time on Reddit'}]),
                                                            dict(label = 'BTC Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, True, False, False]},
                                                                         {'title': 'BTC sentiment over time on Twitter'}]),
                                                            dict(label = 'ETH Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, False, True, False]},
                                                                         {'title': 'ETH sentiment over time on Twitter'}]),
                                                            dict(label = 'Both: Sentiment on Twitter',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, True, True, False]},
                                                                         {'title': 'BTC and ETH sentiment over time on Twitter'}]),
                                                            dict(label = 'BTC & ETH Sentiment in the News',
                                                                 method = 'update',
                                                                 args = [{'visible': [False, False, False, False, True]},
                                                                         {'title': 'BTC and ETH Sentiment in the News'}])
                                                        ]),
                                                    )
                                                ])
                                                ,
                                                xaxis = dict(title = 'Time'), #, range = [minDate, maxDate]),
                                                yaxis = dict(title = 'Sentiment')
                                        )}
                                        ),
# Sentiment grouped
                    dcc.Graph(id='pie2',
                    figure = {'data':[
                        go.Pie(
                            labels=btc_grouped.sentiment,
                            values=btc_grouped.nr_of_tweets,
                            name = 'BTC Sentiment on Reddit',
                            visible=True,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686']) # set the colors to red, yellow and green for pie chart
                        ),
                        go.Pie(
                            labels=eth_grouped.sentiment,
                            values=eth_grouped.nr_of_tweets,
                            name = 'ETH Sentiment on Reddit',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686'])
                        ),
                        go.Pie(
                            labels=twitter_btc_grouped.sentiment,
                            values=twitter_btc_grouped.nr_of_tweets,
                            name = 'BTC Sentiment on Twitter',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686'])
                        ),
                        go.Pie(
                            labels=twitter_eth_grouped.sentiment,
                            values=twitter_eth_grouped.nr_of_tweets,
                            name = 'ETH Sentiment on Twitter',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686'])
                        ),
                        go.Pie(
                            labels=ccn_grouped.sentiment,
                            values=ccn_grouped.nr_of_articles,
                            name = 'BTC and ETH Sentiment in the News',
                            visible=False,
                            marker=dict(colors=['#fc586e', '#fffaaa', '#87d686'])
                        )
                        ],
                        'layout':go.Layout(title = 'BTC sentiment on Reddit', showlegend=True,
                                            updatemenus = list([
                                                dict(active=-1,
                                                     buttons=list([
                                                        dict(label = 'BTC sentiment on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [True, False, False, False, False]},
                                                                     {'title': 'BTC sentiment on Reddit'}]),
                                                        dict(label = 'ETH sentiment on Reddit',
                                                             method = 'update',
                                                             args = [{'visible': [False, True, False, False, False]},
                                                                     {'title': 'ETH sentiment on Reddit'}]),
                                                         dict(label = 'BTC sentiment on Twitter',
                                                              method = 'update',
                                                              args = [{'visible': [False, False, True, False, False]},
                                                                      {'title': 'BTC sentiment on Twitter'}]),
                                                         dict(label = 'ETH sentiment on Twitter',
                                                              method = 'update',
                                                              args = [{'visible': [False, False, False, True, False]},
                                                                      {'title': 'ETH sentiment on Twitter'}]),
                                                        dict(label = 'BTC & ETH Sentiment in the News',
                                                              method = 'update',
                                                              args = [{'visible': [False, False, False, False, True]},
                                                                      {'title': 'BTC and ETH Sentiment in the News'}])
                                                    ]),
                                                )
                                            ])
                                        )}
                                        ),
# BTC/ETH values over time
html.H3("You can also look at the recent development in the currency values. If you are interested in a specific time interval, \
you can zoom in by selecting the desired period. If you click at the small house icon, you can reset the axis again."),
                    dcc.Graph(id='scatterplot3',
                    figure = {'data':[
                            go.Scatter(
                            x = btc_values_df.timestamp,
                            y = btc_values_df.close,
                            name = 'BTC',
                            mode = 'markers+lines'
                            ),
                                go.Scatter(
                                x = btc_values_df.timestamp,
                                y = [btc_values_df.close.mean()]*len(btc_values_df.timestamp),
                                name = 'BTC Average',
                                visible = False,
                                line=dict(color='#33CFA5', dash='dash')
                                ),
                            go.Scatter(
                            x = eth_values_df.timestamp,
                            y = eth_values_df.close,
                            name = 'ETH',
                            mode = 'markers+lines'
                            ),
                                go.Scatter(
                                x = eth_values_df.timestamp,
                                y = [eth_values_df.close.mean()]*len(eth_values_df.timestamp),
                                name = 'ETH Average',
                                visible = False,
                                line=dict(color='#33CFA5', dash='dash')
                                )
                            ],
                    'layout':go.Layout(title = 'BTC and ETH values over time', showlegend=True,
                                        updatemenus = list([
                                            dict(active=-1,
                                                 buttons=list([
                                                    dict(label = 'BTC',
                                                         method = 'update',
                                                         args = [{'visible': [True, True, False, False]},
                                                                 {'title': 'BTC values over time',
                                                                 'annotations': [
                                                                 dict(x=btc_values_df.iloc[btc_values_df.close.idxmax()]["timestamp"],
                                                                   y=btc_values_df.close.max(),
                                                                   xref='x', yref='y',
                                                                   text='Max value:<br>'+str(btc_values_df.close.max()),
                                                                   ax=0, ay=-40),
                                                                 dict(x='2017-09-01 00:00:00',
                                                                     y=btc_values_df.close.mean(),
                                                                     xref='x', yref='y',
                                                                     text='Average value in the displayed time period:<br>'+str(round(btc_values_df.close.mean(), 2)),
                                                                     ax=0, ay=-40)
                                                                 ]},
                                                                 ]),
                                                    dict(label = 'ETH',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, True, True]},
                                                                 {'title': 'ETH values over time',
                                                                 'annotations': [
                                                                 dict(x=eth_values_df.iloc[eth_values_df.close.idxmax()]["timestamp"],
                                                                   y=eth_values_df.close.max(),
                                                                   xref='x', yref='y',
                                                                   text='Max value:<br>'+str(eth_values_df.close.max()),
                                                                   ax=0, ay=-40),
                                                                dict(x='2017-09-01 00:00:00',
                                                                     y=eth_values_df.close.mean(),
                                                                     xref='x', yref='y',
                                                                     text='Average value in the displayed time period:<br>'+str(round(eth_values_df.close.mean(), 2)),
                                                                     ax=0, ay=-40)
                                                                 ]}]),
                                                    dict(label = 'Both',
                                                         method = 'update',
                                                         args = [{'visible': [True, False, True, False]},
                                                                 {'title': 'BTC and ETH values over time',
                                                                 'annotations': []}])
                                                ]),
                                            )
                                        ]),
                                        xaxis = {'title':'Time'},
                                        yaxis = {'title':'Value (in USD)'}
                                        )}
                                        ),
# BoW plot
html.H3("Additionally, you can see the most common words that are used in all discussions around Bitcoin and Ethereum on diverse channels. \
You can select the channel and the currency you are interested in from the dropdown menu on the left."),
                    dcc.Graph(id='barplot4',
                    figure = {'data':[
                    go.Bar(
                        x=generate_word_list(text_col= btc_reddit.title).word,
                        y=generate_word_list(text_col= btc_reddit.title).word_count,
                        name = 'BTC words on Reddit',
                        visible=True,
                        marker=dict(color='#f2a900') # set the marker color to gold
                    ),
                    go.Bar(
                        x=generate_word_list(text_col = eth_reddit.title).word,
                        y=generate_word_list(text_col = eth_reddit.title).word_count,
                        name = 'ETH words on Reddit',
                        visible=True,
                        marker=dict(color='#4d4d4e') # set the marker color to silver
                    ),
                    go.Bar(
                        x=generate_word_list(text_col = twitter_btc_df.text).word,
                        y=generate_word_list(text_col = twitter_btc_df.text).word_count,
                        name = 'BTC words on Twitter',
                        visible=False,
                        marker=dict(color='#f2a900') # set the marker color to gold
                    ),
                    go.Bar(
                        x=generate_word_list(text_col = twitter_eth_df.text).word,
                        y=generate_word_list(text_col = twitter_eth_df.text).word_count,
                        name = 'ETH words on Twitter',
                        visible=False,
                        marker=dict(color='#4d4d4e') # set the marker color to silver
                    ),
                    go.Bar(
                        x=generate_word_list(text_col = ccn_df.article).word,
                        y=generate_word_list(text_col = ccn_df.article).word_count,
                        name = 'Top words in Cryptocurrency News',
                        visible=False,
                        marker=dict(color='#f2a900') # set the marker color to gold
                    )
                    ],
                    'layout':go.Layout(title = str(n) +' most common words currently used in Bitcoin/Ethereum discussions', showlegend=True,
                                        updatemenus = list([
                                            dict(active=-1,
                                                 buttons=list([
                                                    dict(label = 'BTC words on Reddit',
                                                         method = 'update',
                                                         args = [{'visible': [True, False, False, False, False]},
                                                                 {'title': str(n) + ' most common words currently used about Bitcoin on Reddit'}]),
                                                    dict(label = 'ETH words on Reddit',
                                                         method = 'update',
                                                         args = [{'visible': [False, True, False, False, False]},
                                                                 {'title': str(n) + ' most common words currently used about Ethereum on Reddit'}]),
                                                    dict(label = 'Both Reddit',
                                                         method = 'update',
                                                         args = [{'visible': [True, True, False, False, False]},
                                                                 {'title': str(n)+ ' most common words currently used about Bitcoin and Ethereum on Reddit'}]),
                                                    dict(label = 'BTC words on Twitter',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, True, False, False]},
                                                                 {'title': str(n) + ' most common words currently used about Bitcoin on Twitter'}]),
                                                    dict(label = 'ETH words on Twitter',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, False, True, False]},
                                                                 {'title': str(n) + ' most common words currently used about Ethereum on Twitter'}]),
                                                    dict(label = 'Both Twitter',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, True, True, False]},
                                                                 {'title': str(n) + ' most common words currently used about Bitcoin and Ethereum on Twitter'}]),
                                                    dict(label = 'Cryptocurrency News',
                                                         method = 'update',
                                                         args = [{'visible': [False, False, False, False, True]},
                                                                 {'title': str(n) + ' most common words currently used about Bitcoin and Ethereum in the News'}])
                                                ]),
                                            )
                                        ])
                                        ,
                                        xaxis = {'title':'Word'},
                                        yaxis = {'title':'Word count'}
                                    )}
                                        ),
# Interactive tables to inspect raw data
                html.Div([
                html.H2('Let\'s go more into detail: Reddit data'),
                dt.DataTable(
                    rows = reddit.to_dict('records'),
                    filterable=True,
                    sortable=True
                )]),
                html.Div([
                html.H2('Twitter data'),
                dt.DataTable(
                rows = twitter.to_dict('records'),
                filterable=True,
                sortable=True
                )]),
                html.Div([
                html.H2('The News'),
                dt.DataTable(
                rows = ccn_df.to_dict('records'),
                filterable=True,
                sortable=True
                )])
            ])

if __name__ == '__main__':
    app.run_server()
