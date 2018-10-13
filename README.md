# Data Science Project 2
This project is about cryptocurrency analysis. In partcular:
1. I scraped or retrieved Bitcoin and Ethereum related data from Social media (Reddit, Twitter), as well as from the News, Cryptocurrency values over time, etc.
2. I wrote a script to continuously stream the data over 2 weeks and store it in PostgreSQL database (continuously appending new rows in the Spark Structured Streaming fashion)
3. For interactive visualizations and reporting, I created:
- a dashboard using Python's dash library & plotly,
- social media analysis using interactive graph in R to find Cryptocurrency influencers on Twitter. The graph allows to find influencers, that are most commonly retweeted and requoted.

Since the data is mainly in the text form, I conducted various Natural Language Processing & Data Cleaning steps to analyze streamed data within the dashboard, a.o. sentiment analysis, BoW-model, removing duplicates & aggregation of the financial time series data.
