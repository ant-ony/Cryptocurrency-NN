# COGS298-Cryptocurrency-NN

 ----------------------------------
 Author: "Antony Tokarr" (Manokhin)
 Spring 2018
 COGS 298 - Deep Learning
 Prof. Josh de Leeuw
 ----------------------------------

 ----------------------
# How to Run the Project
 ----------------------
Make sure to have all of the necessary imports:
python -m pip install --user numpy scipy matplotlib pandas
pip install --upgrade tensorflow

Make sure to be using Python 3.x.

Then from the command line you can simply run:
python cryptoprediction.py

Output will occur in the terminal and in the matplotlib graph.

You can change lines 172 and 137 in cryptoprediction.py
to change the speed at which the graph changes and the number of total epochs.

 ------------------------
# What's in the Repository
 ------------------------
cryptoprediction.py --> contains all neural network code
all_currencies - kaggle dataset.csv --> the raw dataset
data_cryptocurrencies --> cleaned and reformatted dataset
Figures --> PNG files of figures from sample trial runs at certain mean squared errors.
README  --> Description and Technical Report of the Project

 -------------------------
# Motivation Behind Project
 -------------------------

In today's world, cryptocurrencies are becoming a more and more relevant and plausible form of asset exchange.  Goldman Sachs and many other major financial firms are looking into creating divisions solely dedicated to cryptocurrency finance, and the NYSE recently discussed the possibility of adding cryptocurrencies to their exchange.  The current market capitalization of Bitcoin, the leading cryptocurrency, is at $145 billion and experts have it projecting as high as half a trillion dollars to crashing completely to zero dollars net market capitalization.   

Naturally in this new, volatile, and speculative environment, investors are clamoring to find a way to predict price and anticipate the market.  Though there are many so-called "stock predicting" neural networks and deep learning solutions out there, I have sought to use the same principles in creating a cryptocurrency tracker.  

 -------
# Methods
 -------

I used the following resources as references/tutorials:
https://www.youtube.com/watch?v=ftMq5ps503w
https://www.youtube.com/watch?v=7vunJlqLZok

I used the following raw data set in my project, cleaned and reformatted for
my use:
https://www.kaggle.com/taniaj/cryptocurrency-market-history-coinmarketcap

I constructed an artificial neural network using Google's TensorFlow API.  Most standard stock prediction neural network models use the S&P 500 Index to model the data but in our case, the cryptocurrency market is still so young that there is no equivalent S&P 500 Index.  An index could be constructed from averaging prices of the top cryptocurrencies, but data on prices is very inconsistent and scattered given that so many major cryptocurrencies have only recently entered the market.  As a result, given my experience with the market and in my research, I use the Bitcoin price as an index; Bitcoin's price, given its prominence in the cryptocurrency space, often moves with the market -- if Bitcoin's price is down 10%, most of the market moves down 10%... if Bitcoin's price is up 5%, then most of the market moves up 5%.  Some currencies do have their own individual surges, but for the most part the market moves with Bitcoin.  Additionally in my data, I collected prices of 13 of the top cryptocurrencies from 4/28/2013 until 5/3/2018.  The neural network I constructed consists of the input layer, the hidden layers, the output layer.  The neural network in this project uses 4 hidden layers with decreasing levels of neurons (1024, 512, 256, 128). It is soley a feedforward network, but it the hidden layers do take the previous layer as an input.

 ----------------------
# Results and Discussion
 ----------------------

Cryptocurrencies prove very difficult to predict given their volatile nature.

After a series of ten trials:
 Recording Final Mean Squared Results: Trial Runs
  1: 387.100
  2: 4374.7363
  3: 1713.9575
  4: 484.83093
  5: 914.0567
  6: 2014.0585
  7: 7713.2705
  8: 73.09535
  9: 555.4124
 10: 555.4557

My average mean squared error was: 1878.597388

I have included a number of figures  in the repository illustrating different results in accordance to their mean squared error.

I am satisified that the shape of the graph is relatively similar to the testing line, however the magnitude of the graph is often exaggerated.  This is likely because the testing set is when cryptocurrencies are most volatile with the sudden rise and dramatic crash of the market earlier this year.

In analyzing and providing a discussion on these results a number of matters immediately come to mind.  The first is that cryptocurrencies are generally a lot more volatile than stocks, and our model has been primarily constructed for stocks -- shifting weights and varying our approach may yield better results.  Secondly, is that the testing set (the data that is set aside and not learned) consists of data from when cryptocurrency has been most volatile: the meteoric rise in December and January (Bitcoin at prices of around $19000) and then the sudden crash in the spring thereafter (Bitcoin at prices of around $6000).  This drastic rise and crash is shown in the neural network's performance, but is exaggerated.  Most of the training set is data when prices were relatively stable and incremental, but the testing set is when the data is most volatile -- this complicates our results and makes predictions difficult.  Future improvements that can be made would be using a different neural network model, such as a recurrent neural network, dropping out layers of neurons, and altering the biases and weights (as mentioned earlier).