# Jane Street Market Prediction
![](./Banner.png)

## Introduction
“Buy low, sell high.” It sounds so easy….

In reality, trading for profit has always been a difficult problem to solve, even more so in today’s fast-moving and complex financial markets. Electronic trading allows for thousands of transactions to occur within a fraction of a second, resulting in nearly unlimited opportunities to potentially find and take advantage of price differences in real time.

In a perfectly efficient market, buyers and sellers would have all the agency and information needed to make rational trading decisions. As a result, products would always remain at their “fair values” and never be undervalued or overpriced. However, financial markets are not perfectly efficient in the real world.

Developing trading strategies to identify and take advantage of inefficiencies is challenging. Even if a strategy is profitable now, it may not be in the future, and market volatility makes it impossible to predict the profitability of any given trade with certainty. As a result, it can be hard to distinguish good luck from having made a good trading decision.

In the first three months of this challenge, you will build your own quantitative trading model to maximize returns using market data from a major global stock exchange. Next, you’ll test the predictiveness of your models against future market returns and receive feedback on the leaderboard.

Your challenge will be to use the historical data, mathematical tools, and technological tools at your disposal to create a model that gets as close to certainty as possible. You will be presented with a number of potential trading opportunities, which your model must choose whether to accept or reject.

In general, if one is able to generate a highly predictive model which selects the right trades to execute, they would also be playing an important role in sending the market signals that push prices closer to “fair” values. That is, a better model will mean the market will be more efficient going forward. However, developing good models will be challenging for many reasons, including a very low signal-to-noise ratio, potential redundancy, strong feature correlation, and difficulty of coming up with a proper mathematical formulation.

Jane Street has spent decades developing their own trading models and machine learning solutions to identify profitable opportunities and quickly decide whether to execute trades. These models help Jane Street trade thousands of financial products each day across 200 trading venues around the world.

Admittedly, this challenge far oversimplifies the depth of the quantitative problems Jane Streeters work on daily, and Jane Street is happy with the performance of its existing trading model for this particular question. However, there’s nothing like a good puzzle, and this challenge will hopefully serve as a fun introduction to a type of data science problem that a Jane Streeter might tackle on a daily basis. Jane Street looks forward to seeing the new and creative approaches the Kaggle community will take to solve this trading challenge.

## Methods
Tools:
- NumPy, Pandas, XGBoost and Scikit-learn for data analysis and inference
- Hyperopt for hyperparameter tuning
- GitHub for version control
- Pycharm as IDE

Methods used with Scikit:
- Model Training: train_test_split
- Metrics: accuracy_score

Methods used with Hyperopt:
- Hyperparameter tuning: tpe

Methods used with XGBoost:
- Model Selection: XGBClassifier

## Results
Notebook is live at https://www.kaggle.com/githendumukiri/notebook-xgb-hyperopt
Public submission is live at https://www.kaggle.com/githendumukiri/xgb-hyperopt


## Discussion
After learning about stock market data and common models used to tackle the problems in relation to timeseries data, I setled on an RNN using LSTM, only to come to find out that the dates and timestamps are not available rather each entry has a uniquie tsid which is stored in some form of chronological order.[6][7] 

I studied many of the kaggle notebooks and saw how ineffective this strategy would be given the format of the data. In addition, it seems as if the high perfroming models had turned this into a classification problem rather than a regression one. The regression approaches all focused on expert level timeseries anyalasis or feature enginering that stems from finincial theory corresponding to the resp and the feature set. All of which I was not familiiar with. [8][9]

Upon further investigation I stumbled upon boosting and XGBoost. Due to the nature of XGBoost ability to combine multiple weak learners into one strong leaner through identifying rules in the feature set, I figured it would be the best model to at least begin working with. The train and test files had thousands of entries and I figured this model would be complex enough to handle the large dataset. I understood at high level how the model used descion trees to train and predict. My strategy for the remainder of the competiton is to use hyperopt to select the most optimal hyperparameters for the model and improve my score in the rankings.[1][2][3][4][5] 

I ran into alot of challenges during this project. The size of the dataset put a strain on how well my model would perform for the presentation, I was able to get a working model prior to presenting, but it had taken nearly 10 hours to tune the hyperparameters which I had to hault. Meaning, after improving my model 5 percentage points I had a few hours remaining in the number of evaluations. Hyperopt is very complicated and it has been a challenge selecting the appropriate spaces for the baysien infrence algorithim to learn from. In addition, not having any sort of descriptions of what the features are has been like walking into a fight blind folded and swinging. In adittion, my computer kept hanging because I have only 16gb of RAM and only 13GB available on Kaggle. I'm heavily considering upgrading to Google Cloud Platfrom for the remainder of the competition. 

## Summary
My submision uses a XGBoost model that has been tuned using Hyperopt to predict wether or not entry is a good trade to make i.e. class 1 or a trade to pass on class 0.
This model combines weak learning models to form rules regarding the complicated feature set and combines them to form a strong learner which can make a very accurate prediction. I will continue to adjust the hyperparameters and try to at least learn some of the advanced time series approches used with the vague data to produce a competitive model for that $100,000 jackpot.

## Bibliography
[1] [XGBoost Model w/GPU] (https://www.kaggle.com/hamditarek/market-prediction-xgboost-with-gpu-fit-in-1min)

[2] [XGBoost Evalution] (https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/)

[3] [XGBoost Hyperopt] (https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9)

[4] [Hyperopt] (https://www.kaggle.com/henrylidgley/xgboost-with-hyperopt-tuning)

[5] [Boosting] (https://www.youtube.com/watch?v=kho6oANGu_A&t=939s)

[6] [LSTM Stock Prediction] (https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf_

[7] [LSTM Stock Prediction] (https://ieeexplore.ieee.org/ielx7/6287639/8948470/09165760.pdf?tp=&arnumber=9165760&isnumber=8948470&ref=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8=)

[8] [Build LSTM] (https://www.youtube.com/watch?v=arydWPLDnEc)

[9] [Build LSTM] (https://www.youtube.com/watch?v=QIUxPv5PJOY&t=1599s)
