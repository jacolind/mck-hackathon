read "ML mckhack postmortem notes1"

convert the above approach to a  pipelines. see `notes_pipe.py`

add an eda section in the bottom - an appendix of how i "realized" each thing should be cleaned.

install and use xgboost

gridsearchcv did not improve roc_auc in Xp or mck solution checker

votingclassifier (see docs /ensemble.html)

handle NA differently?

use oob? http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#sphx-glr-auto-examples-ensemble-plot-ensemble-oob-py

use these variables? ,'source',  and  'customer_existing_primary_bank_code'

explore how to use the date column, I hav not thought about it for now

***

# about 

this note is almost like a "todo" for myself after the ML competition 

cd Dropbox/aap/JLPM/mck

# pipeline 

todo: read this post and implement similar code http://scikit-learn.org/stable/auto_examples/plot_feature_stacker.html#sphx-glr-auto-examples-plot-feature-stacker-py

use 1 clf and 1 param to change in the paramgrid. try to get a small pipeline working, then increase the size. see also the guy who presented on youtube 

# medium Mr. 83

post  https://medium.com/@Stormblessed/my-first-mlhackathon-in-analyticsvidhya-99fad92840d2

code https://github.com/GokulEpiphany/contests-final-code/blob/master/av/McKinsetAnalytics-20Jan.ipynb

Things that worked:

1. Augmenting Approved rows (i.e find the rows which were approved and add it several times to the dataframe) The optimal number of times was 8–15. Above which the model began to overfit
2. Validation split of (39713,30000)
3. Increasing the number of trees to about 7000
4. Making sure the model uses only 5000 rows for each tree
5. Using only 10% of features for each split (This removes overfitting, most models that I have dealt with works at 50% , this dataset worked the best at 10%)

Things I can do abouts this

0. Try out rf not only DecisionTree / xgboost

1. This is similar to [insert link to kaggle page that used classification and played around with skewed datasets]

2. Use a larger test_size in the cross validation to avoid overfitting

3. use a gridserach for nr of trees (xgboost: )
4. subsampling rows (xgboost: subsample = 5000/nrows) nrows=df.shape[0]

5. subsample cols (xgboost: colsample_bytree = 0.5)

6. select the important features to avoid overfitting

7. looks like he replaces NA with zero but then creates a new column where `colname_NA = True` for those rows that had NA. 

# pandas summary 

maybe use https://pypi.python.org/pypi/pandas-summary/0.0.41

# imbalance 

start with asking yourself: are the target classes imblanaceed? for this binary task, use `df.y.mean()` 

if it is imbalances, xgboost has a parameter for that. 

in the mck data, train.csv and test.csv had high imbalances, whereas the dataset used when submitting the probabilities had a 50/50 split (since submitting all zeroes or all ones would give an accuracy of 50%).

# NA 

nr 7 above can be done using a simple function. 
