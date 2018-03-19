'''
denna kod kör snabbt men jag får sämre performance. wtf?
'''

# Setup the parameters
param_dist = {"max_depth": [None, 3, 5, 10, 20], # 30-50% av nr features
              "max_features": [None, 'sqrt'],
              "min_samples_leaf": [1, 5, 10],
              "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# Instantiate the GridSearchCV() object: tree_cv
tree_cv = GridSearchCV(tree, param_dist, cv=5, scoring='roc_auc', n_jobs = -1)

# Fit it to the data
load_tree = False
if load_tree == True:
    tree_cv = joblib.load('tree_cv.pkl')
else:
    t1 = datetime.datetime.now()
    tree_cv.fit(Xm, ym)
    t2 = datetime.datetime.now()
    tree_td = t2-t1
    print("Fitting time H:MM:SS ", tree_td)
    # save model
    joblib.dump(tree_cv, "tree_cv.pkl")

# Print the tuned parameters and score
print("tree")
print("Tuned Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
