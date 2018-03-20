below are my extracts from the docs on how to use it. these are especially useful since @Stormblessed said they worked for him 

## xgboost subsampling

`subsample` [default=1]
subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
range: (0,1]

`colsample_bytree` [default=1]
subsample ratio of columns when constructing each tree.
range: (0,1]

## xgboost  learning rate

You can also reduce stepsize `eta`, but needs to remember to increase `num_round` when you do so.

`eta` [default=0.3, alias: learning_rate]
step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
range: [0,1]


## reduce model complexity

`max_depth, min_child_weight, gamma`

https://xgboost.readthedocs.io/en/latest/parameter.html :

`max_depth` [default=6]
maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting. 0 indicates no limit, limit is required for depth-wise grow policy.

`min_child_weight` [default=1]
If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. The larger, the more conservative the algorithm will be.
range: [0,∞]

`gamma` [default=0, alias: min_split_loss]
minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
range: [0,∞]


## imbalance in xgboost

`scale_pos_weight`:
For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of xgboost model, and there are two ways to improve it. If you care only about the ranking order (AUC) of your prediction: Balance the positive and negative weights, via scale_pos_weight

If you care about predicting the right _probability_
In such a case, you cannot re-balance the dataset
In such a case, set parameter `max_delta_step` to a finite number (say 1) will help convergence

