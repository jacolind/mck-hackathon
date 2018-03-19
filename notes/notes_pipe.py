################################################
## impute

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(Xm)
Xm = imp.transform(Xm)

## standardize X

scaler = StandardScaler().fit(Xm)
Xm = scaler.transform(Xm)
Xp = scaler.transform(Xp)
```

##########################################3
# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])





##########################
# current:
-clean
    - infer dtype
    - convert each dtype
- EDA
    - clean: fix age

    - select cols based on EDA e.g. NAcount and freqs
    groupby

- choose X
- convet to dummies
- handle NA
- split
- fit and predict

# with a pipeline:
- infer dtype (utanför pipe)
- feature union (skapas först och hamnar sen i pipe )
    - datetime  & age (som blir numeric)
    - boolean, specialfix
    - categorical
        - get dummies här?

- select X (kanske ligger valet utanför en pipe, men valet ANVÄNDS i pipe)
- get dummies , ligger nog i pipe
- pipe: impute
- pipe: scale
- pipe.fit(X_train, y_train)
- y_pred_proba = pipe.predict_proba(X_test)
- roc_auc_score(y_test, y_pred_proba)
& sen kmr final submisisons på ngt sätt.

# inte så mkt pipeline!
- concat Xm and Xp
-
