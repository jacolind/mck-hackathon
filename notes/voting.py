eclf = VotingClassifier(estimators=[('1', DecisionTreeClassifier())
                                ,('2', LogisticRegression()),
                                #,('3', KNeighborsClassifier())
                               ]
                    , voting='soft')
eclf = eclf.fit(X_train, y_train)

print(roc_auc_score(y_test, eclf.predict_proba(X_test)[:, 1]))
# had .79 AUC for y_test

eclf.fit(Xm, ym)
y_pred_proba_sumbit = clf.predict_proba(Xp)[:, 1]votin
