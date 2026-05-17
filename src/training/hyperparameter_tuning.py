from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning(estimator, param_grid, X_train, y_train, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1):
    """Run grid search to find the best hyperparameters for an estimator."""
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
