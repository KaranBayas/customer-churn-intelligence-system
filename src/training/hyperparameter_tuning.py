def hyperparameter_tuning(model, param_grid, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    # Placeholder for hyperparameter tuning code
    print("Starting hyperparameter tuning")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_
