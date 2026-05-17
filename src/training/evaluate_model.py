def evaluate_model(model,y_test,X_test):
    from sklearn.metrics import classification_report, confusion_matrix
    # Placeholder for model evaluation code
    print("Evaluating model with test data")
    # Simulate evaluation process
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  
    