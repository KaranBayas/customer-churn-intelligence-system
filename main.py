"""Train the customer churn prediction pipeline and save the trained model artifact."""

import argparse
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.config import PIPELINE_PATH, RAW_DATA_PATH
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessor,
    load_raw_data,
    prepare_features,
    split_and_save_data,
)
from src.training.evaluate_model import evaluate_model
from src.training.model import build_classifier


def save_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    """Save a trained sklearn pipeline to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def save_evaluation_report(metrics: dict, output_path: Path) -> None:
    """Write model evaluation metrics and a classification report to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write("Model evaluation report\n")
        report_file.write("========================\n\n")
        report_file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        report_file.write(f"Precision: {metrics['precision']:.4f}\n")
        report_file.write(f"Recall: {metrics['recall']:.4f}\n")
        report_file.write(f"F1 score: {metrics['f1_score']:.4f}\n")
        report_file.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        report_file.write("Classification report:\n")
        report_file.write(metrics["classification_report"])
        report_file.write("\n\nConfusion matrix:\n")
        report_file.write(str(metrics["confusion_matrix"]))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training the model."""
    parser = argparse.ArgumentParser(
        description="Train the customer churn prediction pipeline and save the model artifact."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["logistic", "random_forest", "xgboost"],
        help="Classifier type to train. Defaults to XGBoost.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for model testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_raw_data(RAW_DATA_PATH)
    df = prepare_features(df)

    X_train, X_test, y_train, y_test = split_and_save_data(
        df,
        target_column="Churn",
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )

    preprocessor = create_preprocessor(X_train)
    classifier = build_classifier(args.model)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    pipeline.fit(X_train, y_train)
    evaluation = evaluate_model(pipeline, X_test, y_test)

    save_pipeline(pipeline, PIPELINE_PATH)
    report_path = PIPELINE_PATH.parent / "training_report.txt"
    save_evaluation_report(evaluation, report_path)

    print(f"Saved trained pipeline to: {PIPELINE_PATH}")
    print(f"Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    main()
