"""
Clean the hotel booking data, train the cancellation model, and save the results.

Run once before starting the app:
    python train_model.py
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RAW_FILE = DATA_DIR / "hotel_bookings_raw.csv"
CLEAN_FILE = DATA_DIR / "hotel_bookings_clean.csv"

# Booking details a manager knows at the time of booking.
NUMERIC = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
           "children", "babies", "is_repeated_guest", "previous_cancellations",
           "previous_bookings_not_canceled", "adr", "required_car_parking_spaces",
           "total_of_special_requests", "has_agent"]
CATEGORICAL = ["hotel", "arrival_date_month", "meal", "market_segment",
               "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]
TARGET = "is_canceled"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # Leakage: these record what happened to the booking, so the model must not see them.
    # Fairness: country is excluded so guests are not rated by nationality.
    # assigned_room_type, booking_changes and days_in_waiting_list can change after booking.
    df = df.drop(columns=["reservation_status", "reservation_status_date", "country",
                          "assigned_room_type", "booking_changes", "days_in_waiting_list",
                          "company"])
    df["children"] = df["children"].fillna(0).astype(int)
    df["has_agent"] = df["agent"].notna().astype(int)
    df = df.drop(columns=["agent"])
    df["meal"] = df["meal"].replace("Undefined", "SC")  # both mean "no meal package"

    # Remove impossible records and cap extreme prices.
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[df["adr"] >= 0]
    df["adr"] = df["adr"].clip(upper=500)

    # One arrival date column for the dashboard charts.
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    return df.reset_index(drop=True)


def scores(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def rate_test(df: pd.DataFrame, group: pd.Series) -> dict:
    """Cancellation rate in each group and a chi-square test of the difference."""
    _, p_value, _, _ = chi2_contingency(pd.crosstab(group, df[TARGET]))
    rates = df[TARGET].groupby(group).mean()
    return {"rate_true": float(rates[True]), "rate_false": float(rates[False]),
            "p_value": float(p_value)}


# if less than 0.001 round up to 0.001, as that is more visual than 0.000001
def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

# Needed to ensure that what I assumed about the data held once it was ran
def test_hypotheses(df: pd.DataFrame, accuracy: float) -> list:
    median_lead = df["lead_time"].median()
    h2 = rate_test(df, df["lead_time"] > median_lead)
    h3 = rate_test(df, (df["total_of_special_requests"] + df["required_car_parking_spaces"]) > 0)
    return [
        {"id": "H1", "statement": "The model will predict cancellations with at least 80% accuracy.",
         "evidence": f"Test accuracy {accuracy:.1%}",
         "result": "Accepted" if accuracy >= 0.80 else "Rejected"},
        {"id": "H2", "statement": f"Bookings made more than {median_lead:.0f} days ahead (the median) "
                                  "cancel more often than bookings made closer to arrival.",
         "evidence": f"{h2['rate_true']:.1%} vs {h2['rate_false']:.1%}, chi-square {fmt_p(h2['p_value'])}",
         "result": "Accepted" if h2["rate_true"] > h2["rate_false"] and h2["p_value"] < 0.05
         else "Rejected"},
        {"id": "H3", "statement": "Guests with special or parking requests cancel less often "
                                  "than guests without them.",
         "evidence": f"{h3['rate_true']:.1%} vs {h3['rate_false']:.1%}, chi-square {fmt_p(h3['p_value'])}",
         "result": "Accepted" if h3["rate_true"] < h3["rate_false"] and h3["p_value"] < 0.05
         else "Rejected"}]


def main():
    raw = pd.read_csv(RAW_FILE)
    df = clean(raw)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Raw rows: {len(raw):,}  Clean rows: {len(df):,}")

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    encode = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])

    # Baseline model: logistic regression.
    baseline = Pipeline([("prep", encode), ("model", LogisticRegression(max_iter=2000))])
    baseline.fit(X_train, y_train)
    baseline_scores = scores(baseline, X_test, y_test)
    print(f"Logistic regression accuracy: {baseline_scores['accuracy']:.3f}")

    # Main model: random forest, tuned with 5-fold cross-validation.
    forest = Pipeline([("prep", encode),
                       ("model", RandomForestClassifier(n_estimators=100, random_state=42,
                                                        n_jobs=-1))])
    grid = GridSearchCV(forest, {"model__max_depth": [15, 25],
                                 "model__min_samples_leaf": [1, 3]},
                        cv=5, scoring="accuracy", n_jobs=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    forest_scores = scores(best, X_test, y_test)
    print(f"Random forest accuracy: {forest_scores['accuracy']:.3f}  "
          f"best settings: {grid.best_params_}")

    # Keep the better of the two models. I know that based on the cited papers and previous work in the model
    # that the random forest model would beat the baseline, but I still wanted to make sure that held.
    use_forest = forest_scores["accuracy"] >= baseline_scores["accuracy"]
    final = best if use_forest else baseline

    names = final.named_steps["prep"].get_feature_names_out()
    importance = []
    if use_forest:
        importance = sorted(zip(names, final.named_steps["model"].feature_importances_),
                            key=lambda pair: pair[1], reverse=True)[:10]

    final_scores = forest_scores if use_forest else baseline_scores
    hypotheses = test_hypotheses(df, final_scores["accuracy"])
    for h in hypotheses:
        print(f"{h['id']}: {h['result']} ({h['evidence']})")

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chosen_model": "Random forest" if use_forest else "Logistic regression",
        "test": final_scores,
        "hypotheses": hypotheses,
        "grid_results": [{"settings": str(p), "cv_accuracy": float(s)} for p, s in
                         zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"])],
        "baseline_logistic_regression": baseline_scores,
        "always_not_canceled_accuracy": float(1 - y_test.mean()),
        "cv_accuracy": float(grid.best_score_),
        "best_params": grid.best_params_,
        "top_features": [[n.split("__", 1)[1], float(v)] for n, v in importance],
        "train_rows": len(X_train), "test_rows": len(X_test)}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
