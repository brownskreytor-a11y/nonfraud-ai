import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def true_fraud_probability(velocity_count, home_distance_km, issuer_distance_km, scaled_amount):
    """
    Smooth risk function (logistic curve) instead of a hard threshold.
    This avoids the old model jumping from 3% to 85% at one arbitrary cutoff,
    which made borderline transactions unpredictable.
    """
    risk_score = (
        0.10 * np.minimum(velocity_count - 1, 6) +
        0.35 * np.minimum(home_distance_km / 2000.0, 1.0) +
        0.30 * np.minimum(issuer_distance_km / 2500.0, 1.0) +
        0.25 * np.minimum(scaled_amount / 9.5, 1.0)
    )
    # logistic squashing centered at risk_score = 0.5
    return 1.0 / (1.0 + np.exp(-8.0 * (risk_score - 0.5)))


def generate_balanced_fraud_dataset(num_samples=60000):
    np.random.seed(42)

    # --- Amount: realistic dollar amounts, THEN log1p it exactly like app.py does ---
    # lognormal gives a realistic spend spread: lots of small purchases, a long tail
    # of big ones, matching real-world card transaction behaviour.
    raw_amounts = np.random.lognormal(mean=4.2, sigma=1.3, size=num_samples)
    raw_amounts = np.clip(raw_amounts, 1, 50000)
    scaled_amounts = np.log1p(raw_amounts)  # <-- same transform app.py applies at inference

    transaction_hours = np.random.randint(0, 24, size=num_samples)

    velocity_counts = np.random.choice(
        [1, 2, 3, 4, 5, 6, 8, 10],
        size=num_samples,
        p=[0.50, 0.20, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
    )

    # --- Distances: continuous lognormal spread instead of two disjoint buckets ---
    # This removes the "dead zone" (e.g. 150km-500km) that the old generator
    # never produced examples for, which made mid-range distances unreliable.
    home_distances = np.clip(np.random.lognormal(mean=3.3, sigma=1.9, size=num_samples), 0, 20000)
    issuer_distances = np.clip(np.random.lognormal(mean=3.0, sigma=2.0, size=num_samples), 0, 20000)

    fraud_prob = true_fraud_probability(velocity_counts, home_distances, issuer_distances, scaled_amounts)
    fraud_labels = np.random.binomial(1, fraud_prob)

    X = np.column_stack((scaled_amounts, transaction_hours, velocity_counts, home_distances, issuer_distances))
    y = fraud_labels
    return X, y


def train_and_save_model():
    print("Generating training data with realistic feature distributions...")
    X, y = generate_balanced_fraud_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training Random Forest classifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    print("\nModel Evaluation Performance:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"\nModel successfully updated and saved to {model_filename}!")


if __name__ == "__main__":
    train_and_save_model()
