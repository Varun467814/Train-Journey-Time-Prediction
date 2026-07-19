from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing import (
    load_data,
    clean_data,
    create_train_dataset,
)

# Paths
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_model.pkl"


def main():
    # Load and preprocess
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Create one row per train
    train_df = create_train_dataset(df)

    print("\nTrain-wise Dataset:")
    print(train_df.head())

    # Features and target
    X = train_df[["Distance", "Stops"]]
    y = train_df["Journey_Duration"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Actual vs Predicted Table
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    print("\nActual vs Predicted")
    print(results.head(10))

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Actual vs Predicted Graph
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2
    )

    plt.xlabel("Actual Journey Duration (minutes)")
    plt.ylabel("Predicted Journey Duration (minutes)")
    plt.title("Actual vs Predicted Journey Duration")
    plt.grid(True)

    plt.show()

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("\n✅ Model saved successfully!")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()