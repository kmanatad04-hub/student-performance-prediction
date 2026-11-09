
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("STUDENT PERFORMANCE MODEL TRAINING")

df = pd.read_csv("student_data.csv")
df = df.drop_duplicates().copy()

# Target: G3 < 10 = AT RISK
# Target: G3 >= 10 = NOT AT RISK
y = (df["G3"] < 10).astype(int)

# G3 is excluded from model input
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

print("\\nMODEL EVALUATION RESULTS")
print("=" * 50)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))

print("\\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

print("\\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

joblib.dump(model, "student_model.pkl")

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\\nModel and metrics saved successfully!")
