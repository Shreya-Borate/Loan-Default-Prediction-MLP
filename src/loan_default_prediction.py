
# ============================================================
# LOAN DEFAULT PREDICTION USING MULTI-LAYER PERCEPTRON (MLP)
# Deep Learning Assignment
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# STEP 1: PATHS AND CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Loan_Default.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Default"

print("=" * 70)
print("LOAN DEFAULT PREDICTION USING MLP")
print("=" * 70)

# ------------------------------------------------------------
# STEP 2: LOAD AND UNDERSTAND THE DATASET
# ------------------------------------------------------------

print("\nSTEP 1: LOAD THE DATASET")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place loan_default_dataset.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Remove completely empty rows
df.dropna(how="all", inplace=True)

required_columns = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "EmploymentYears", "ExistingLoans", "MonthlyDebt",
    "LoanTerm", "PreviousDefault", "HomeOwnership", "Default"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# STEP 3: CLEAN THE DATASET
# ------------------------------------------------------------

print("\nSTEP 2: CLEAN THE DATASET")

df.drop_duplicates(inplace=True)

# Convert numeric features to numeric data types
numeric_columns = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "EmploymentYears", "ExistingLoans", "MonthlyDebt",
    "LoanTerm"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize categorical values
for col in ["PreviousDefault", "HomeOwnership"]:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.title()
    )
    df[col] = df[col].replace("", pd.NA)

# Validate the target column
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# Only 0 and 1 are valid target labels
df.loc[~df[TARGET].isin([0, 1]), TARGET] = np.nan

# Rows without a known target cannot be used for supervised learning
df.dropna(subset=[TARGET], inplace=True)
df[TARGET] = df[TARGET].astype(int)

if df.empty:
    raise ValueError("No valid records remain after cleaning.")

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The Default target must contain both classes: 0 and 1."
    )

print("Shape after cleaning:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

# Target distribution
print("\nDefault class counts:")
print(df[TARGET].value_counts().sort_index())

print("\nDefault class percentages:")
print(
    (df[TARGET].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=TARGET)
plt.title("Distribution of Loan Default Classes")
plt.xlabel("Default (0 = Low Risk, 1 = High Risk)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "default_distribution.png", dpi=300)
plt.close()

# Numerical feature distributions
df[numeric_columns].hist(figsize=(14, 10), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "numerical_distributions.png", dpi=300)
plt.close()

# Correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(
    df[numeric_columns + [TARGET]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# Missing-value visualization
plt.figure(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Before Imputation")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "missing_values.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# STEP 5: CHECK CLASS BALANCE
# ------------------------------------------------------------

print("\nSTEP 4: CHECK TARGET CLASS BALANCE")

class_counts = df[TARGET].value_counts()
class_percentages = df[TARGET].value_counts(normalize=True) * 100

print("Class counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages.round(2))

minority_percentage = class_percentages.min()

if minority_percentage < 40:
    print("\nThe target classes may be imbalanced.")
    print("We will evaluate precision, recall and F1-score.")
else:
    print("\nThe target classes are reasonably balanced by this check.")

# ------------------------------------------------------------
# STEP 6: SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

print("\nSTEP 5: SEPARATE X AND y")

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_columns = [
    "PreviousDefault",
    "HomeOwnership"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_columns
]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Numerical features:", numeric_features)
print("Categorical features:", categorical_columns)

# ------------------------------------------------------------
# STEP 7: TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\nSTEP 6: SPLIT THE DATASET")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

print(
    "\nStratified splitting preserves approximately the same "
    "target-class proportions in training and testing data."
)

# ------------------------------------------------------------
# STEP 8: ENCODE CATEGORICAL FEATURES AND SCALE NUMERICAL FEATURES
# ------------------------------------------------------------

print("\nSTEP 7: PREPROCESS THE FEATURES")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Fit preprocessing only on training data to avoid data leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# ------------------------------------------------------------
# STEP 9: BUILD THE BASELINE MLP CLASSIFIER
# ------------------------------------------------------------

print("\nSTEP 8: CREATE THE MLP CLASSIFIER")

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

print(mlp)

# ------------------------------------------------------------
# STEP 10: TRAIN THE MODEL
# ------------------------------------------------------------

print("\nSTEP 9: TRAIN THE MODEL")

mlp.fit(X_train_processed, y_train)

print("Model training completed.")
print("Training iterations:", mlp.n_iter_)
print("Final training loss:", round(mlp.loss_, 6))
print("Converged within max_iter:", not mlp.n_iter_ == mlp.max_iter)

# ------------------------------------------------------------
# STEP 11: MAKE PREDICTIONS
# ------------------------------------------------------------

print("\nSTEP 10: PREDICT ON TEST DATA")

y_pred = mlp.predict(X_test_processed)

# ------------------------------------------------------------
# STEP 12: EVALUATE MODEL ACCURACY
# ------------------------------------------------------------

print("\nSTEP 11: MODEL ACCURACY")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ------------------------------------------------------------
# STEP 13: CONFUSION MATRIX
# ------------------------------------------------------------

print("\nSTEP 12: CONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Risk (0)", "High Risk (1)"],
    yticklabels=["Low Risk (0)", "High Risk (1)"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# STEP 14: CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nSTEP 13: CLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Low Default Risk", "High Default Risk"],
    zero_division=0
)

print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# Save baseline metrics
baseline_results = {
    "Model": "Baseline MLP",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "Iterations": mlp.n_iter_,
    "Final_Loss": mlp.loss_
}

# ------------------------------------------------------------
# STEP 15: PLOT TRAINING LOSS
# ------------------------------------------------------------

print("\nSTEP 14: PLOT TRAINING LOSS")

plt.figure(figsize=(9, 5))
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_loss.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# STEP 16: HYPERPARAMETER EXPERIMENTS
# Change one parameter at a time
# ------------------------------------------------------------

experiment_results = []

def evaluate_experiment(name, model):
    """Train and evaluate an MLP using the same data split."""

    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_test_processed)

    result = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Iterations": model.n_iter_,
        "Final_Loss": model.loss_
    }

    experiment_results.append(result)

    print(
        f"{name:<35} "
        f"Accuracy={result['Accuracy']:.4f}  "
        f"Precision={result['Precision']:.4f}  "
        f"Recall={result['Recall']:.4f}  "
        f"F1={result['F1_Score']:.4f}"
    )

    return model


# ------------------------------------------------------------
# EXPERIMENT 1: ACTIVATION FUNCTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 1: ACTIVATION FUNCTION")
print("=" * 70)

activation_results = []

for activation in ["identity", "logistic", "tanh", "relu"]:

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation=activation,
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    result = evaluate_experiment(
        f"Activation: {activation}",
        model
    )

    activation_results.append({
        "Activation": activation,
        "Accuracy": accuracy_score(
            y_test, result.predict(X_test_processed)
        )
    })

activation_df = pd.DataFrame(activation_results)
activation_df.to_csv(
    RESULTS_DIR / "activation_experiment.csv",
    index=False
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=activation_df,
    x="Activation",
    y="Accuracy"
)
plt.title("Activation Function Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "activation_experiment.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# EXPERIMENT 2: HIDDEN LAYER STRUCTURE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 2: HIDDEN LAYERS")
print("=" * 70)

hidden_layer_results = []

hidden_layer_options = [
    (10,),
    (20, 10),
    (50, 25),
    (100, 50, 25)
]

for layers in hidden_layer_options:

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    result = evaluate_experiment(
        f"Hidden layers: {layers}",
        model
    )

    hidden_layer_results.append({
        "Hidden_Layers": str(layers),
        "Accuracy": accuracy_score(
            y_test, result.predict(X_test_processed)
        )
    })

hidden_df = pd.DataFrame(hidden_layer_results)
hidden_df.to_csv(
    RESULTS_DIR / "hidden_layers_experiment.csv",
    index=False
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=hidden_df,
    x="Hidden_Layers",
    y="Accuracy"
)
plt.title("Hidden Layer Structure Comparison")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "hidden_layers_experiment.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# EXPERIMENT 3: LEARNING RATE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 3: LEARNING RATE")
print("=" * 70)

learning_rate_results = []
learning_rates = [0.0001, 0.001, 0.01, 0.1]

for rate in learning_rates:

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=rate,
        max_iter=1000,
        random_state=42
    )

    result = evaluate_experiment(
        f"Learning rate: {rate}",
        model
    )

    learning_rate_results.append({
        "Learning_Rate": rate,
        "Accuracy": accuracy_score(
            y_test, result.predict(X_test_processed)
        )
    })

learning_rate_df = pd.DataFrame(learning_rate_results)
learning_rate_df.to_csv(
    RESULTS_DIR / "learning_rate_experiment.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(
    learning_rate_df["Learning_Rate"],
    learning_rate_df["Accuracy"],
    marker="o"
)
plt.xscale("log")
plt.title("Learning Rate Comparison")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "learning_rate_experiment.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# STEP 17: COMPARE ALL EXPERIMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HYPERPARAMETER EXPERIMENT SUMMARY")
print("=" * 70)

experiment_df = pd.DataFrame(experiment_results)
experiment_df.to_csv(
    RESULTS_DIR / "all_experiment_results.csv",
    index=False
)

print(experiment_df.to_string(index=False))

# Identify the best observed configuration by F1-score.
# This is a comparison on the test set, so use it as an
# educational experiment, not as an unbiased final estimate.
best_experiment = experiment_df.loc[
    experiment_df["F1_Score"].idxmax()
]

print("\nBest observed experiment by F1-score:")
print(best_experiment.to_string())

# ------------------------------------------------------------
# STEP 18: PREDICT FOR A NEW LOAN APPLICANT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEW LOAN APPLICANT PREDICTION")
print("=" * 70)

# Example applicant; replace these values with actual inputs.
new_applicant = pd.DataFrame([{
    "Age": 35,
    "Income": 600000,
    "LoanAmount": 250000,
    "CreditScore": 700,
    "EmploymentYears": 8,
    "ExistingLoans": 1,
    "MonthlyDebt": 15000,
    "LoanTerm": 36,
    "PreviousDefault": "No",
    "HomeOwnership": "Own"
}])

# Use the same preprocessing fitted on the training set
new_applicant_processed = preprocessor.transform(new_applicant)

prediction = mlp.predict(new_applicant_processed)[0]
probabilities = mlp.predict_proba(new_applicant_processed)[0]

class_positions = {
    class_label: index
    for index, class_label in enumerate(mlp.classes_)
}

default_probability = probabilities[class_positions[1]]

print("\nApplicant details:")
print(new_applicant.to_string(index=False))

print("\nPredicted class:", prediction)
print(f"Estimated default probability: {default_probability:.2%}")

if prediction == 1:
    print("Result: HIGH DEFAULT RISK")
else:
    print("Result: LOW DEFAULT RISK")

print(
    "\nNote: This is an educational model prediction, "
    "not a real lending decision."
)

# ------------------------------------------------------------
# STEP 19: SAVE FINAL BASELINE METRICS
# ------------------------------------------------------------

pd.DataFrame([baseline_results]).to_csv(
    RESULTS_DIR / "baseline_metrics.csv",
    index=False
)

print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED")
print("=" * 70)

print(f"\nResults saved in: {RESULTS_DIR}")
