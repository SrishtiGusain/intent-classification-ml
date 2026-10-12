
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. LOAD / MERGE DATASETS
# ==========================================

from pathlib import Path

DATA_DIR = Path(__file__).parent
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found. Put your dataset inside the data folder."
    )

dataframes = []

for file in csv_files:
    temp = pd.read_csv(file)

    # Ensure each file has the required columns
    required = {"utterance", "intent"}
    if required.issubset(temp.columns):
        dataframes.append(temp[["utterance", "intent"]])
    else:
        print(f"Skipping incompatible file: {file.name}")

if not dataframes:
    raise ValueError("No CSV files contained utterance and intent columns.")

df = pd.concat(dataframes, ignore_index=True)

print("Merged dataset shape:", df.shape)
print(df.head())

# ==========================================
# 2. PREPROCESS DATA
# ==========================================

# Remove rows with missing input or target
df = df.dropna(subset=["utterance", "intent"])

# Convert values to strings and remove extra spaces
df["utterance"] = df["utterance"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

# Remove empty rows
df = df[
    (df["utterance"] != "") &
    (df["intent"] != "")
]

# Remove duplicate input-target pairs
df = df.drop_duplicates(subset=["utterance", "intent"])

# Keep the same utterance from having conflicting labels
label_counts = df.groupby("utterance")["intent"].nunique()
conflicting_texts = label_counts[label_counts > 1].index
df = df[~df["utterance"].isin(conflicting_texts)]

print("\nCleaned dataset shape:", df.shape)
print("\nIntent distribution:")
print(df["intent"].value_counts())

# Check if there is enough data to stratify
if df["intent"].nunique() < 2:
    raise ValueError("At least two intent classes are required.")

if df["intent"].value_counts().min() < 2:
    raise ValueError(
        "Each intent needs at least two examples for stratified splitting."
    )

# ==========================================
# 3. SPLIT DATASET
# ==========================================

X = df["utterance"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. BUILD MODEL AND TUNE HYPERPARAMETERS
# ==========================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode"
    )),
    ("model", LogisticRegression(max_iter=2000))
])

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "model__C": [0.5, 1, 2]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest hyperparameters:")
print(grid_search.best_params_)

print("\nBest cross-validation macro F1:")
print(round(grid_search.best_score_, 4))

# ==========================================
# 5. EVALUATE MODEL
# ==========================================

y_pred = best_model.predict(X_test)

print("\nTest accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

labels = best_model.named_steps["model"].classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(12, 10))
display.plot(
    ax=ax,
    xticks_rotation="vertical",
    colorbar=False
)
plt.title("Intent Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

joblib.dump(best_model, "intent_model.joblib")
print("\nModel saved as intent_model.joblib")


# ==========================================
# 8. INFERENCE ON NEW MESSAGES
# ==========================================

def predict_intent(message):
    message = message.strip()

    if not message:
        print("Please enter a valid message.")
        return None

    # Predict the most likely intent
    prediction = best_model.predict([message])[0]

    # Get probabilities for all intent classes
    probabilities = best_model.predict_proba([message])[0]

    # Get intent labels
    classes = best_model.named_steps["model"].classes_

    # Highest predicted probability
    confidence = probabilities.max() * 100

    print("\n================================")
    print("Input message:", message)
    print("Predicted intent:", prediction)
    print(f"Highest predicted probability: {confidence:.2f}%")

    # Display top 3 predicted intents
    top_indices = probabilities.argsort()[::-1][:3]

    print("\nTop 3 possible intents:")
    for i in top_indices:
        print(f"{classes[i]}: {probabilities[i] * 100:.2f}%")

    return prediction


# Test with sample messages
predict_intent("I want to cancel my order")
predict_intent("Where is my package?")
predict_intent("I need help with my payment")
predict_intent("I forgot my password")
predict_intent("I want to create a new account")

# ==========================================
# 9. INTERACTIVE INFERENCE
# ==========================================

while True:
    message = input(
        "\nEnter a customer message (or type 'exit'): "
    )

    if message.strip().lower() == "exit":
        print("Inference stopped.")
        break

    if not message.strip():
        print("Please enter a message.")
        continue

    predict_intent(message)