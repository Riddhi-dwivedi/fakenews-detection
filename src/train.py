import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------

data = pd.read_csv("../data/fake_real_datasett.csv", engine="python", on_bad_lines="skip")

print("Dataset Loaded Successfully")

# ---------------------------
# 2️⃣ Clean Dataset
# ---------------------------

# Remove empty rows
data = data.dropna()

# Convert label to numeric
data['label'] = pd.to_numeric(data['label'], errors='coerce')

# Remove invalid labels
data = data.dropna(subset=['label'])

# Convert label to int
data['label'] = data['label'].astype(int)

# Keep only 0 and 1 labels
data = data[data['label'].isin([0,1])]
data = data.groupby('label').apply(lambda x: x.sample(data['label'].value_counts().min())).reset_index(drop=True)
print("Cleaned Label Distribution:")
print(data['label'].value_counts())

# ---------------------------
# 3️⃣ Define Features
# ---------------------------

X = data['text']
y = data['label']

# ---------------------------
# 4️⃣ Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 5️⃣ TF-IDF Vectorization
# ---------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------
# 6️⃣ Define Models
# ---------------------------

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=15),
    "Passive Aggressive": PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000)
}

best_accuracy = 0
best_model = None

# ---------------------------
# 7️⃣ Create Output Folders
# ---------------------------

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs/confusion_matrices", exist_ok=True)
os.makedirs("../outputs/roc_curves", exist_ok=True)
os.makedirs("../trainoutput", exist_ok=True)

results = []

# ---------------------------
# 8️⃣ Train Models
# ---------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy}")
    print(report)

    # ---------------------------
    # AUC Score
    # ---------------------------

    try:
        y_prob = model.predict_proba(X_test_tfidf)[:,1]
    except:
        y_prob = model.decision_function(X_test_tfidf)

    auc_score = roc_auc_score(y_test, y_prob)

    print(f"{name} AUC Score: {auc_score}")
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Report": report
    })

    # Select best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

    # ---------------------------
    # Confusion Matrix
    # ---------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f"{name} Confusion Matrix\n(0=True, 1=False)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"../outputs/confusion_matrices/{name}_cm.png")

    plt.close()
    # ---------------------------
    # ROC Curve
    # ---------------------------

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"{name} ROC Curve")

    plt.legend(loc="lower right")

    plt.savefig(f"../outputs/roc_curves/{name}_roc.png")

    plt.close()
# ---------------------------
# 9️⃣ Save Accuracy Report
# ---------------------------

with open("../trainoutput/model_results.txt", "w") as f:

    for r in results:

        f.write(f"\nModel: {r['Model']}\n")
        f.write(f"Accuracy: {r['Accuracy']}\n")

        f.write("Classification Report:\n")

        f.write(r["Report"])

        f.write("\n" + "-"*50 + "\n")

print("\nAccuracy and Classification Report saved successfully")

# ---------------------------
# 🔟 Save Best Model
# ---------------------------

pickle.dump(best_model, open("../models/best_model.pkl", "wb"))

pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

print("\nBest Model Accuracy:", best_accuracy)

print("Model and vectorizer saved successfully")