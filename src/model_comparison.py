import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("../data/churn_data.csv")

# preprocess
df['churn'] = df['churn'].map({'Yes':1,'No':0})

X = df.drop(['churn','customer_id'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# models
log_model = LogisticRegression()
rf_model = RandomForestClassifier()

# train
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# predictions
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# ROC Curve
log_probs = log_model.predict_proba(X_test)[:,1]
rf_probs = rf_model.predict_proba(X_test)[:,1]

log_auc = roc_auc_score(y_test, log_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.plot(log_fpr, log_tpr, label=f"Logistic Regression AUC={log_auc:.2f}")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest AUC={rf_auc:.2f}")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.show()
