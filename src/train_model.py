from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from preprocessing import load_data, preprocess

# Load and preprocess
df = load_data("data/german_credit_data.csv")
(X_train, X_test, y_train, y_test), encoder, scaler = preprocess(df)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump({"model": clf, "encoder": encoder, "scaler": scaler}, "models/credit_model.pkl")