from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

labels = ["cvpr", "neurips", "emnlp", "tmlr", "kdd"]
y_multiclass = [labels[i % 5] for i in range(len(preprocessed_documents))]  # Replace with true labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_multiclass)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(objective='multi:softmax', num_class=5, eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump((model, le), "conference_xgb_model.pkl")
