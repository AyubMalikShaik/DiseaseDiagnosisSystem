import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df_comb = pd.read_csv("./Dataset/dis_sym_dataset_comb.csv")
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y.values.ravel())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(X, Y_encoded)
Y_res=xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, Y_res)
print("Accuracy:",xgb_accuracy)
import lightgbm as lgb

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X, Y_encoded)
Y_pred_lgb = lgb_model.predict(X_test)
lgb_accuracy = accuracy_score(Y_test, Y_pred_lgb)
print(f"LightGBM Accuracy: {lgb_accuracy:.4f}")