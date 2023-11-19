import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cancer_df)

cancer_df_scaled = pd.DataFrame(scaled_data, columns = cancer.feature_names)
cancer_df_scaled['target'] = cancer.target
data = cancer_df_scaled.groupby('target').mean().T
data['diff'] = abs(data.iloc[:, 0] - data.iloc[:, 1])
data = data.sort_values(by = ['diff'], ascending = False)

features = list(data.index[:10])
X = cancer_df_scaled[features]
y = cancer_df_scaled['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
model_matrix = confusion_matrix(y_test, y_pred, labels = [1,0])
model_matrix_df = pd.DataFrame(model_matrix)

from sklearn.metrics import accuracy_score

model_accuracy = accuracy_score(y_test, y_pred)
round(model_accuracy, 2)
