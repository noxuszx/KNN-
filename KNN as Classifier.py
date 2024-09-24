import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


csvData = pd.read_csv('car_evaluation.csv')
csvData.columns = ['buying_price', 'maintenance_cost', 'number_of_doors', 'number_of_persons', 'lug_boot', 'safety', 'decision']

le = LabelEncoder()
csvData['buying_price'] = le.fit_transform(csvData['buying_price'])
csvData['maintenance_cost'] = le.fit_transform(csvData['maintenance_cost'])
csvData['number_of_doors'] = le.fit_transform(csvData['number_of_doors'])
csvData['number_of_persons'] = le.fit_transform(csvData['number_of_persons'])
csvData['lug_boot'] = le.fit_transform(csvData['lug_boot'])
csvData['safety'] = le.fit_transform(csvData['safety'])
csvData['decision'] = le.fit_transform(csvData['decision'])

X = csvData.iloc[:, :-1].values
y = csvData.iloc[:, 6].values


X_train, X_test, y_train, y_test = train_test_split(X ,y,test_size =0.40, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors =8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)