import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

random.seed(100)


#Loading Data
df = pd.read_csv("financial_data.csv")
#print(df.columns)
#print(df.head())

#print(df.isna().any())

df1 = df.drop(columns = ["entry_id", "pay_schedule", "e_signed"])
#print(df.head())

#Visualizing
##fig  = plt.figure(figsize = (10,8))
##plt.suptitle("Histogram", fontsize = 15)
##for i in range(df1.shape[1]):
##    plt.subplot(6,3,i+1)
##    f = plt.gca()
##    f.set_title(df1.columns.values[i])
##
##    vals = np.size(df1.iloc[:,i].unique())
##    if vals > 100:
##        vals = 100
##
##    plt.hist(df1.iloc[:,i], bins = vals, color = "green")
##plt.tight_layout(rect=[0,0.03, 1, 0.95])
##plt.show()
##
##corr = df1.corrwith(df.e_signed).plot.bar(title = "Correlation", fontsize = 10, rot = 45, grid = True)
##plt.show()

##sn.set(style = "white")
##corr = df1.corr()
##
##mask = np.zeros_like(corr, dtype = np.bool)
##mask[np.triu_indices_from(mask)] = True
##
##f, ax = plt.subplots(figsize = (18,15))
##
##cmap = sn.diverging_palette(220,10,as_cmap = True)
##
##sn.heatmap(corr,mask = mask, cmap = cmap, vmax = 3, center = 0,
##           square = True, linewidth = 5, cbar_kws = {"shrink":5})
##plt.show()

#Feature Engg.
df2 = df.drop(columns = ["months_employed"])
df2[ "account_months"] = (df.personal_account_m)+(df.personal_account_y*12)
df2 = df2.drop(columns = ["personal_account_m","personal_account_y"])
#print(df2.head())

df2 = pd.get_dummies(df2)
#print(df2.columns)
df2 = df2.drop(columns = ["pay_schedule_semi-monthly"])

response = df2["e_signed"]
users = df2["entry_id"]
df2 = df2.drop(columns = ["e_signed", "entry_id"])

#splitting data
x_train, x_test, y_train, y_test = train_test_split(df2, response, test_size = 0.3, random_state=0)
sc_x = ss()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.fit_transform(x_test))
x_train2.columns = x_train.columns
x_test2.columns = x_test.columns
#print(x_train2)
x_train = x_train2
x_test = x_test2

#Model
#Logistice Regression
LR = LogisticRegression(random_state = 0, penalty = "l1")
LR.fit(x_train, y_train)

y_pred = LR.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result = pd.DataFrame([["Linear Regression", acc, prec, rec, f1]],
             columns  = ["Model", "Accuracy", "precision", "Recall", "F1 score"])




#SVM(linear)
svm = SVC(random_state = 0, kernel = "linear")
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result1 = pd.DataFrame([["SVM(Linear)", acc, prec, rec, f1]],
             columns  = ["Model", "Accuracy", "precision", "Recall", "F1 score"])
 
result = result.append(result1, ignore_index = True)

#SVM(rbf)
svm = SVC(random_state = 0, kernel = "rbf")
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result2 = pd.DataFrame([["SVM(rbf)", acc, prec, rec, f1]],
             columns  = ["Model", "Accuracy", "precision", "Recall", "F1 score"])
 
result = result.append(result2, ignore_index = True)

#Random Forest
RF = RandomForestClassifier(random_state = 0, n_estimators =100, criterion = "entropy")
RF.fit(x_train, y_train)

y_pred = RF.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result3 = pd.DataFrame([["Random Forest", acc, prec, rec, f1]],
             columns  = ["Model", "Accuracy", "precision", "Recall", "F1 score"])
 
result = result.append(result3, ignore_index = True)

print(result)


#k-fold
accuracies = cross_val_score(estimator = RF, X = x_train, y = y_train)
print(accuracies.mean())
print(accuracies.std()*2)

#DNN

