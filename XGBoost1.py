import pandas as pd
data=pd.read_excel(r'C:\Users\28374\Desktop\Data\Day1.xlsx')
data=data.drop('NO.',axis=1)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
data.iloc[:,-1]=LabelEncoder().fit_transform(data.iloc[:,-1])
data.iloc[:,1:]=OrdinalEncoder().fit_transform(data.iloc[:,1:])

X=data.loc[:,data.columns!='INSOMNIA DAY1']
y=data.loc[:,data.columns=='INSOMNIA DAY1']

from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=420)

clf=XGBC(n_estimators=100,random_state=420,booster='gbtree',learning_rate=0.04).fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_true=y_test
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score,confusion_matrix,plot_roc_curve
print('Confusion matrix:',confusion_matrix(y_true,y_pred))
print("Precision score:",precision_score(y_true, y_pred))
print("Recall score:",recall_score(y_true, y_pred))
print("Accuracy score:",accuracy_score(y_true,y_pred))
print("F1 score:",f1_score(y_true, y_pred))
import matplotlib.pyplot as plt
plot_roc_curve(estimator=clf,X=X_test,y=y_test,color='r',linewidth=1)
plt.plot([0,1],[0,1],color='k',linestyle='--')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize=12,loc=4)
plt.show()