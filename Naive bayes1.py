import pandas as pd
data=pd.read_excel(r'C:\Users\28374\Desktop\Data\Day1.xlsx')
data=data.drop('NO.',axis=1)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
data.iloc[:,-1]=LabelEncoder().fit_transform(data.iloc[:,-1])
data.iloc[:,1:]=OrdinalEncoder().fit_transform(data.iloc[:,1:])

X=data.loc[:,data.columns!='INSOMNIA DAY1']
y=data.loc[:,data.columns=='INSOMNIA DAY1']

from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score,confusion_matrix,plot_roc_curve
import matplotlib.pyplot as plt
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=420)
gnb=gnb().fit(X_train,y_train)

y_pred=gnb.predict(X_test)
y_true=y_test
CM=confusion_matrix(y_true,y_pred)
print('Confusion matrix:',CM)
print("Precision score:",precision_score(y_true, y_pred))
print("Recall score:",recall_score(y_true, y_pred))
print("Accuracy score:",accuracy_score(y_true,y_pred))
print("F1 score:",f1_score(y_true, y_pred))

from sklearn.metrics import brier_score_loss
print('brier_score_loss:',brier_score_loss(y_true,y_pred))    #布里尔分数，可用可不用

plot_roc_curve(estimator=gnb,X=X_test,y=y_test,color='r',linewidth=1)
plt.plot([0,1],[0,1],color='k',linestyle='--')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize=12,loc=4)
plt.show()