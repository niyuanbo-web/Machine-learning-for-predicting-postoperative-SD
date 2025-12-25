import pandas as pd
data=pd.read_excel(r'C:\Users\28374\Desktop\Data\Day1.xlsx')       #path of the document may need to be modified
data=data.drop('NO.',axis=1)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
data.iloc[:,-1]=LabelEncoder().fit_transform(data.iloc[:,-1])
data.iloc[:,1:]=OrdinalEncoder().fit_transform(data.iloc[:,1:])

X=data.loc[:,data.columns!='INSOMNIA DAY1']
y=data.loc[:,data.columns=='INSOMNIA DAY1']

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score,confusion_matrix,plot_roc_curve
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=420)
dtc=DecisionTreeClassifier(random_state=420,criterion='gini',max_depth=5,min_samples_split=2,splitter='random').fit(X_train,y_train)

y_pred=dtc.predict(X_test)
y_true=y_test
CM=confusion_matrix(y_true,y_pred)
print('Confusion matrix:',CM)
print("Precision score: ",precision_score(y_true, y_pred))
print("Recall score: ",recall_score(y_true, y_pred))
print("Accuracy score: ",accuracy_score(y_true,y_pred))
print("F1 score:",f1_score(y_true, y_pred))

#ROC curve
ROC=plot_roc_curve(estimator=dtc, X=X_test,y=y_test,linewidth=1,color='r')
plt.plot([0,1],[0,1],color='k',linestyle='--')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize=12,loc=4)
plt.show()
