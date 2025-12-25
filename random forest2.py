import pandas as pd
data=pd.read_excel(r'C:\Users\28374\Desktop\Data\Day2.xlsx').drop('NO.',axis=1)
from sklearn.preprocessing import LabelEncoder
data.iloc[:,-1]=LabelEncoder().fit_transform(data.iloc[:,-1])
X=data.loc[:,data.columns!='INSOMNIA DAY2']
y=data.loc[:,data.columns=='INSOMNIA DAY2']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=420,test_size=0.3)
rfc=RandomForestClassifier().fit(X_train,y_train)
y_pred=rfc.predict(X_test)
y_true=y_test
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score,confusion_matrix,plot_roc_curve
import matplotlib.pyplot as plt
print('Accuracy Score:',accuracy_score(y_true,y_pred))
print('Precision Score:',precision_score(y_true,y_pred))
print('Recall Score:',recall_score(y_true,y_pred))
print('CM:',confusion_matrix(y_true,y_pred))
print('f1 Score:',f1_score(y_true,y_pred))
ROC=plot_roc_curve(estimator=rfc,X=X_test,y=y_test,linewidth=1,color='r')
plt.plot([0,1],[0,1],color='k',linestyle='--')
plt.title('Receiver operating characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize=12,loc=4)
plt.show()