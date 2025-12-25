import pandas as pd
data=pd.read_excel(r'C:\Users\28374\Desktop\Data\Day2.xlsx').drop('NO.',axis=1)
from sklearn.preprocessing import LabelEncoder
data.iloc[:,-1]=LabelEncoder().fit_transform(data.iloc[:,-1])
X=data.loc[:,data.columns!='INSOMNIA DAY2']
y=data.loc[:,data.columns=='INSOMNIA DAY2']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=420)
from sklearn.linear_model import Lasso,LassoCV
ls_=LassoCV(eps=0.00001,n_alphas=500,cv=10).fit(X_train,y_train)
print('alpha output',ls_.alpha_)     #alpha=0.023423300981663793
LASSO=Lasso(alpha=0.023423300981663793).fit(X_train,y_train)
from sklearn.metrics import r2_score,mean_squared_error
y_pred=LASSO.predict(X_test)
print('R2',r2_score(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))
print('coefficients of each variants',[*zip(X.columns,LASSO.coef_)])