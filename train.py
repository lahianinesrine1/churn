import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore') #filtrer les avertissements
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
from sklearn import metrics

df=pd.read_csv('./data/Churn_Modelling.csv')
x=df.iloc[:,3:13].values
y=df.iloc[:,13].values
le=LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])
x[:,2]=le.fit_transform(x[:,2])
ct=ColumnTransformer([("Geography",OneHotEncoder(),[1])],remainder='passthrough')
x=ct.fit_transform(x)
x=x[:,1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)
knn=KNeighborsClassifier (n_neighbors = 5)
knn.fit(x_train_sc,y_train)
y_predictKNN=knn.predict(x_test_sc)

cm= confusion_matrix(y_test,y_predictKNN)
print (cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] )
ax.set_yticklabels([''] )
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.savefig('confusion_matrix.png')
plt.clf()


print('accurancy KNN {0:.3f}'.format(accuracy_score(y_test,y_predictKNN)))
print('Precision KNN {0:.3f}'.format(precision_score(y_test,y_predictKNN)))
with open('metrics.txt', 'w') as file:
    file.write('accurancy KNN {0:.3f}'.format(accuracy_score(y_test,y_predictKNN))+ '\n')
    file.write('Precision KNN {0:.3f}'.format(precision_score(y_test,y_predictKNN)))

fpr,tpr,thr=metrics.roc_curve(y_test,y_predictKNN)
auc=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr,'-',lw=3,label='gamma=0.01,AUC=%2.f'%auc)  
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title ('KNN  Roc Curves')
plt.legend(loc='lower right') 
plt.savefig('roc_curves.png')
