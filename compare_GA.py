import pandas as pd
from sklearn.metrics import accuracy_score
a=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,3,3,3,1,3,3,3,3,3,3,1,1,3,3,3,3,1,3,1,3,1,3,3,1,1,3,3,3,3,3,3,3,3,3,3,1,3,3,3,1,3,3,3,1,3,3,1]
data = pd.read_csv('iris_data.csv')
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
y_corrected = []
for i in a:
    if i == 1:
        y_corrected.append(1)
    elif i == 2:
        y_corrected.append(0)
    else:
        y_corrected.append(2)
print(y_corrected)
acc=accuracy_score(y_corrected,y)
print(acc)