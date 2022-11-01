import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# 데이터 불러오기
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv("iris.data", names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)

# 전처리 (Scaling)
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# 모델 생성 및 훈련
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

# 모델 정확도 평가
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

# 최적의 k 찾기
k = 10
acc_array = np.zeros(k)
for k in np.arange(1, k + 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("Best K: {}, Accuracy: {}".format(k, max_acc))
