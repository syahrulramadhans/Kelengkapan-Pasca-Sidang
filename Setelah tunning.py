import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import cv2
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# print(sklearn.__version__)


# Load face data
faces = []
labels = []
for label, name in enumerate(os.listdir('faces')):
    for image_name in os.listdir(f'faces/{name}'):
        image = cv2.imread(f'faces/{name}/{image_name}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to match the input size of the camera
        resized = cv2.resize(gray, (64, 48))

        faces.append(resized.flatten())
        labels.append(label)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    np.array(faces), np.array(labels), test_size=0.6)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)


# # Create SVC classifier
svc = SVC(C=0.1, kernel='rbf')


# # Create Decision Tree classifier
dt = DecisionTreeClassifier()


# K-Fold Cross Validation
kf = KFold(n_splits=5, random_state=1, shuffle=True)


def Predict(model, X, Y):
    fold = 0
    acc = []

    for train_index, test_index in kf.split(X):
        fold += 1
        print(f'Fold ke-{fold} ')
        # X_train, X_test = X.loc[train_index], X.loc[test_index]
        # y_train, y_test = Y.loc[train_index], Y.loc[test_index]
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]
        # i += 1

        # Training Classifier Model
        model.fit(X_train, y_train)

        # Melakukan Prediksi
        y_predict = model.predict(X_test)
        # Membandingkan hasil Prediksi terhadap data asli
        print("Classification Report")
        print(classification_report(y_test, y_predict))
        akurasi = np.round(accuracy_score(y_test, y_predict), 4)*100
        print(f'Accuracy : {akurasi}%')
        acc.append(akurasi)

        cnf_matrix = confusion_matrix(y_test, y_predict)
        s = sn.heatmap(cnf_matrix/np.sum(cnf_matrix),
                       annot=True,
                       fmt='.2%',
                       xticklabels=['Negatif', 'Positif'],
                       yticklabels=['Negatif', 'Positif'],
                       )
        s.set_xlabel('Predicted Label')
        s.set_ylabel('True Label')
        plt.show()
        print('-'*100)

    print('='*100)
    print(f'{fold} Fold Validation Mean Accuracy')
    print('akurasi tertinggi : ', np.max(acc))
    print('akurasi terendah : ', np.min(acc))
    print('rata-rata akurasi : ', np.mean(acc))



Predict(knn, X_train, y_train)
# print(Predict(svc, X_train, y_train))
# print(Predict(dt, X_train, y_train))
