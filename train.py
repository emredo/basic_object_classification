import os
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

#Loading data with numpy.
data = np.load("./models/data.npy")
x = data[:,:-2]
y = data[:,-1]

#Scaling the pixel values.
sc = StandardScaler()
X = sc.fit_transform(x)

#Train and test data generation.
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

#Creating our Support Vector Classifier.
svc_model = SVC()  #Kernel settings or other parameter's tunings can be done also.

#Training the data with created classifier.
svc_model.fit(x_train,y_train)

#Taking first predictions, after hot training.. :)
prediction = svc_model.predict(x_test)

#Result metrics (confusion matrix and accuracy score)
cf = confusion_matrix(y_test,prediction)
accuracy = accuracy_score(y_test,prediction)
print("Confusion matrix results\n",cf)
print("Accuracy score results",accuracy)

#Saving model with pickle.
if not os.path.isdir("./models"):
    os.mkdir("./models")
pickle.dump(svc_model, open(f'./models/model_{time.time()}.pkl', 'wb'))