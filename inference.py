import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pickle

class Classifier:
    def __init__(self,model_path,sample_data_path):
        self.model = pickle.load(open(model_path, "rb"))
        sample_data = np.load(sample_data_path)
        x = sample_data[:, :-2]
        self.sc = StandardScaler()
        self.sc.fit(x)

    def preprocess(self,arr):
        arr = cv2.resize(arr,(64,64))
        data = np.asarray(arr)
        data = data.reshape((1,64*64*3))
        transformed = self.sc.transform(data)
        return transformed

    def predict(self,img):
        data = self.preprocess(img)
        prediction = self.model.predict(data)
        return prediction

img = cv2.imread("cls0.jpg")
classifier = Classifier("./models/log_reg_model.sav","./models/data.npy")
prediction = classifier.predict(img)
print(prediction)