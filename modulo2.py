import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Funcion que calcula la distancia eucladiana entre dos numeros
#En este caso es la principal manera de calcular las distancias de las k
def euclidean_distance(i,x):
    distance = np.sqrt(np.sum((i - x) ** 2))
    return distance

#Clase KNN donde se define el algoritmo de machine learning K Nearest Neighbors (KNN)
class KNN:
    #Init de la clase, definimos un k para que lo use la clase en general
    def __init__(self,k):
        self.k = k
    #El metodo fit, es para darle la informacion de entrenamiento al algoritmo
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    #Este metodo sirve para hacer la prediccion de un set de datos, hace llamadas a makeAprediction por cada dato
    def predict(self,X):
        predictions = [self.makeAprediction(i) for i in X]
        return np.array(predictions)
    #Este metodo hace las predicciones de cada registro.
    def makeAprediction(self,i):
        #Calcular las distancias
        distances = [euclidean_distance(i,x)  for x in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
#Leer el csv y dividir los datos en datos de entrenamiento y prueba
df = pd.read_csv("winequality-red.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=78)

train_x = train_df.iloc[:, :-1].values
test_x = test_df.iloc[:, :-1].values

train_y = np.array(train_df["quality"])
test_y = np.array(test_df["quality"])

#Hacer objeto de modelo KNN y hacerle fit con los datos de entrenamiento
k = 1
model = KNN(k)
model.fit(train_x,train_y)

#Hacer la prediccion usando los datos de test y revisar la accuracy de las predicciones
predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions)

#Imprimir los resultados
print(f"K: {k}")
print("KNN Accuracy:", accuracy)

print("Predicted quality:")
print(predictions)
print("Actual quality:")
print(test_y)
