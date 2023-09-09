import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
        #Calcula las distancias usando el metodo de distancia euclideana
        distances = [euclidean_distance(i,x)  for x in self.X_train]
        
        #Aqui usamos un sort para ordernar la lista de distancias en orden ascendiente y elegimos las primeras
        #K distancias
        k_indices = np.argsort(distances)[:self.k]
        
        #Aqui extraemos la salida de los valores K mas cercanos y los metemos en una lista.
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #Finalmente sacamos el valor de salida mas comun dentro de nuestra lista de valores mas cercanos y
        #lo regresamos como nuestra respuesta de predicción.
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
#Leer el csv y dividir los datos en datos de entrenamiento y prueba
df = pd.read_csv("winequality-red.csv")

#Para demostrar que el modelo genraliza se hace un loop para probar con diferentes valores para el random_state
#del split del dataset. Esto con el objetivo de ver modelo hacer predicciones con diferentes datos de entrenamiento
#y de validación
for i in range(50,75):
    print(i)

    #Dividimos el dataset en uno de entrenamiento y otro de validación. Usamos el numero "i" como
    #semilla para el random split.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=i)

    #Dividimos los datasets de entrenamiento en 2, los valores de entrada y los valores de salida.
    train_x = train_df.iloc[:, :-1].values
    test_x = test_df.iloc[:, :-1].values

    train_y = np.array(train_df["quality"])
    test_y = np.array(test_df["quality"])



    #Hacer objeto de modelo KNN y hacerle fit con los datos de entrenamiento, suponemos una K = 1
    k = 1
    model = KNN(k)
    model.fit(train_x,train_y)

    #Hacer la prediccion usando los datos de test
    predictions = model.predict(test_x)

    #Usamos sklearn para rapidamente calcular la precición del modelo, asi como
    #la matriz de confusión, el recall y el f1 para tener buenas metricas para evaluar el modelo.
    accuracy = accuracy_score(test_y, predictions)
    confusion_matrix_result = confusion_matrix(test_y, predictions)
    recall = recall_score(test_y, predictions, average='macro')
    f1 = f1_score(test_y, predictions, average='macro')


    #Imprimir los resultados
    print(f"K: {k}")
    print("KNN Accuracy:", accuracy)
    print()
    print("Matriz de Confusion:")
    print(confusion_matrix_result)
    print()
    print("Recall:")
    print(confusion_matrix_result)
    print()
    print("F1:")
    print(confusion_matrix_result)




