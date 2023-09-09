import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

#Leer el csv y dividir los datos en datos de entrenamiento y prueba
#En este caso se usa una version revisada del dataset de la actividad anterior
#i.e se removieron la mayoria de los outliers.
df = pd.read_csv("winequality_red_revised.csv")


#Para demostrar que el modelo genraliza se hace un loop para probar con diferentes valores para el random_state
#del split del dataset. Esto con el objetivo de ver modelo hacer predicciones con diferentes datos de entrenamiento
#y de validación
for i in range(50,75):

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=i)

    #Dividimos los datasets de entrenamiento y validacion en 2, los valores de entrada y los valores de salida.
    train_x = train_df.iloc[:, :-1].values
    test_x = test_df.iloc[:, :-1].values
    
    train_y = np.array(train_df["quality"])
    test_y = np.array(test_df["quality"])

    #Creamos un objeto de la clase de arbol de desicion de Sklearn
    tree = DecisionTreeClassifier(max_depth = 8)

    #Entrenamos el modelo usando nuestros datos de entrenamiento
    tree.fit(train_x,train_y)

    #Hacemos predicciones usando nuestro set de pruebas
    predictions = tree.predict(test_x)

    #Usamos sklearn para rapidamente calcular la precición del modelo, asi como
    #la matriz de confusión, el recall y el f1 para tener buenas metricas para evaluar el modelo.
    accuracy = accuracy_score(test_y, predictions)
    confusion_matrix_result = confusion_matrix(test_y, predictions)
    recall = recall_score(test_y, predictions, average='macro')
    f1 = f1_score(test_y, predictions, average='macro')

    #Imprimir los resultados
    print()
    print("Accuracy:", accuracy)
    print("Matriz de Confusion:")
    print(confusion_matrix_result)
    print("Recall:")
    print(confusion_matrix_result)
    print("F1:")
    print(confusion_matrix_result)
    print()


