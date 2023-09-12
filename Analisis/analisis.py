import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

df = pd.read_csv("winequality_red_revised.csv")

X = df.drop('quality', axis=1)
y = df['quality']


def fit():

    df = pd.read_csv("winequality_red_revised.csv")

    X = df.drop('quality', axis=1)
    y = df['quality']

    tree = DecisionTreeClassifier(max_depth = 8)

    #Calcular la curva de aprendizaje del modelo
    train_sizes, train_accuracy, test_accuracy = learning_curve(tree, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calcular las medias de la accuracy de ambos datasets
    train_mean = np.mean(train_accuracy, axis=1)
    test_mean = np.mean(test_accuracy, axis=1)

    # Crear gráfico para la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, marker='o', label='Precicsion de Entrenamiento')
    plt.plot(train_sizes, test_mean, marker='o', label='Precision de Prueba')

    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Precisión')
    plt.title('Curva de Aprendizaje')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


#Funcion para graficar varianza
def varianza():

    df = pd.read_csv("winequality_red_revised.csv")

    X = df.drop('quality', axis=1)
    y = df['quality']

    x_values = np.arange(1, 51)

    for i in range(50,53):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=i)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=i)
        tree = DecisionTreeClassifier(max_depth = 8)
        tree.fit(X_train,y_train)
        y_train = y_train[:50]
        y_test = y_test[:50]
        predict_test = tree.predict(X_test)
        predict_test = predict_test[:50]
        plt.figure(figsize=(50,10))
        plt.scatter(x_values,predict_test, label='predicted values')
        plt.plot(x_values, predict_test, color='black', label='model')
        plt.scatter(x_values, y_test, label='real data')
        plt.title('Varianza del Modelo')
        plt.legend()
        plt.show()
        print(accuracy_score(y_test,predict_test))



# tree = DecisionTreeClassifier(max_depth = 8)

# tree.fit(X_train,y_train)

# predictions_test = tree.predict(X_test)
# accuracy_test = accuracy_score(y_test, predictions_test)

# predictions_val = tree.predict(X_val)
# accuracy_val = accuracy_score(y_val,predictions_val)

# predictions_train = tree.predict(X_train)
# accuracy_train = accuracy_score(y_train,predictions_train)

# labels = ['Entrenamiento', 'Prueba', 'Validación']
# accuracies = [accuracy_train, accuracy_test, accuracy_val]

# plt.bar(labels, accuracies)
# plt.ylim(0, 1)  # Ajusta el rango del eje y
# plt.ylabel('Precisión')
# plt.title('Comparación de Precisión en Diferentes Conjuntos')
# plt.show()






# plt.hist(y_train, bins=3, edgecolor='black')
# plt.xlabel('Calidad del Vino')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de la Calidad del Vino en el Conjunto de Entrenamiento')
# plt.show()

# plt.hist(y_test, bins=3, edgecolor='black')
# plt.xlabel('Calidad del Vino')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de la Calidad del Vino en el Conjunto de Testing')
# plt.show()

# plt.hist(y_val, bins=3, edgecolor='black')
# plt.xlabel('Calidad del Vino')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de la Calidad del Vino en el Conjunto de Validation')
# plt.show()

fit()