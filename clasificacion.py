from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import random 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# descargamos el dataset
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

# data contiene las imágenes y target los valores
X, y = mnist["data"].values, mnist["target"].values
print(X.shape, y.shape)

# visualizamos una de estas imágenes
ix = random.randint(0, len(X)-1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()

# separamos 60.000 para entrenar y 10.000 para el test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# creamos unas nuevas listas para que nos devuelva true en caso de una imágen el número 5
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# entrenamos el modelo con un máximo de 1000 iteraciones pasando los datos de train y el del número 5
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# mostramos la predicción con una figura indicando true en caso de ser 5
# false en caso de ser cualquier otro
ix = random.randint(0, len(X)-1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()
print(sgd_clf.predict([some_digit]))


# ---- Métricas de clasificación ----

# entrenamos con validación cruzada 3 modelos distintos usando la métrica de accuracy
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("clasificador")
print(scores)

# creamos un clasificador personalizado 
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

# hacemos que el clasificador personalizado que siempre dice que la imagen no es un 5, pasándolo por el modelo anterior
# resultado peor que con el clasificador anterior
never_5_clf = Never5Classifier()
scores = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("clasificador personalizado")
print(scores)

# matriz de confusión

score = y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# recision vs Recall

score = precision_score(y_train_5, y_train_pred)
print("Recision")
print(score)

score = recall_score(y_train_5, y_train_pred)
print("Recall")
print(score)

# f1 score

score = f1_score(y_train_5, y_train_pred)
print("F1 score")
print(score)

# recision vs recall tradeoff

# nos devuelve un número
y_scores = sgd_clf.decision_function([some_digit])
y_scores

# al que podemos usar para determinar su etiqueta respecto al threshold
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)


# la curva ROC

# pasamos primero todo el conjunto de dataset de test
y_scores = sgd_clf.decision_function(X_train)

# instanciamos la curva roc con los datos de test y puntuación
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# pintamos el gráfico 
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                         
plot_roc_curve(fpr, tpr)              
plt.show()


# ---- Clasificación Multiclase ----

# OvO (One versus One)
# entrenamos un modelo con lo primeros 1000 datos y predice un dígito
svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])
svm_clf.predict([some_digit])

# extraemos los valores de la puntuación de predicción
some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)

# obtenemos el valor más alto
print(np.argmax(some_digit_scores))



# OVR (One versus Rest)
# entrenamos un modelo con los primeros 1000 datos y predice un dígito
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
print(ovr_clf.predict([some_digit]))


# entrenamiento y predicción con sgdclassifier
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))
