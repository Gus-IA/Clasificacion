from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import random 
from sklearn.linear_model import SGDClassifier

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