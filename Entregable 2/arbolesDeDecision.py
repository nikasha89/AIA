from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import clasificationSystem


X_train = clasificationSystem.X_train
X_test = clasificationSystem.X_test
y_train = clasificationSystem.y_train
y_test = clasificationSystem.y_test
X_names = clasificationSystem.target_names

#Declaramos el tipo de clasificador
clasificador = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)

clasificador.fit(X_train, y_train)
instancia = clasificationSystem.Xn_train

print("----------------------------------------------------------------------------------------------------------------")
print("Árboles de Decisión")
print("----------------------------------------------------------------------------------------------------------------")
print("Prediccion: {}".format(clasificador.predict(instancia)))
print("Prediccion_proba: {}".format(clasificador.predict_proba((instancia))))
print("Score: {}".format(clasificador.score(X_train, y_train)))

export_graphviz(clasificador, feature_names=X_names, out_file="iris.dot")