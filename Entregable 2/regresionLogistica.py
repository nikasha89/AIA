from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import clasificationSystem



X_train = clasificationSystem.X_train
X_test = clasificationSystem.X_test
y_train = clasificationSystem.y_train
y_test = clasificationSystem.y_test
X_names = clasificationSystem.target_names
normalizador = clasificationSystem.fitNormalizado


Xn_data = normalizador.transform(X_train)
Xn_train = normalizador.transform(X_train)


clasificador = LogisticRegression().fit(Xn_train, y_train)
print("Score: {}".format(clasificador.score(Xn_train, y_train)))

instancia = Xn_train
print("Prediccion: {}".format(clasificador.predict(instancia)))
print("Prediccion_proba: {}".format(clasificador.predict_proba(instancia)))

print("Xn_train: \n{}".format(Xn_data[:5]))
print("Xn_train transform: \n{}".format(clasificador.transform(Xn_data[:5])))
