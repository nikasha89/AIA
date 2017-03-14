from sklearn.linear_model import LogisticRegression
import clasificationSystem



X_train = clasificationSystem.X_train
X_test = clasificationSystem.X_test
y_train = clasificationSystem.y_train
y_test = clasificationSystem.y_test
X_names = clasificationSystem.target_names

clasificador3 = LogisticRegression().fit(X_train, y_train)

#Obtenemos una instancia ejemplo del clasificador creado anteriormente
# Obtenemos datos del entrenamiento
X_train, X_test, y_train, y_test = clasificationSystem.train_test_split(clasificationSystem.XmedicalData, clasificationSystem.ymedicalData, test_size=0.75)
X_train, y_train = clasificationSystem.prepare_to_fit(clasificationSystem.example,X_train, y_train)
instancia = clasificationSystem.fitNormalizado.transform(X_train)
#Imprimimos
print("----------------------------------------------------------------------------------------------------------------")
print("Regresión Logística ")
print("----------------------------------------------------------------------------------------------------------------")
print("Parámetros Ajustados: ")
print("Datos del Clasificador: \n" + str((clasificador3)))
print("----------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------")
print("Regresión Logística - Predicción crosstab")
print("----------------------------------------------------------------------------------------------------------------")
print("Score: {}".format(clasificador3.score(instancia, y_train)))
clasificationSystem.cross_validation_(clasificador3, X_train,y_train,10)

#Anexo
#Método predict Standart Scaler:
def predict_Standart_Scaler_Regresion():
    print("Regresión Logística - Método Predict SGDClassifier")
    print("----------------------------------------------------------------------------------------------------------------")
    print("Prediccion: {}".format(clasificador3.predict(instancia)))
    print("Prediccion_proba: {}".format(clasificador3.predict_proba(instancia)))

    print("Xn_train: \n{}".format(X_train[:5]))
    print("Xn_train transform: \n{}".format(clasificador3.transform(X_train[:5])))
