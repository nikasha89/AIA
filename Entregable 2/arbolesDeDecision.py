from sklearn.tree import DecisionTreeClassifier, export_graphviz
import clasificationSystem
import regresionLogistica
from sklearn.cross_validation import train_test_split

X_names = clasificationSystem.target_names
# Obtenemos datos del entrenamiento:
X_train, X_test, y_train, y_test = train_test_split(clasificationSystem.XmedicalData, clasificationSystem.ymedicalData, test_size=0.75)
# Preparamos el formato de los datos obtenidos del paso anterior:
X_train, y_train = clasificationSystem.prepare_to_fit(clasificationSystem.example, X_train, y_train)
#Declaramos el tipo de clasificador
clasificador2 = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=5).fit(X_train, y_train)

instancia = clasificationSystem.Xn_train



export_graphviz(clasificador2, feature_names=X_names, out_file="medicalData.dot")

print("----------------------------------------------------------------------------------------------------------------")
print("Árboles de Decisión ")
print("----------------------------------------------------------------------------------------------------------------")
print("Parámetros Ajustados: ")
print("Datos del Clasificador: \n" + str((clasificador2)))
print("----------------------------------------------------------------------------------------------------------------")
print("Árboles de Decisión - Predicción crosstab")
print("----------------------------------------------------------------------------------------------------------------")
clasificationSystem.cross_validation_(clasificador2, X_train,y_train,6)



#Anexo
#Método predict Standart Scaler:
def predict_Standart_Scaler_Trees():
    print("----------------------------------------------------------------------------------------------------------------")
    print("Árboles de Decisión - Método Predict SGDClassifier")
    print("----------------------------------------------------------------------------------------------------------------")
    print("Prediccion: {}".format(clasificador2.predict(instancia)))
    print("Prediccion_proba: {}".format(clasificador2.predict_proba((instancia))))
    print("Score: {}".format(clasificador2.score(X_train, y_train)))
