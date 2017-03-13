#from loader import load_no_show_issue
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import csv
import warnings
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
import scipy

#Define days to cast data later:
day = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
medicalData = []

def cross_validation_(data_x,data_y, k = 5):
    model = Pipeline([('normalizador', StandardScaler()), ('modelo', clasificador)])
    kfold = KFold(data_x.shape[0], k, shuffle=True)
    values = cross_val_score(model, data_x, data_y, cv=kfold)
    predicted = cross_val_predict(model, data_x, data_y, cv=kfold)

    print("Valoraciones:", values)
    print("Media:", np.mean(values))
    print("Error:", scipy.stats.sem(values))
    print("Precision:",metrics.accuracy_score(data_y, predicted))
    print("Matrix de con:", metrics.confusion_matrix(data_y, predicted))
    print("Resumen:", metrics.classification_report(data_y, predicted))

def load_csv():
    with open('prueba.csv') as csvarchivo:
        entrada = csv.reader(csvarchivo)
        for reg in entrada:
            medicalData.append(reg)
def prepare_to_fit():
    warnings.filterwarnings("ignore")

    #Prepare X-train:
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if j == 1:
                X_train[i][1] = day[X_train[i][1]]
    #prepare Y_train:
    for i in range(len(y_train)):
        #show-up = 1; else 0:
        showUp = 1.0
        if y_train[i] == "No-Show":
            showUp = 0.0
        y_train[i] = showUp
def representacion_grafica(datos,caracteristicas,objetivo,clases,c1,c2):
    for tipo,marca,color in zip(range(len(clases)),"soD","rgb"):
        plt.scatter(datos[objetivo == tipo,c1],datos[objetivo == tipo,c2],marker=marca,c=color)
    plt.xlabel(caracteristicas[c1])
    plt.ylabel(caracteristicas[c2])
    plt.legend(clases)
    plt.show()

load_csv()
#Obtenemos los datos de edad y dia semana:
columnas_edad_dayWeek = np.array(medicalData)[1:100,[0,4]]
columna5_estado = np.array(medicalData)[1:100,5]
XmedicalData = columnas_edad_dayWeek
ymedicalData = columna5_estado
target_names = ["Age","DiaSemana"]
# Conjuntos posibles en los que clasificar los datos:
y_names = ["Show Up", "No Show"]
#Obtenemos datos del entrenamiento
X_train, X_test, y_train, y_test = train_test_split(XmedicalData,ymedicalData,test_size = 0.25)

#Preparamos el formato de los datos obtenidos del paso anterior:
prepare_to_fit()
#Normalizamos el conjunto escogido de datos:
fitNormalizado = StandardScaler().fit(X_train, y_train)
Xn_train = fitNormalizado.transform(X_train)
print("----------------------------------------------------------------------------------------------------------------")
print("Clasificador Lineal")
print("----------------------------------------------------------------------------------------------------------------")
print("Datos usados: "+str(target_names))
print("Parámetros Ajustados: ")
print("Normalización mean --> "+ str(fitNormalizado.mean_))
print("Normalización std --> "+ str(fitNormalizado.std_))
print("Media --> "+str(np.mean(Xn_train)))
print("Desviación --> "+str(np.std(Xn_train)))
#Ahora predecirmos los datos con un nuevo clasificador para esos datos normalizados:
clasificador = SGDClassifier().fit(Xn_train,y_train)
y_train_predicted = clasificador.predict(Xn_train)

#Imprimimos las estadísticas del experimento:
print("Predecido "+str(fitNormalizado.n_samples_seen_))
precision = metrics.accuracy_score(y_train, y_train_predicted)
print('Exactitud: {}'.format(precision))
print("\n Resumen"+metrics.classification_report(y_train, y_train_predicted))
#representacion_grafica(Xn_train,X_test,y_test,y_names,0,1)
print('Matriz de confusión \n{}'.format(metrics.confusion_matrix(y_train, y_train_predicted)))
print("----------------------------------------------------------------------------------------------------------------")
print("Clasificador Lineal - Predicción crosstab")
print("----------------------------------------------------------------------------------------------------------------")
cross_validation_(X_train,y_train)