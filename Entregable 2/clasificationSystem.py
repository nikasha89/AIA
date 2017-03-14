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

#Define days to convert data later:
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

def cross_validation_(clasificator, data_x,data_y, k = 5):
    model = Pipeline([('normalizador', StandardScaler()), ('modelo', clasificador)])
    kfold = KFold(data_x.shape[0], k, shuffle=True)
    values = cross_val_score(model, data_x, data_y, cv=kfold)
    predicted = cross_val_predict(model, data_x, data_y, cv=kfold)

    print("Número de muestras: " + str(len(data_x)))
    print("Valoraciones Obtenidas:", values)
    print("Media Valoraciones Obtenidas:", np.mean(values))
    print("Error Valoraciones Obtenidas:", scipy.stats.sem(values))
    print("Exactitud Real vs Estimado:",metrics.accuracy_score(data_y, predicted))
    print("Matriz de confusión:\n", metrics.confusion_matrix(data_y, predicted))
    print("Resumen:\n", metrics.classification_report(data_y, predicted))

    #representacion_grafica(data_x, data_y, target_names, y_names)

def representacion_grafica(x_data, y_data,x_name, y_name,x_limit =7, y_limit = 7):
    for tipo, marca, color in zip(range(2), "soD", "rgb"):
        plt.scatter(x_data[:, x_limit][int(y_data == tipo)], x_data[:, y_limit][int(y_data == tipo)], marker=marca,c=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)


plt.show()

def load_csv():
    with open('prueba.csv') as csvarchivo:
        entrada = csv.reader(csvarchivo)
        for reg in entrada:
            medicalData.append(reg)
def prepare_to_fit(example, X_train_,y_train_):
    warnings.filterwarnings("ignore")
    if example==0:
        #Prepare X-train:
        for i in range(len(X_train_)):
            for j in range(len(X_train_[i])):
                if j == 1:
                    X_train_[i][1] = day[X_train_[i][1]]
    elif example == 1 or example == 2 or example == 3:
        # Prepare X-train:
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                X_train_[i][0] = float(X_train_[i][0])
                X_train_[i][1] = float(X_train_[i][1])
    elif example == 4:
        for i in range(len(X_train_)):
            for j in range(len(X_train_[i])):
                if j == 1:
                    X_train_[i][1] = day[X_train_[i][1]]
                else:
                    X_train_[i][j] = float(X_train_[i][j])

    #prepare Y_train:
    for i in range(len(y_train_)):
        #show-up = 1; else 0:
        showUp = 1.0
        if y_train_[i] == "No-Show":
            showUp = 0.0
        y_train_[i] = showUp
    return X_train_,y_train_

# Conjuntos posibles en los que clasificar los datos:
y_names = ["Show Up", "No Show"]
load_csv()
example = 4
columna5_estado = np.array(medicalData)[1:, 5]
ymedicalData = columna5_estado
XmedicalData = []
target_names = []
if example == 0:
    #Obtenemos los datos de edad y dia semana:
    columnas_edad_dayWeek = np.array(medicalData)[1:,[0,4]]
    XmedicalData = columnas_edad_dayWeek
    target_names = ["Age","DiaSemana"]
elif example == 1:
    columnas_diabetes_tuberculosis = np.array(medicalData)[1:,[7,12]]
    XmedicalData = columnas_diabetes_tuberculosis
    target_names = ["diabetes","tuberculosis"]
elif example == 2:
    columnas_smsreminder_awaitingtime = np.array(medicalData)[1:,[13,14]]
    XmedicalData = columnas_smsreminder_awaitingtime
    target_names = ["SMSReminder","Awaiting Time"]
elif example == 3:
    columnas_hipertension_minusvalia = np.array(medicalData)[1:, [10, 9]]
    XmedicalData = columnas_hipertension_minusvalia
    target_names = ["Hipertensión", "Minusvalía"]
    #Todos los datos posibles:
elif example == 4:
    columnas_diabetes_tuberculosis = np.array(medicalData)[1:, [0,4,7,12,10, 9, 13, 14]]
    XmedicalData = columnas_diabetes_tuberculosis
    target_names = ["Age","DiaSemana","diabetes","tuberculosis","SMSReminder","Awaiting Time","Hipertensión", "Minusvalía"]
# Conjuntos posibles en los que clasificar los datos:
# Obtenemos datos del entrenamiento
X_train, X_test, y_train, y_test = train_test_split(XmedicalData, ymedicalData, test_size=0.75)
# Preparamos el formato de los datos obtenidos del paso anterior:
X_train, y_train = prepare_to_fit(example, X_train, y_train)



#Creamos el clasificador lineal
clasificador = SGDClassifier().fit(X_train,y_train)
#Normalizamos lo datos obtenidos del entrenamiento:
Xn_train = clasificador.transform(X_train)

print("Datos usados: "+str(target_names))
print("----------------------------------------------------------------------------------------------------------------")
print("Clasificador Lineal")
print("----------------------------------------------------------------------------------------------------------------")
print("Parámetros Ajustados: ")
print("Datos del Clasificador: \n" + str((clasificador)))
print("----------------------------------------------------------------------------------------------------------------")
print("Clasificador Lineal - Predicción crosstab")
print("----------------------------------------------------------------------------------------------------------------")
cross_validation_(clasificador,X_train,y_train)


#Anexo:
#Método predict SGDClassifier:
# Entrenamos el conjunto escogido de datos:
fitNormalizado = StandardScaler().fit(X_train, y_train)
# Normalizamos lo datos obtenidos del entrenamiento:
Xn_train_StandardScaler = fitNormalizado.transform(X_train)
def predict_Standart_Scaler():
    print("----------------------------------------------------------------------------------------------------------------")
    print("Clasificador Lineal - Método Predict Standart Scaler")
    print("----------------------------------------------------------------------------------------------------------------")
    #Ahora predecirmos los datos con un nuevo clasificador para esos datos normalizados:
    y_train_predicted = clasificador.predict(Xn_train_StandardScaler)
    print("Parámetros Ajustados: ")
    print("Número de muestras: " + str(fitNormalizado.n_samples_seen_))
    print("Normalización mean --> " + str(fitNormalizado.mean_))
    print("Normalización std --> " + str(fitNormalizado.std_))
    print("Media --> " + str(np.mean(Xn_train_StandardScaler)))
    print("Desviación --> " + str(np.std(Xn_train_StandardScaler)))
    #Imprimimos las estadísticas del experimento:
    precision = metrics.accuracy_score(y_train, y_train_predicted)
    print('Exactitud: {}'.format(precision))
    print("\n Resumen"+metrics.classification_report(y_train, y_train_predicted))
    #representacion_grafica(Xn_train,X_test,y_test,y_names,0,1)
    print('Matriz de confusión \n{}'.format(metrics.confusion_matrix(y_train, y_train_predicted)))
