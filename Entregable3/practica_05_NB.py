# MII-AIA 2016-17
# Práctica del tema 6 - Parte 0 
# =============================

# Este trabajo está inspirado en el proyecto "Classification" de The Pacman
# Projects, desarrollados para el curso de Introducción a la Inteligencia
# Artificial de la Universidad de Berkeley.

# Se trata de implementar el algoritmo Naive Bayes y de aplicarlo a dos
# problemas de aprendizaje para clasificación automática. Estos problemas son:
# adivinar el partido político (republicano o demócrata) de un congresista USA
# a partir de lo votado a lo largo de un año, y reconocer un dígito a partir de
# una imagen del mismo escrito a mano.

# Conjuntos de datos
# ==================

# En este trabajo se manejarán dos conjuntos de datos, que serán usados para
# probar la implementación. A su vez cada conjunto de datos se distribuye en
# tres partes: conjunto de entrenamiento, conjunto de validación y conjunto de
# prueba. El primero de ellos se usará para el aprendizaje, el segundo para
# ajustar determinados parámetros de los clasificadores que finalmente se
# aprendan, y el tercero para medir el rendimiento de los mismos.

# Los datos que usaremos son:

#  - Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#    17 votaciones realizadas durante 1984. En votes.py están estos datos, en
#    formato python. Este conjunto de datos está tomado de UCI Machine Learning
#    Repository, donde se puede encontrar más información sobre el mismo. Nótese
#    que en este conjunto de datos algunos valores figuran como desconocidos.

#  - Un conjunto de imágenes (en formato texto), con una gran cantidad de
#    dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#    base de datos MNIST. En digitdata.zip están todos los datos en formato
#    comprimido. Cada imagen viene dada por 28x28 píxeles, y cada pixel vendrá
#    representado por un caracter "espacio en blanco" (pixel blanco) o los
#    caracteres "+" (borde del dígito) o "#" (interior del dígito). En nuestro
#    caso trataremos ambos como un pixel negro (es decir, no distinguiremos
#    entre el borde y el interior). En cada conjunto, las imágenes vienen todas
#    seguidas en un fichero de texto, y las clasificaciones de cada imagen (es
#    decir, el número que representan) vienen en un fichero aparte, en el mismo
#    orden. Será necesario, por tanto, definir funciones python que lean esos
#    ficheros y obtengan los datos en el mismo formato python en el que se dan
#    los datos del punto anterior.

# Implementación del clasificador Naive Bayes
# ===========================================

# La implementación de ambos algoritmos deberá realizarse completando el código
# que se da más abajo, siguiendo las indicaciones que aparecen en el mismo.

# Aunque el código se aplicará a los conjuntos de datos anteriores, debe
# realizarse de manera independiente, para que sea posible aplicarlo a
# cualquier otro ejemplo de clasificación.

# Implementar el algoritmo Naive Bayes, tal y como se ha descrito en clase,
# usando suavizado de Laplace y logaritmos. La fase de ajuste en Naive Bayes
# consiste en encontrar el mejor k para el suavizado, de entre un conjunto
# de valores candidatos, probando los distintos rendimientos en el conjunto
# de validación (ver detalles en los comentarios del código).

# El algoritmo debe poder tratar ejemplos con valores desconocidos en algún
# atributo (como los que aparecen en el caso de los votos). Para ello,
# simplemente ignorarlos (tanto para entrenamiento como para clasificación).

# Se pide dar el rendimiento (proporción de aciertos) de cada clasificador
# sobre el conjunto de prueba proporcionado. Mostrar y comentar los resultados
# (incluyéndolos como comentarios al código). En todos los casos, un
# rendimiento aceptable debería estar por encima del 70% de aciertos sobre el
# conjunto de prueba.

# ----------------------------------------------------------------------------

# "*********** COMPLETA EL CÓDIGO **************"

# ----------------------------------------------------------------------------
# Clase genérica MetodoClasificacion
# ----------------------------------------------------------------------------

# EN ESTA PARTE NO SE PIDE NADA, PERO ES NECESARIO LEER LOS COMENTARIOS DEL
# CÓDIGO. 

# Clase genérica para los métodos de clasificación. Los métodos de
# clasificación que se piden deben ser subclases de esta clase genérica. 

# NO MODIFICAR ESTA CLASE.
#import warnings
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
import math
import votes
import copy

#warnings.filterwarnings("ignore")

class MetodoClasificacion:
    """
    Clase base para métodos de clasificación
    """

    def __init__(self, atributo_clasificacion,clases,atributos,valores_atributos):

        """
        Argumentos de entrada al constructor (ver un caso concreto en votos.py)
         
        * atributo_clasificacion: es el atributo con los valores de clasificación. 
        * clases: lista de posibles valores del atributo de clasificación.  
        * atributos: lista de atributos, excepto el de clasificación. También
                    denominados "características". 
        * valores_atributos: diccionario que a cada atributo le asigna la
                             lista de sus posibles valores 
        """
        self.atributo_clasificacion = atributo_clasificacion
        self.clases = clases
        self.atributos = atributos
        self.valores_atributos = valores_atributos


    def entrena(self,entr,clas_entr,valid,clas_valid,autoajuste):
        """
        Método genérico para entrenamiento y ajuste del clasificador. Deberá
        ser definido para cada clasificador en particular. 
        
        Argumentos de entrada (ver un ejemplo en votos.py):

        * entr: ejemplos del conjunto de entrenamiento (sin incluir valor de
                clasificación) 
        * clas_entr: valores de clasificación de los ejemplos del conjunto de
                     entrenamiento
        * valid: ejemplos del conjujnto de validación (sin incluir valor de
                 clasificación)
        * clas_valid: valores de clasificación de los ejemplos del conjunto de 
                      validación
        * autoajuste: booleano para indicar si se hace autoajuste
        
        """
        pass

    def clasifica(self, ejemplo):
        """
        Método genérico para clasificación de un ejemplo, una vez entrenado el
        clasificador. Deberá ser definido para cada clasificador en particular.

        Si se llama a este método sin haber entrenado previamente el
        clasificador, debe devolver un excepción ClasificadorNoEntrenado
        (introducida más abajo) 
        """
        pass

# Excepción que ha de devolverse si se llama al método de clasificación antes de
# ser entrenado  
        
class ClasificadorNoEntrenado(Exception): pass
    
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Naive Bayes
# ----------------------------------------------------------------------------
0
# Implementar los métodos Naive Bayes de entrenamiento (con ajuste) y
# clasificación


class ClasificadorNaiveBayes(MetodoClasificacion):

    def __init__(self, atributo_clasificacion, clases, atributos, valores_atributos,k=1):

        """ 
        Los argumentos de entrada al constructor son los mismos que los de la
        clase genérica, junto con un parámetro k (cuyo valor por defecto es
        uno). Esta "k" es la que se tomará para el suavizado de Laplace,
        siempre que en el entrenamiento no se haga autoajuste (en ese caso, se
        tomará como "k" la que se decida en autoajuste).
        """
        super().__init__(atributo_clasificacion,clases,atributos,valores_atributos)
        self.k = float(k)
        self.P_vj_ai = {}
        self.P_vj = {}
        self.test_class = None

    #Realiza las predicciones en base a los datos de entenamiento
    def SimpleNB(self, ejemplo):
        listPredicciones = []
        for row in range(len(ejemplo)):
            best_prob = None
            prediccion = None
            # Accedemos a la tabla de probabilidades de cada caracteristica
            for clas, atr in self.P_vj_ai.items():
                operation = 0.0
                # Comprobamos si el valor de la caracteristica existe y extraemos la probabilidad
                for column in range(len(ejemplo[row])):
                    if ejemplo[row][column] in self.valores_atributos[self.atributos[column]]:
                        indexAtr = self.valores_atributos[self.atributos[column]].index(ejemplo[row][column])
                        operation += atr[self.atributos[column]][indexAtr]
                prob = self.P_vj[clas] + operation
                if best_prob is None or prob > best_prob:
                    best_prob = prob
                    prediccion = clas
            listPredicciones.append(prediccion)

        return listPredicciones

    # Devuelve la precision de las predicciones con respecto al test pasados en método entrena:
    def getAccuracy(self, predictions):
        testSet = self.test_class
        correct = 0
        if(len(testSet)<len(predictions)):
            for i in range(len(testSet)):
                if testSet[i] == testSet[i]:
                    correct += 1
        else:
            for i in range(len(predictions)):
                if testSet[i] == predictions[i]:
                    correct += 1
        return round((correct / float(len(testSet))) * 100.0,3)


    def entrena(self,entr,clas_entr,valid,clas_valid,autoajuste=True):

        """ 
        Método para entrenamiento de Naive Bayes, que estima las probabilidades
        a partir del conjunto de entrenamiento y las almacena en forma
        logarítmica. A la estimación de las probabilidades se ha de aplicar
        suavizado de Laplace.  

        Si "autoajuste" es True (valor por defecto), el parámetro "k" del
        suavizado ha de elegirse de entre los siguientes valores candidatos,
        según su rendimiento sobre el conjunto de validación:
        
        [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100] 

        que se van obteniendo, y el "k" finalmente escogido 
        Durante el ajuste, imprimir por pantalla los distintos rendimientos

        Si "autoajuste" es False, para el suavizado se tomará el "k" que se ha
        dado como argumento del constructor de la clase.

        Tener en cuenta que los ejemplos (tanto de entrenamiento como de
        clasificación) podrían tener algunos atributos con valores
        desconocidos. En ese caso, simplemente ignorar esos valores (pero no
        ignorar el ejemplo).
        """
        # suavizado laplace y logaritmo:
        #Clasificación en base a clases
        # Obtenemos en un diccionario todas las probabilidades asociadas a cada valor:
        # dict_P_vj_ai = collections.Counter(arAux)
        # *********** COMPLETA EL CÓDIGO **************
        # count occurrences of values per row (clase):
        dict_P_vj = {}
        for elem in range(len(clas_entr)):
            if clas_entr[elem] in self.clases:
                if clas_entr[elem] in dict_P_vj.keys():
                    dict_P_vj[clas_entr[elem]] += 1.0
                else:
                    dict_P_vj[clas_entr[elem]] = 1.0

        counts = {}
        #Inicializamos counts para realizar el conteo de los tipos de atributos por cada clase:
        for clas in self.clases:
            counts[clas] = {}
            for atr in self.atributos:
                votes = []
                for pos in range(len(self.valores_atributos[atr])):
                    votes.append(1)
                counts[clas][atr] = votes
        #Guardamos las ocurrencias de cada valor de cada tipo para cada atributo de cada clase:
        for row in range(len(entr)):
            for column in range(len(entr[row])):
                atr = self.atributos[column]
                voto = entr[row][column]
                if voto in self.valores_atributos[atr]:
                    indexAtr = self.valores_atributos[atr].index(voto)
                    counts[clas_entr[row]][self.atributos[column]][indexAtr] += 1

        #Si hay auto ajuste realizamos búsqueda de mejor k:
        if(autoajuste == True):
            self.k = autoAjuste(self.atributo_clasificacion, self.clases, self.atributos, self.valores_atributos,entr,clas_entr,valid,clas_valid)
            print("\n----------------------------")
            print("----- Mejor Valor de K -----")
            print("----------------------------\n")

        #Realizamos el cálculo de sus probabilidades aplicando logaritmo
        for clas in counts.keys():
            dicAux = counts[clas]
            for vote in dicAux.keys():
                list_aux = []
                k = 0
                for poss in range(len(self.valores_atributos[vote])):
                    log_P_vj_aj = math.log((dicAux[vote][poss] + self.k) / (dict_P_vj[clas] + len(self.valores_atributos[vote]) * self.k))
                    list_aux.append(log_P_vj_aj)
                    k += 1
                dict_prob_aux = {}
                if clas in self.P_vj_ai:
                    dict_prob_aux = self.P_vj_ai[clas]
                dict_prob_aux.__setitem__(vote, list_aux)
                self.P_vj_ai.__setitem__(clas, dict_prob_aux)



        #Guardamos la probabilidad de cada clase aplicando logaritrmo:
        for key in dict_P_vj.keys():
            dict_P_vj[key] = math.log(dict_P_vj[key] / len(entr))

        #Guargadmos los logaritmos de los conteos (con su suavizado k correspondiente) por clases en:
        self.P_vj = dict_P_vj

    def clasifica(self, ejemplo, clase):

        """ 
        Método para clasificación de ejemplos, usando el clasificador Naive
        Bayes obtenido previamente mediante el entrenamiento.

        Si se llama a este método sin haber entrenado previamente el
        clasificador, debe devolver una excepción ClasificadorNoEntrenado

        Tener en cuenta que los ejemplos (tanto de entrenamiento como de
        clasificación) podrían tener algunos atributos con valores
        desconocidos. En ese caso, simplimente ignorar esos valores (pero no
        ignorar el ejemplo).
        """
        self.test_class = clase
        if(self.P_vj_ai != {}):
            clasificacion = None
            #Usamos método SimpleNB:
            clasificacion = self.SimpleNB(ejemplo)
            print("Valor de k: ",self.k)
            print("Precisión: "+str(self.getAccuracy(clasificacion)) +"%\n")
        else:
            raise ClasificadorNoEntrenado("No has entrenado el clasificador")
        return clasificacion

#Funciones útiles:

def autoAjuste(atributo_clasificacion, clases, atributos, valores_atributos, entr,clas_entr,valid,clas_valid):
    print("\n----------------------------")
    print("-------- Autoajuste --------")
    print("----------------------------\n")
    list_k = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
    best_k = None
    best_prec = None
    for k in list_k:
        p = ClasificadorNaiveBayes(atributo_clasificacion, clases, atributos, valores_atributos,k=k)
        p.entrena(entr,clas_entr,valid,clas_valid,autoajuste=False)
        precision = p.getAccuracy(p.clasifica(valid,clas_valid))
        if(best_k is None or best_prec is None or best_prec<precision):
            best_k = k
            best_prec = precision
    return best_k

#Función que lee los atributos de los datos de un archivo:
def readAtr(pathToFile):
    res = []
    max_count = 28
    count = 0
    with open(pathToFile) as row:
        atr = []
        for data in row:
            if count == max_count:
                c = atr[:]
                count = 0
                res.append(c)
                del atr[:]
            line = data.split("\n")
            for char in line:
                for val in char:
                    if val == ' ':
                        atr.append(0)
                    elif val == '+' or val == '#':
                        atr.append(1)
            count += 1
    return res

#Función que lee los valores de los datos de un archivo:
def readValues(pathToFile):
    res = []
    with open(pathToFile) as raw_data:
        for item in raw_data:
            res.append(item.split("\n")[0])
    return res

def otrosEjemplos(ejemplo):
    switcherExample = {
        0: "Ejemplo 1: Clasificador Gaussiano Naive Bayes--> Caso más sencillo con datos manuales",
        1: "Ejemplo 2: Clasificador Gaussiano Naive Bayes--> Caso Datos de Iris",
        2: "Ejemplo 3: Clasificador Multnomial Naive Bayes --> Caso más sencillo con datos random",
    }
    if ejemplo in switcherExample:
        if ejemplo == 0:
            # Definimos los datos ejemplo:
            X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
            Y = np.array([1, 1, 1, 2, 2, 2])
            # Definimos el clasificador Naive Bayes:
            clf = GaussianNB()
            clf.fit(X, Y)
            GaussianNB(priors=None)
            print(clf.predict([[-0.8, -1]]))
            clf_pf = GaussianNB()
            clf_pf.partial_fit(X, Y, np.unique(Y))
            GaussianNB(priors=None)
            print(clf_pf.predict([[-0.8, -1]]))
        elif ejemplo == 1:
            iris = datasets.load_iris()
            gnb = GaussianNB()
            y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
            print("Number of mislabeled points out of a total %d points : %d" % (
            iris.data.shape[0], (iris.target != y_pred).sum()))
        elif ejemplo == 2:
            X = np.random.randint(5, size=(6, 100))
            y = np.array([1, 2, 3, 4, 5, 6])
            clf = MultinomialNB()
            clf.fit(X, y)
            MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
            print(clf.predict(X[2:3]))

# ---------------------------------------------------------------------------


#Ejemplo Votos:
print("\n----------------------------------------------------------------------------------------------------------")
print("-------------------------------------------- EJEMPLO VOTOS -----------------------------------------------")
print("----------------------------------------------------------------------------------------------------------\n")
p = ClasificadorNaiveBayes(votes.votos_atributo_clasificacion, votes.votos_clases,votes.votos_atributos,votes.votos_valores_atributos,k=0.01)
#print(str(p._contents(p.valores_atributos)))
#print(str(p._contents(p.valores_atributos,False)))
p.entrena(votes.votos_entr,votes.votos_entr_clas,votes.votos_valid,votes.votos_valid_clas)
print("Clasificación Predecida:\n"+str(p.clasifica(votes.votos_test,votes.votos_test_clas)))

#Ejemplo Dígitos:
print("\n----------------------------------------------------------------------------------------------------------")
print("------------------------------------------- EJEMPLO DÍGITOS ----------------------------------------------")
print("----------------------------------------------------------------------------------------------------------\n")
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
digitos_atributos = []
digitos_valor_atributos = {}
digitos_atributo_clasificacion = 'digitos'
tam = 28
for i in range(tam**2):
    digitos_atributos.append('pixel' + str(i))
    digitos_valor_atributos['pixel' + str(i)] = [0, 1]
q = ClasificadorNaiveBayes(digitos_atributo_clasificacion,digits,digitos_atributos,digitos_valor_atributos)
train_data = readAtr("digitdata/trainingimages")
train_clas = readValues("digitdata/traininglabels")
val_data = readAtr("digitdata/validationimages")
val_clas = readValues("digitdata/validationlabels")
test_data = readAtr("digitdata/testimages")
test_clas = readValues("digitdata/testlabels")
q.entrena(train_data, train_clas, val_data, val_clas, True)
print("Clasificación Predecida:\n"+str(q.clasifica(test_data,test_clas)))