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

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB

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

        self.atributo_clasificacion=atributo_clasificacion
        self.clases = clases
        self.atributos=atributos
        self.valores_atributos=valores_atributos


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
        abstract

    def clasifica(self, ejemplo):
        """
        Método genérico para clasificación de un ejemplo, una vez entrenado el
        clasificador. Deberá ser definido para cada clasificador en particular.

        Si se llama a este método sin haber entrenado previamente el
        clasificador, debe devolver un excepción ClasificadorNoEntrenado
        (introducida más abajo) 
        """
        abstract

# Excepción que ha de devolverse si se llama al método de clasificación antes de
# ser entrenado  
        
class ClasificadorNoEntrenado(Exception): pass
    
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Naive Bayes
# ----------------------------------------------------------------------------

# Implementar los métodos Naive Bayes de entrenamiento (con ajuste) y
# clasificación


# LEER LOS COMENTARIOS AL CÓDIGO

class ClasificadorNaiveBayes(MetodoClasificacion):

    def __init__(self,atributo_clasificacion,clases,atributos,valores_atributos,k=1):

        """ 
        Los argumentos de entrada al constructor son los mismos que los de la
        clase genérica, junto con un parámetro k (cuyo valor por defecto es
        uno). Esta "k" es la que se tomará para el suavizado de Laplace,
        siempre que en el entrenamiento no se haga autoajuste (en ese caso, se
        tomará como "k" la que se decida en autoajuste).
        """

        # *********** COMPLETA EL CÓDIGO **************
        MetodoClasificacion.__init__(self,atributo_clasificacion,clases,atributos,valores_atributos)
        self.k = k
        
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

        Durante el ajuste, imprimir por pantalla los distintos rendimientos
        que se van obteniendo, y el "k" finalmente escogido 

        Si "autoajuste" es False, para el suavizado se tomará el "k" que se ha
        dado como argumento del constructor de la clase.

        Tener en cuenta que los ejemplos (tanto de entrenamiento como de
        clasificación) podrían tener algunos atributos con valores
        desconocidos. En ese caso, simplemente ignorar esos valores (pero no
        ignorar el ejemplo).
        """

        # *********** COMPLETA EL CÓDIGO **************

    #
    def clasifica(self,ejemplo):

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


        # *********** COMPLETA EL CÓDIGO **************
        switcherExample = {
            0: "Ejemplo 1: Clasificador Básico Naive Bayes--> Caso más sencillo con datos manuales",
            1: "Ejemplo 2: Clasificador Básico Naive Bayes--> Caso Datos de Iris",
            2: "Ejemplo 3: Clasificador Multnomial Naive Bayes --> Caso más sencillo con datos random",
        }
        if ejemplo in switcherExample:
            if ejemplo == 0:
                #Definimos los datos ejemplo:
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
                print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
            elif ejemplo == 2:
                X = np.random.randint(5, size=(6, 100))
                y = np.array([1, 2, 3, 4, 5, 6])
                clf = MultinomialNB()
                clf.fit(X, y)
                MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
                print(clf.predict(X[2:3]))


# ---------------------------------------------------------------------------
