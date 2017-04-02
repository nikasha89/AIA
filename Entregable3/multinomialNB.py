import numpy, json, os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit
path = 'tweets/'

def leeTweet(path):
    id = ""
    data = []
    classes = []
    with open(path, mode='r') as input_file:
        for row in input_file:
            row = row.split(',')
            id = str((row[2][1:-2]))
            # Sólo se consideran los tweets con texto no vacío, lo cual denota, en función de load_json, que están en inglés
            if cargaJSON(id) != "":
                # Para cada tweet se consideran su texto y su sentimiento
                data.append(cargaJSON(id).lower())
                classes.append((str(row[1][1:-1])))
    return data, classes


def cargaJSON(id):
    data = ""
    if os.path.exists(path + "rawdata/" + id + ".json"):
        with open(path + "rawdata/" + id + ".json") as data_row:
            text = json.load(data_row)
            if text["lang"] == "en":
                data = str(text["text"])
    return data

def imprimeEstadiscticas(prediction, classes_test, target_names):
    print("\n----------------------------")
    print("-------- Estadísticas ------")
    print("----------------------------\n")
    print("\nClases: \n" + str(target_names))
    print("\nMedia: \n" + str(numpy.mean(prediction == classes_test)))
    print("\nMatriz de Confusión: \n" + str(metrics.confusion_matrix(classes_test, prediction)))
    print("\nClasificación: \n\n" + str(metrics.classification_report(classes_test, prediction, target_names)))

def entrenaYpredice(data_test, data_train, classes_train,gridSearch):
    gridSearch.fit(data_train, classes_train)
    return gridSearch.predict(data_test)

def creaGridSearchCV():
    tfidf__ngram_range = [(1, 1), (1, 2), (1, 3)]
    tfidf__stop_words = [stopwords.words("english")]
    tfidf__use_idf = [True, False]
    tfidf__smooth_idf = [True, False]
    tfidf__sublinear_tf = [True, False]
    tfidf__binary = [True, False]
    clf__alpha = [1e-2, 1e-3, 1e-4]
    params = {
        'tfidf__ngram_range': tfidf__ngram_range,
        'tfidf__stop_words': tfidf__stop_words,
        'tfidf__use_idf': tfidf__use_idf,
        'tfidf__smooth_idf': tfidf__smooth_idf,
        'tfidf__sublinear_tf': tfidf__sublinear_tf,
        'tfidf__binary': tfidf__binary,
        'clf__alpha': clf__alpha,
    }
    model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    return GridSearchCV(model, params, verbose=True, cv=ShuffleSplit(train_size=.25, n_splits=3, random_state=1))

def cargaDatos(tipo=''):
    data, classes = leeTweet(path + 'corpus.csv')
    data = np.array(data)
    classes = np.array(classes)
    target = None
    print("\n----------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------- EJEMPLO TWEET -----------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------\n")
    if tipo == '+VS-':
        data = data[(classes == 'positive') | (classes == 'negative')]
        target = classes[(classes == 'positive') | (classes == 'negative')]
        print("\n-----------------------------------")
        print("----- Positivos VS Negativos -----")
        print("------------------------------------\n")
    elif tipo == '+o-VSvoid':
        # Generamos el conjunto para distinguir entre tweets con sentimientos frente al resto
        data = data[(classes == 'positive') | (classes == 'negative') |  (classes == 'neutral')]
        target = classes[(classes == 'positive') | (classes == 'negative')| (classes == 'neutral')]
        print("\n------------------------------------")
        print("--- Con Sentimiento VS Resto ----")
        print("-------------------------------------\n")
    elif tipo == '+VS-oVoid':
        data = data[(classes == 'positive') | (classes == 'negative')| (classes == 'neutral')]
        target = classes[(classes == 'positive') | (classes == 'negative')| (classes == 'neutral')]
        print("\n--------------------------------")
        print("----- Positivos VS Resto -----")
        print("---------------------------------\n")
    elif tipo == '-VS+oVoid':
        data = data[(classes == 'negative') | (classes == 'irrelevant') | (classes == 'positive')  | (classes == 'neutral')]
        target = classes[(classes == 'negative') | (classes == 'irrelevant') | (classes == 'positive')| (classes == 'neutral')]
        print("\n--------------------------------")
        print("----- Negativos VS Resto -----")
        print("--------------------------------\n")
    else:
        print("\n----------------------------")
        print("----- Todos con Todos -----")
        print("-----------------------------\n")
    #Si no se pasa tipo, se sobreentiendes todos con todos los conjuntos:
    data_train, data_test, classes_train, classes_test = train_test_split(data, target, test_size=0.50)
    return data_train, data_test, classes_train, classes_test, list(set(classes_train))

def main(tipo):


    #Cargamos datos de los tweets ya entrenados:
    data_train, data_test, classes_train, classes_test, target_names = cargaDatos(tipo)

    gridSearch = creaGridSearchCV()

    #Emtremamos GridSearchCV y predecimos:
    prediction = entrenaYpredice(data_test, data_train, classes_train,gridSearch)
    imprimeEstadiscticas(prediction, classes_test, target_names)


#Ejemplo Tweet:
#tipo == '+VS-'
#tipo == '+o-VSvoid'
#tipo == '-VS+oVoid'
#tipo == '+VS-ovoid'

#main('+VS-')
main('+o-VSvoid')
#main('-VS+oVoid')
#main('+VS-oVoid')