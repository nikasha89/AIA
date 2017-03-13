import letterUnigram as uni
import letterBigram as bi
import wordUnigram as wordUni
from random import randint
#Devuelve la palabra más frecuente consecutiva a una dada:
def count_stadistics_word(word,num):
    ant = ''
    res = ''
    cont = 0
    stadWordBigram = {}
    for i in uni.text:
        wordToNumber = ''
        for j in i:
            wordToNumber += str(wordUni.convertletter2digit(j.lower()))
        #print("Palabra: "+i.lower()+" numero:"+ wordToNumber)
        if ant == '':
            ant = i.lower()
        elif ant == word:
            if(wordToNumber in stadWordBigram):
                cont = stadWordBigram[wordToNumber][0]
            if (wordToNumber != ''):
                # Key:codigo_equi_str ; 1er elemento: frecuencia, 2º elemento: palabra
                cont = cont + 1
                stadWordBigram[wordToNumber] = [cont, i.lower()]
        ant = i.lower()
    max_numb = -1
    list_word_same_size = []
    max_probability = ""
    for w in stadWordBigram:
        if len(w) == len(num):
            list_word_same_size.append(w)
        #print("w: "+str(w[0:1])+ " word: "+str(w))
        #print("num: "+str(num[0:1])+" word: "+str(num))
        if max_numb < stadWordBigram[w][0]:
            max_numb = stadWordBigram[w][0]
            max_probability = stadWordBigram[w][1]
    if(len(list_word_same_size)!=0):
        more_probability_word = ""
        menor_diferencia = None
        for p in list_word_same_size:
            if len (p) == len(num):
                res = stadWordBigram[p][0]
            elif more_probability_word == "":
                more_probability_word = stadWordBigram[p][0]
                menor_diferencia = len(num) - len(p)
                if menor_diferencia<0:
                    more_probability_word =stadWordBigram[p][0]
                    menor_diferencia = menor_diferencia *(-1)
            elif menor_diferencia>(len(num) - len(p)) or menor_diferencia>(len(num) - len(p))*(-1):
                more_probability_word = stadWordBigram[p][0]
                menor_diferencia = len(num) - len(p)
                if menor_diferencia<0:
                    menor_diferencia = menor_diferencia *(-1)
        print(more_probability_word)
        res = more_probability_word
    else:
        res = max_probability
    return res


def intersection(lista_a, lista_b):
    lista_a.sort(cmp=None, key=None, reverse=False)
    lista_b.sort(cmp=None, key=None, reverse=False)
    lista_nueva = []
    for i in lista_a:
        for j in lista_b:
            if i == j:
                if i not in lista_nueva:
                    lista_nueva.append(i)

    return lista_nueva

def bigram_predict_words(numList):
    res = ''
    if numList is not None:
        numListSplit = numList.split(" ")
        i = 0
        ant = ''
        for num in numListSplit:
            if i==0:
                ant = wordUni.unigram_word(num)
                res = ant
            else:
                ant = count_stadistics_word(ant,num)
                res += ' '+str(ant)
            i = i+ 1
    return res

print("Predicción Bigrama Palabra code: 836426 922574 4626684636242 52379 27835 frase: "+bigram_predict_words('922574 282636 4626684636242 52379 27835'))
print("Unigram Word code: 836426; word: "+wordUni.unigram_word('836426') + "; code: 922574; word: "+ wordUni.unigram_word('922574'))
print("Unigram Word code: 2747862728426 (está en corpus); word: "+wordUni.unigram_word('2747862728426'))
print("Unigram Word code: 567456 (no está en corpus); word: "+ wordUni.unigram_word('567456'))
print("Bigram Letter --> Word code: 836426 (está en corpus); word: "+bi.bigram_predict_letter_word('836426'))
print("Bigram Letter --> Word code: 567456 (no está en corpus); word: "+bi.bigram_predict_letter_word('567456'))
print("Unigram Letter: code: 3 letra: "+uni.unigram_predict_letter(3))



#print(bigram_predict_word('26682 52379 92274 26682 3786426 92274'))
