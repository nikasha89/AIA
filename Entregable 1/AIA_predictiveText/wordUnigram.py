import letterBigram as bi
import letterUnigram as uni

stadistics_word = {}
def convertletter2digit(letter):
    digit = ''
    for key, value in uni.keyboard.items():
        for let in value:
            if letter == let:
                digit = key
    #print(str(value)+" "+str(key))
    return digit
for i in uni.keyboard.values():

    wordToNumber = ''
    for j in i:
        wordToNumber += str(convertletter2digit(j.lower()))
    if (wordToNumber != ''):
        # stadistics_word[codigo_equi_str] = [1er elemento: frecuencia, 2º elemento: palabra]
        stadistics_word[wordToNumber] = [0,'']
def count_words():
    for i in uni.text:
        wordToNumber = ''
        for j in i:
            wordToNumber += str(convertletter2digit(j))
        #print (wordToNumber)
        if(wordToNumber!=''):
            if(not wordToNumber in stadistics_word):
                stadistics_word[wordToNumber] = [0, '']
            cont = stadistics_word[wordToNumber][0]
            stadistics_word[wordToNumber] = [cont + 1, i]

def unigram_word(number_word):
    count_words()
    res = ""
    if number_word in stadistics_word:
        res = stadistics_word[number_word][1]
    else:
        res = bi.bigram_predict_letter_word(number_word)
    return res
count_words()
#print("Unigram Word code: 2747862728426 (está en corpus); word: "+unigram_word('2747862728426'))
#print("Unigram Word code: 567456 (no está en corpus); word: "+unigram_word('567456'))


