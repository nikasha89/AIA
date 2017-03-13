# coding=utf-8
import re
import unicodedata

keyboard = {
    2: ['a', 'b', 'c'],
    3: ['d', 'e', 'f'],
    4: ['g', 'h', 'i'],
    5: ['j', 'k', 'l'],
    6: ['m', 'n', 'ñ', 'o'],
    7: ['p', 'q', 'r', 's'],
    8: ['t', 'u', 'v'],
    9: ['w', 'x', 'y', 'z']
}
def elimina_tildes(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


def read_text():
    fichero = "Nietzshe - Mas alla del bien y del mal.txt"
    infile = open("./" + fichero, encoding="utf8")
    cad = []
    for i in infile:
        cad.append(i + "\n")
    infile.close()
    return cad


def count_letters():
    for i in text:
        for c in i:
            stadistics[c.lower()] += 1

def unigram_predict_letter(num):
    charList = keyboard[num]
    res = 'a'
    max_numb = -1
    for letter in charList:
        if max_numb < stadistics[letter]:
            max_numb = stadistics[letter]
            res = letter
    return res

file = "Nietzshe - Mas alla del bien y del mal.txt"

#text = re.findall(r"[\D\S.]+", elimina_tildes(str(read_text())))
text = re.findall(r"[A-Za-zÁ-Úá-ú]+", elimina_tildes(str(read_text())))
#print(text)
Nletter = sum((len(i) for i in text))

stadistics = {}
for i in keyboard.values():
    for v in i:
        stadistics[v] = 0
count_letters()
#print("Unigram Letter:"+unigram_predict_letter(3))
