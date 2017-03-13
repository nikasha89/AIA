import letterUnigram as uni

def bigram_predict_letter(letter, num):
    res = ''
    if letter is None:
        res = uni.unigram_predict_letter(num)
    else:
        charList = uni.keyboard[num]
        res = count_stadistics(letter,charList)
    return res

def count_stadistics(letter,listChar):
    ant = letter
    stadLetterBigram = {}
    for i in listChar:
        stadLetterBigram[i] = 0
    for i in uni.text:
        for c in i:
            if ant == letter and c in listChar:
                stadLetterBigram[c.lower()] += 1
            ant = c
    max_numb = -1
    for letr in listChar:
        if max_numb < stadLetterBigram[letr]:
            max_numb = stadLetterBigram[letr]
            res = letr
    return res
def bigram_predict_letter_word(number_word):
    word = ''
    let = ''
    i = 0
    for num in number_word:
        if i==0:
            let = None
        let = bigram_predict_letter(let, int(num))
        word += let
        i = i +1
    return word
#print(count_stadistics('a',{'j','k','l'}))
#print("Bigram Letter --> Word code: 567456 (no está en corpus); word: "+bigram_predict_letter_word('5674562375667886'))