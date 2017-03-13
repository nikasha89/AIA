# ==========================================
# Ejercicios de Python
# Aplicaciones de la Inteligencia Artificial
# Máster de Ingeniería Informática
# ==========================================
import math
import random
# ------------
# EJERCICIO 1
# ------------

# Podemos representar una matriz bidimensional nxm en Python, como una lista
# que tiene n elementos que a su vez son listas de de m elementos
# numéricos. Por ejemplo, la siguiente lista de listas representa una matriz
# 4x7:  

# [[3,2,4,2,6,1,6],
#  [2,1,6,9,3,7,8],
#  [1,5,2,2,0,2,7],
#  [1,0,1,2,9,1,4]]

# Definir una función escalar_mat(x,m),que recibiendo un número x y una matriz
# m (representada como se ha indicado), devuelve la matriz que resulta de
# multiplicar cada elemento de la matriz por x. 

# Por ejemplo:

# >>> m=[[3,2,4,2,6,1,6],[2,1,6,9,3,7,8],[1,5,2,2,0,2,7],[1,0,1,2,9,1,4]]
# >>> escalar_mat(3,m)
# [[9, 6, 12, 6, 18, 3, 18], [6, 3, 18, 27, 9, 21, 24], [3, 15, 6, 6, 0, 6, 21], [3, 0, 3, 6, 27, 3, 12]]
# >>> m
# [[3, 2, 4, 2, 6, 1, 6], [2, 1, 6, 9, 3, 7, 8], [1, 5, 2, 2, 0, 2, 7], [1, 0, 1, 2, 9, 1, 4]]

# Definir también una versión escalar_mat_destr(x,m) que devuelve lo mismo,
# pero además modifica m para que contenga lo calculado.  

# Por ejemplo:

# >>> m
# [[3, 2, 4, 2, 6, 1, 6], [2, 1, 6, 9, 3, 7, 8], [1, 5, 2, 2, 0, 2, 7], [1, 0, 1, 2, 9, 1, 4]]
# >>> escalar_mat_destr(3,m)
# [[9, 6, 12, 6, 18, 3, 18], [6, 3, 18, 27, 9, 21, 24], [3, 15, 6, 6, 0, 6, 21], [3, 0, 3, 6, 27, 3, 12]]
# >>> m
# [[9, 6, 12, 6, 18, 3, 18], [6, 3, 18, 27, 9, 21, 24], [3, 15, 6, 6, 0, 6, 21], [3, 0, 3, 6, 27, 3, 12]]
from _operator import concat


def escalar_mat(x,m):
    return [i for row in m for i in row]
def escalar_mat_alt(x, m):
    return [i * x for row in m for i in row]
x = 3
m = [
    [3, 2, 4, 2, 6, 1, 6],
    [2 ,1 ,6 ,9 ,3 ,7 ,8],
    [1, 5, 2, 2, 0, 2, 7],
    [1, 0, 1, 2, 9, 1, 4]
    ]
print("Ejercicio 1 ")
print("Matriz sin modificar" + str(m))
print("Escalar_mat " + str(escalar_mat(x, m)))
print("Escalar_destr" + str(escalar_mat_alt(x, m)))
print()

# ------------
# EJERCICIO 2
# ------------

# Definir una función medias_matriz(m),que recibiendo una matriz m
# (representada como se ha indicado) cuyos elementos son números, devuelve una
# tupla con tres valores: 

# * La lista de las medias de las columnas de la matriz 
# * La lista de las medias columnas de la matriz
# * La media de todos los números de la matriz

# Por ejemplo:

# >>> medias_matriz([[1,2,5,2],[3,1,7,2],[2,1,6,1]])
# ([2.0, 1.3333333333333333, 6.0, 1.6666666666666667], [2.5, 3.25, 2.5], 2.75)

def medias_matriz(m):
    n_row = len(m)
    n_col = len(m[0])
    return [sum(col[i] for col in m) / n_row for i in range(n_col)],[sum(i) / n_col for i in m], sum(i for row in m for i in row) / (n_row * n_col)
m = [
    [1,2,5,2],
    [3,1,7,2],
    [2,1,6,1]
    ]
print("Ejercicio 2 ")
print("Matriz sin modificar" + str(m))
print("Medias matriz " + str(medias_matriz(m)))
print()

# ------------
# EJERCICIO 3
# ------------

# Definir una función producto_matrices(a,b), tal que recibiendo dos matrices
# a y b (representadas como listas de listas, tal y como se explica en el
# ejercicio anterior), devuelve la matriz resultante de multiplicar a y b
# matricialmente. Supondremos que el número de columnas de a es el mismo que
# el número de filas de b. 

# Por ejemplo:

# >>> a=[[3,1,4,5],[2,0,3,5],[1,1,4,1]]
# >>> b=[[1,2],[2,8],[4,3],[3,1]]
# >>> producto_matrices(a,b)
# [[36, 31], [29, 18], [22, 23]]

def producto_matrices(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]for i in range(len(a))]
a = [
    [3, 1, 4, 5],
    [2, 0, 3, 5],
    [1, 1, 4, 1]
    ]
b=[
    [1,2],
    [2,8],
    [4,3],
    [3,1]
    ]
print("Ejercicio 3 ")
print("Matriz a sin modificar" + str(a))
print("Matriz b sin modificar" + str(b))
print("Producto matrices " + str(producto_matrices(a,b)))
print()

# ------------
# EJERCICIO 4
# ------------
#

# Definir una función factoriza_primos(n), que recibiendo como entrada un número 
# natural n, cuya factorizaciçón en números primos es n=p1^e1*p2^e2*...*p_m^em,
# devueve la lista [(p1,e1),(p2,e2),...,(p_m,em)] 

# Ejemplos:

# >>> factoriza_primos(171)
# >>> [(3, 2), (19, 1)]
# >>> factoriza_primos(272250)
# [(2, 1), (3, 2), (5, 3), (11, 2)]
# >>> factoriza_primos(358695540883235472)
# [(2, 4), (3, 1), (7, 1), (83, 2), (173, 5)]

# NOTA: Hacerlo sin usar una lista predefinida de números primos.
# SUGERENCIA: se puede hacer mediante dos bucles "while" anidados. El más interno 
# calcula el exponente de cada posible divisor del número, dividiendo con ese
# divisor mientras sea divisible. El bucle externo contendría al bucle interno
# y además iría incrementando en uno el valor de ese posible divisor.    
# 

def factoriza_primos(n):
    i = 2
    l = []
    j = 1
    while(n > 1):
        while(i <= n):
            if (n % i ==0):
               n = n / i
               if(n % i != 0):
                   l.append([i,j])
               j += 1
            else:
                j = 1
                i += 1
    return l        
n = 171
print("Ejercicio 4 ")
print("Numero natural " + str(n))
print("Factoriza primos " + str(factoriza_primos(n)))
print()

# ------------
# EJERCICIO 5
# ------------
#
# 
# En este ejercicio vamos a "comprimir" y "descomprimir" listas.

#  Apartado (a).
#  Definir la función compresion(l) que devuelva la lista resultante de
#  comprimir la lista l que recibe como entrada, en el siguiente sentido: 
#  * Si el elemento x aparece n (n > 1) veces de manera consecutiva en l
#    sustituimos esas n ocurrencias por la tupla (n, x)
#  * Si el elemento x es distinto de sus vecinos, entonces lo dejamos
#    igual
#  Ejemplo:
 
#  >>> compresión([1, 1, 1, 2, 1, 3, 2, 4, 4, 6, 8, 8, 8])
#  [[3, 2], 1, 3, 2, 4, [2, 6], 8, [3, 8]]
#  >>> compresión(["a", "a", "a", "b", "a", "c", "b", "d", "d", "f", "h", "h", "h"])
#  [[3, 'b'], 'a', 'c', 'b', 'd', [2, 'f'], 'h', [3, 'h']]

#  Apartado (b).
#  Definir la función descompresion(l) que devuelva la lista l descomprimida,
#  suponiendo que ha sido comprimida con el método del apartado anterior.
#  Ejemplo:

#  >>> descompresión([[3, 1], 2, 1, 3, 2, [2, 4], 6, [3, 8]])
#  [1, 1, 1, 2, 1, 3, 2, 4, 4, 6, 8, 8, 8]
# ----------------------------------------------------------------------------

#def compresion2(m):
    #cont=1
    #l = [[i+1 ,m[i]] for i in range(len(m)-1) if m[i]!= m[i+1] ]
    #l.append([len(m),m[len(m)-1]])
    #return l
def compresión(m):
    l = []
    cont = 1
    for i in range(len(m) - 1):
        if m[i] != m[i + 1]:
            l.append([cont, m[i]])
            cont = 1
        else:
            cont += 1
    l.append([cont+1, m[i]])
    return l
def descompresión(m):
    l = []
    for i in m:
        try:
            l.extend([i[1]] * i[0])
        except TypeError:    
            l.append(i);
    return l
m=[1, 1, 1, 2, 1, 3, 2, 4, 4, 6, 8, 8, 8]
print("Ejercicio 5 ")
print("Matriz m sin modificar " + str(m))
print("Compresión " + str(compresión(m)))
print("Descompresión", descompresión(compresión(m)))
print()

# -----------
# EJERCICIO 6
# -----------

# Usando como técnica principal la definición de secuencias por comprensión,
# definir las siguientes funciones:

# a) Dada una lista de números naturales, la suma de los cuadrados de los
#    números pares de la lista.
# Ejemplo:
# >>> suma_cuadrados([9,4,2,6,8,1])
# 120

# b) Dada una lista de números l=[a(1),...,a(n)], calcular el sumatorio de i=1
#    hasta n de i*a(i).
# Ejemplo:
# >>> suma_fórmula([2,4,6,8,10])
# 110

# c) Dados dos listas numéricas de la misma longitud, representado dos puntos
#    n-dimensionales, calcular la distancia euclídea entre ellos. 
# Ejemplo:
# >>> distancia([3,1,2],[1,2,1])
# 2.449489742783178

# d) Dada un par de listas (de la misma longitud) y una función de dos
#    argumentos, devolver la lista de los resultados de aplicar la función a
#    cada par de elementos que ocupan la misma posición en las listas de
#    entrada.
# Ejemplo:
# >>> map2_mio((lambda x,y: x+y) ,[1,2,3,4],[5,2,7,9])
# [6, 4, 10, 13]

# e) Dada una lista de números, contar el número de elementos que sean múltiplos
#    de tres y distintos de cero. 
# Ejemplo:
# >>> m3_no_nulos([4,0,6,7,0,9,18])
# 3

# f) Dadas dos listas de la misma longitud, contar el número de posiciones en
#    las que coinciden los elementos de ambas listas.  
# Ejemplo:
# >>> cuenta_coincidentes([4,2,6,8,9,3],[3,2,1,8,9,6])
# 3

def suma_cuadrados(m):
    return sum(m[i] ** 2 for i in range(len(m)) if m[i] % 2 == 0)
def suma_fórmula(m):
    return sum((i + 1) * m[i] for i in range(len(m)))
def distancia(a,b):
    return math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(len(a))))
def map2_mio(f, a, b):
    return [f(a[i], b[i]) for i in range(len(a))]
def m3_no_nulos(m):
    return sum(i != 0 and i % 3 == 0 for i in m)
def cuenta_coincidentes(a,b):
    return len([i for i in range(len(m)) if a[i] == b[i]])

#Apartado A
m = [9, 4, 2, 6, 8, 1]
#Apartado B
m2 = [2, 4, 6, 8, 10]
#Apartado C
a = [3, 1, 2]
b = [1, 2, 1]
#Apartado D
a2 = [1, 2, 3, 4]
b2 = [5, 2, 7, 9]
f = (lambda x, y: x + y)
#Apartado E
m3 = [4, 0, 6, 7, 0, 9, 18]
#Apartado F
a3 = [4, 2, 6, 8, 9, 3]
b3 = [3, 2, 1, 8, 9, 6]

print("Ejercicio 6 ")
print("Matriz 1 sin modificar" + str(m))
print("Suma cuadrados " + str(suma_cuadrados(m)))
print("Matriz 2 sin modificar" + str(m2))
print("Suma fórmula " + str(suma_fórmula(m2)))
print("Matriz 3 sin modificar" + str(a))
print("Matriz 4 sin modificar" + str(b))
print("Distancia " + str(distancia(a, b)))
print("Matriz 3 sin modificar" + str(a2))
print("Matriz 4 sin modificar" + str(b2))
print("Map 2 mio " + str(map2_mio(f, a2, b2)))
print("Matriz 5 sin modificar" + str(m3))
print("M3 no nulos " + str(m3_no_nulos(m3)))
print("Matriz 6 sin modificar" + str(a3))
print("Matriz 7 sin modificar" + str(b3))
print("Cuenta coincidentes " + str(cuenta_coincidentes(a3, b3)))
print()

# -----------
# EJERCICIO 7
# -----------
#
# 
# Antiguamente, cuando las impresoras eran matriciales y se compartían en un
# centro de trabajo, era normal que cada trabajo de impresión llevara una
# portada con dígitos de gran tamaño que indicaba el número del trabajo de
# impresión. Estos dígitos estaban dibujados mediante algún carácter simple. 

# Por ejemplo, lo que sigue es el número 0123456789 dibujado con asteriscos:

#   ***    *   ***   ***     *   *****  ***  *****  ***   ****  
#  *   *  **  *   * *   *   **   *     *         * *   * *   *
# *     *  *  *  *      *  * *   *     *        *  *   * *   *
# *     *  *    *     **  *  *    ***  ****    *    ***   ****
# *     *  *   *        * ******     * *   *  *    *   *     *
#  *   *   *  *     *   *    *   *   * *   * *     *   *     *
#   ***   *** *****  ***     *    ***   ***  *      ***      *

# Definir una función dígitos_grandes(x) que recibiendo un número entero
# positivo, lo escriba por pantalla usando dígitos grandes. Por ejemplo:


# >>> dígitos_grandes(8)
#   *** 
#  *   *
#  *   *
#   *** 
#  *   *
#  *   *
#   *** 
# >>> dígitos_grandes(4)
#     *  
#    **  
#   * *  
#  *  *  
#  ******
#     *  
#     *  

# INDICACIÓN:

# Puede ser de utilidad tener definidas las siguientes listas, que almacenan
# las distintas líneas que sirven para dibujar cada dígito grande:



def digitos_grandes(x):

    cero = ["  ***  ", 
            " *   * ", 
            "*     *", 
            "*     *", 
            "*     *", 
            " *   * ", 
            "  ***  "]

    uno = [" * ", 
           "** ", 
           " * ", 
           " * ", 
           " * ", 
           " * ", 
           "***"]

    dos = [" *** ", 
           "*   *", 
           "*  * ", 
           "  *  ", 
           " *   ", 
           "*    ", 
           "*****"]

    tres = [" *** ", 
            "*   *", 
            "    *", 
            "  ** ", 
            "    *", 
            "*   *", 
            " *** "]

    cuatro = ["   *  ", 
              "  **  ", 
              " * *  ", 
              "*  *  ", 
              "******", 
              "   *  ", 
              "   *  "]

    cinco = ["*****", 
             "*    ", 
             "*    ", 
             " *** ", 
             "    *", 
             "*   *", 
             " *** "]

    seis = [" *** ", 
            "*    ", 
            "*    ", 
            "**** ", 
            "*   *", 
            "*   *", 
            " *** "]

    siete = ["*****", 
             "    *", 
             "   * ", 
             "  *  ", 
             " *   ", 
             "*    ", 
             "*    "]

    ocho = [" *** ", 
            "*   *", 
            "*   *", 
            " *** ", 
            "*   *", 
            "*   *", 
            " *** "]

    nueve = [" ****", 
             "*   *", 
             "*   *", 
             " ****", 
             "    *", 
             "    *", 
             "    *"]
    digitos = [cero, uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve]


    '''switcher = {
        0: [i for i in cero if i != '"'],
        1: [i for i in uno if i != '"'],
        2: [i for i in dos if i != '"'],
        3: [i for i in tres if i != '"'],
        4: [i for i in cuatro if i != '"'],
        5: [i for i in cinco if i != '"'],
        6: [i for i in seis if i != '"'],
        7: [i for i in siete if i != '"'],
        8: [i for i in ocho if i != '"'],
        9: [i for i in nueve if i != '"'],
    }'''

    #cad = switcher.get(int(x), "nothing")
    #return cad
    sn = str(n)
    for fila in range(7):
        linea = []
        for digito in sn:
            linea.append(digitos[int(digito)][fila])
        print(" ".join(linea))
# ----------------------------------------------------------------------------
print("Ejercicio 7 ")
digitos_grandes("0123456789")
'''[print(digitos_grandes(0)[i]
       + digitos_grandes(1)[i]
       + digitos_grandes(2)[i]
       + digitos_grandes(3)[i]
       + digitos_grandes(4)[i]
       + digitos_grandes(5)[i]
       + digitos_grandes(6)[i]
       + digitos_grandes(7)[i]
       + digitos_grandes(8)[i]
       + digitos_grandes(9)[i]) for i in range(len(digitos_grandes("0")))]'''
print()
# -------------------------------------------------------------------------- 

# -----------
# EJERCICIO 8
# -----------
#
# Supongamos que recibimos un diccionario cuyas claves son cadenas de
# caracteres de longitud uno y los valores asociados son números enteros 
# entre 0 y 50. 
# Definir una función histograma_horizontal(d), que recibiendo un diccionario 
# de ese tipo, escribe un histograma de barras horizontales asociado, 
# como se ilustra en el siguiente ejemplo:  

# >>> d1={"a":5,"b":10,"c":12,"d":11,"e":15,"f":20,
#         "g":15,"h":9,"i":7,"j":2}
# >>> histograma_horizontal(d1)
# a: *****
# b: **********
# c: ************
# d: ***********
# e: ***************
# f: ********************
# g: ***************
# h: *********
# i: *******
# j: **
#
# Nota: imprimir las barras, de arriba a abajo, en el orden que determina la
#         función "sorted" sobre las claves 
# ---------------------------------------------------------------------------

def histograma_horizontal(dl):
    return [i + ':' + '*' * dl[i] for i in sorted(dl)]    
d1={"a" : 5, "b" : 10, "c" : 12, "d" : 11, "e" :15, "f": 20, "g" : 15, "h" : 9, "i" : 7, "j" : 2}
print("Ejercicio 8 ")
[print(i) for i in histograma_horizontal(d1)]
print()

# ------------
# EJERCICIO 9
# ------------
# Con la misma entrada que el ejercicio anterior, definir una función
# histograma_vertical(d) que imprime el mismo histograma pero con las barras
# en vertical. 

# Ejemplo:

# >>> d2={"a":5,"b":7,"c":9,"d":12,"e":15,"f":20,
#         "g":15,"h":9,"i":7,"j":2}
# >>> histograma_vertical(d2)
#           *         
#           *         
#           *         
#           *         
#           *         
#         * * *       
#         * * *       
#         * * *       
#       * * * *       
#       * * * *       
#       * * * *       
#     * * * * * *     
#     * * * * * *     
#   * * * * * * * *   
#   * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * *   
# * * * * * * * * * * 
# * * * * * * * * * * 
# a b c d e f g h i j

# Nota: imprimir las barras, de izquierda a derecha, en el orden que determina la
#         función "sorted" sobre las claves 
# ---------------------------------------------------------------------------

def histograma_horizontal(dl):
    ks = sorted(dl.keys())
    maxv = max(dl.values())
    for altura in range(maxv, 0, -1):
        lin = ""
        for k in ks:
            lin += ("* " if dl[k] >= altura else "  ")
        print(lin)
    print(" ".join(ks))
dl2={"a" : 5, "b" : 7, "c" : 9, "d" : 12, "e" : 15, "f" : 20, "g" : 15, "h" : 9, "i" : 7, "j" : 2}
print("Ejercicio 9 ")
histograma_horizontal(dl2)
print()


# ------------
# EJERCICIO 10
# ------------
#
# 
# Supongamos que tenemos almacenada, usando diccionario, la información sobre
# el grupo de los alumnos de un curso. Para ello, usamos como clave el nombre
# de los alumnos de un grupo y como valor el grupo que tienen asignado. 

# 1) Definir una función alumnos_grupo(d) que a partir de un diccionario
# de ese tipo, devuelve otro diccionario cuyas claves son los nombres de los
# grupos y cuyo valor asignado a cada clave es la lista los alumnos que
# forman parte del grupo.

# Ejemplos:

# >>> alum={"Juan":"G1", "Rosa":"G2"  , "Joaquín":"G1"   ,"Carmen":"G2"  , 
#           "Isabel":"G1" , "Rocío":"G3" , "Bernardo":"G3", "Jesús":"G2"}
# >>> grupos=alumnos_grupo(alum)
# >>> grupos
# {'G3': ['Rocío', 'Bernardo'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Juan', 'Joaquín']}

# 2) Definir una función nuevo_alumno(dict_n,dict_g,nombre,grupo) tal que
# supuesto que dict_n y dict_g son dos variables conteniendo respectivamente
# el grupo de cada alumno y los alumnos de cada grupo, introduce un nuevo
# alumno con su nombre y grupo, modificando adecuadamente tanto dict_n como
# dict_g. Si el alumno ya está en los diccionarios, modificar el dato
# existente (en ese caso, si además el grupo que se quiere asignar no coincide
# que el que ya tiene se mostrará un mensaje de advertencia). Si se asigna un
# grupo que no está dado de alta, la correspondiente entrada se debe añadir al
# diccionario de grupos.

# Ejemplos:

# >>> nuevo_alumno(alum,grupos,"Bernardo","G3")
# No actualizado. El alumno Bernardo ya está dado de alta en el grupo G3
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Juan': 'G1', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G3'}
# >>> nuevo_alumno(alum,grupos,"Bernardo","G1")
# El alumno Bernardo ya está dado de alta. Se cambia al grupo G1
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Juan': 'G1', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G1'}
# >>> grupos
# {'G3': ['Rocío'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Juan', 'Joaquín', 'Bernardo']}
# >>> nuevo_alumno(alum,grupos,"Ana","G3")
# Nuevo alumno Ana. Incluido en el grupo G3
# >>> nuevo_alumno(alum,grupos,"Juan","G4")
# El alumno Juan ya está dado de alta. Se cambia al grupo G4
# >>> alum
# {'Isabel': 'G1', 'Jesús': 'G2', 'Rocío': 'G3', 'Ana': 'G3', 'Juan': 'G4', 'Carmen': 'G2', 
#  'Rosa': 'G2', 'Joaquín': 'G1', 'Bernardo': 'G1'}
# >>> grupos
# {'G4': ['Juan'], 'G3': ['Rocío', 'Ana'], 'G2': ['Jesús', 'Carmen', 'Rosa'], 
#  'G1': ['Isabel', 'Joaquín', 'Bernardo']}
# --------------------------------------------------------------------------


def alumnos_grupo(alum):
    return {i : [j for j in alum if  i == alum[j]]  for i in set(alum.values())}

def nuevo_alumno(dict_n, dict_g, nombre, grupo):
    busqueda_alumno = ""
    if nombre in dict_g:
        busqueda_alumno = nombre
    if not nombre in dict_n:
        dict_n[nombre] = ""
    if dict_n[nombre] == grupo:
        if busqueda_alumno == nombre:
            print("No actualizado. El alumno {} ya está dado de alta en el grupo {}".format(nombre,grupo))
    else:
        if busqueda_alumno == nombre:
            dict_n[nombre] = grupo
            print ("El alumno {} ya está dado de alta. Se cambia al grupo {}".format(nombre,grupo))
        else:
            dict_n[nombre] = grupo
            print("Nuevo alumno {}. Incluido en el grupo {}".format(nombre,grupo))
    return dict_n
alum={"Juan" : "G1", "Rosa" : "G2", "Joaquín" : "G1", "Carmen" : "G2", "Isabel" : "G1", "Rocío" : "G3", "Bernardo" : "G3", "Jesús" : "G2"}
grupos=alumnos_grupo(alum)
grupo = "G3"
nombre = "Bernardo"
print("Ejercicio 10 ")
print("Alumnos "+ str(alum))
print("Grupos "+ str(grupos))
print("Alumno " + nombre)
print("grupo " + grupo)
alum = nuevo_alumno(alum, grupos, nombre, grupo)
grupos=alumnos_grupo(alum)
print("Alumnos "+ str(alum))
print("Grupos "+ str(grupos))
grupo = "G1"
print("Alumno " + nombre)
print("grupo " + grupo)
alum = nuevo_alumno(alum, grupos, nombre, grupo)
grupos=alumnos_grupo(alum)
print("Alumnos "+ str(alum))
print("Grupos "+ str(grupos))
grupo = "G3"
nombre = "Ana"
print("Alumno " + nombre)
print("grupo " + grupo)
alum = nuevo_alumno(alum, grupos, nombre, grupo)
grupos=alumnos_grupo(alum)
print("Alumnos "+ str(alum))
print("Grupos "+ str(grupos))
grupo = "G4"
nombre = "Juan"
print("Alumno " + nombre)
print("grupo " + grupo)
alum = nuevo_alumno(alum, grupos, nombre, grupo)
grupos=alumnos_grupo(alum)
print("Alumnos "+ str(alum))
print("Grupos "+ str(grupos))
print()



# ------------
# EJERCICIO 11
# ------------


# Definir una función mi_grep(cadena,fichero) similar al comando grep de unix
# (sin uso de patrones). Es decir, escribe por pantalla las líneas de fichero
# en las que ocurre cadena (junto con el número de línea).

# Por ejemplo, si buscamos la cadena "función" en un fichero similar a éste,
# las prímeras líneas de la salida podrían ser similares a éstas: 


# >>> mi_grep("función","ejercicios-03-ipppd.py")
# Línea 12: # Definir una función codifica_descodifica(fichero1,fichero2) que codifique un
#                         ^^^^^^^
# Línea 32: # Indicación: La función chr(n) obtiene el carácter cuyo código
#                            ^^^^^^^
# Línea 33: # es el número "n". A la inversa, la función ord(c) devuelve el código del
#                                                ^^^^^^^
# Línea 47: # Definir una función mi_grep(cadena,fichero) similar al comando grep de unix
#                         ^^^^^^^
# ....
#  
#
# ---------------------------------------------------------------------------

def mi_grep (cadena, fichero):
    infile = open("./" + fichero, 'r')
    cont = 1
    cad = []
    for i in infile:
        if(i.find(cadena) != -1):
            cad.append("Linea "+ str(cont) + ":" + i)
        cont+=1
    infile.close()
    return cad
cadena = "función"
fichero = "input3.txt"
print("Ejercicio 11 ")
[print(i) for i in mi_grep(cadena,fichero)]
print()

# ------------
# EJERCICIO 12
# ------------

# (J. Zelle) Definir una clase para representar "dados" con un número de caras
# dado.  Los métodos de la clase deben servir para: 
# - Tirar el dado y que el dado tenga un valor aleatorio 
# - Fijar un valor del dado, sin aleatoriedad
# - Obtener el valor que marca en ese momento el dado El valor inical del dado
#   debe ser 1

# >>> md=MDado(10)
# >>> md.obtén_valor()
# 1
# >>> md.tira()
# >>> md.obtén_valor()
# 9
# >>> md.tira()
# >>> md.obtén_valor()
# 5
# >>> md.fija_valor(4)
# >>> md.obtén_valor()
# 4
# ----------------------------------------------------------------------------

class MDado:
    def __init__ (self, x):
        self.x = x
    def tira (self):
        self.x = random.randint(1, self.x)
    def fija_valor (self, x):
        self.x = x
    def obtén_valor(self):
        return self.x
    def __str__ (self):
        return "El valor del dado es: " + str(self.x)

print("Ejercicio 12 ")
md=MDado(10)
md.obtén_valor()
print(md)
md.tira()
md.obtén_valor()
print(md)
md.tira()
md.obtén_valor()
print(md)
md.fija_valor(4)
md.obtén_valor()
print(md)
print()

# ------------
# EJERCICIO 13
# -----------

# Apartado 13.1
# -------------

# Supongamos que queremos gestionar los alumnos de una titulación, con las
# asignaturas en las que están matriculados, y las notas que tienen. Para
# ello, se pide implementar una clase Alumno, con las siguientes
# características: 

# - El constructor de la clase recibe como argumentos el nombre del alumno y
#   una lista de las asignaturas que matricula inicialmente (sin nota). Por
#   simplificar, supondremos que el nombre es un string con un nombre de pila
#   y dos apellidos, y que la asignatura viene dada por sus siglas. Se supone
#   que ni el nombre de pila ni los apellidos son compuestos.

# - El nombre debe ser un atributo de datos de la clase. Además, incluir cualquier 
#   atributo que pudiera ser necesario para mantener la información sobre las 
#   asignaturas en las que está matriculado el alumno, y la nota, si la tuviera 
#   (si aún no tiene nota de una asignatura, asignar el valor "-")

# - Los métodos de la clase son los siguientes:
#    * Método __repr__, que devuelve simplemente el nombre del alumno
#    * Método pon_nota, que recibe una asignatura y una nota, y anota
#      al alumno la nota de esa asignatura. Si el alumno no está matriculado
#      en esa asignatura, el método debe devolver la excepción 
#      AsignaturaNoMatriculada, que se define más abajo.
#    * Método consulta_nota, que recibe una asignatura y devuelve la nota que
#      ese alumno tiene en la asigntura. Si el alumno no está matriculado
#      en esa asignatura, el método debe devolver la excepción 
#      AsignaturaNoMatriculada, que se define así:
#         class AsignaturaNoMatriculada(Exception):
#             pass
#    * Método añade_asignatura que recibe una asignatura, y añade esa
#      asignatura al alumno. Si la asignatura ya la tiene el alumno, no hacer
#      nada.
#    * Método asignaturas_matriculadas, que devuelve la lista de asignaturas
#      matriculadas del alumno
#    * Método media_expediente, que recibiendo el plan de estudios del alumno,
#      devuelve la nota media del alumno (ponderada por número de créditos de
#      cada asignatura). El plan de estudios es un diccionario cuyas claves
#      son todas las asignaturas, y el valor asociado a cada clave es el
#      número de creditos de la asignatura (ver ejemplo más abajo). Si una
#      asignatura no está matriculada o evaluada, se considera que está
#      puntuada con un cero.


# Ejemplos:

# >>> alumno1=Alumno("Antonio Ruiz Santos", ["DGPDS1","DGPDS2","IPPPD","FEST","AEM","APCD"])
class AsignaturaNoMatriculada(Exception):
    pass
class Alumno:
    def __init__ (self, nombre, asignaturas):
        self.nombre = nombre
        self.asignaturas = {i: '-' for i in asignaturas }
    def __repr__ (self):
        return self.nombre
    def pon_nota(self, asignatura, nota):
        try:
            self.asignaturas[asignatura] = nota
        except:
            raise
    def consulta_nota(self, asignatura):
        try:
            return self.asignaturas[asignatura]
        except Exception:
            raise AsignaturaNoMatriculada("Asignatura no matriculada para este alumno")
    def añade_asignatura(self, asignatura):
        try:
            self.asignaturas[asignatura]
        except:
            self.asignaturas[asignatura] = '-'
    def asignaturas_matriculadas(self):
       return self.asignaturas.keys()
    def media_expediente(self, plan_de_estudios_MDS):
        cont = 0
        suma = sum([plan_de_estudios_MDS[i] for i in plan_de_estudios_MDS])
        nota = 0.0
        for i in plan_de_estudios_MDS:
            try:
                if self.consulta_nota(i) != '-':
                    nota += float(self.consulta_nota(i))* (plan_de_estudios_MDS[i] / suma)
                    cont += 1
            except AsignaturaNoMatriculada:
                nota += 0.0
        return nota
print("Ejercicio 13")
nombre = "Antonio Ruiz Santos"
asignaturas = ["DGPDS1","DGPDS2","IPPPD","FEST","AEM","APCD"]
alumno1 = Alumno(nombre, asignaturas)      
# >>> alumno1.nombre
# 'Antonio Ruiz Santos'
print("Alumno " + alumno1.nombre)

# >>> alumno1 # Aquí se llamaría al método __repr__
# Antonio Ruiz Santos
print("Alumno __repr__ " + str(alumno1))

# >>> alumno1.consulta_nota("IPPPD")
# '-'
asignatura = "IPPPD"
try:
    print("Consulta nota " + str(alumno1.consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> alumno1.pon_nota("IPPPD",8.9)
nota = 8.9
alumno1.pon_nota(asignatura, nota)

# >>> alumno1.consulta_nota("IPPPD")
# 8.9
try:
    print("Consulta nota " + str(alumno1.consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> alumno1.consulta_nota("ML1")
# Traceback (most recent call last):

#   File "<ipython-input-41-33cce032017f>", line 1, in <module>
#     alumno1.consulta_nota("ML1")

#   File ".......", line 26, in consulta_nota
#     raise AsignaturaNoMatriculada("Asignatura no matriculada para este alumno")

# AsignaturaNoMatriculada: Asignatura no matriculada para este alumno
asignatura = "ML1"
try:
    print("Consulta nota " + str(alumno1.consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> alumno1.añade_asignatura("ML1")
alumno1.añade_asignatura(asignatura)

# >>> alumno1.consulta_nota("ML1")
# '-'

nota = 6.3

# >>> alumno1.pon_nota("ML1",6.3)
alumno1.pon_nota(asignatura, nota)

# >>> alumno1.consulta_nota("ML1")
# 6.3
try:
    print("Consulta nota " + str(alumno1.consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> alumno1.asignaturas_matriculadas()
# ['APCD', 'DGPDS1', 'ML1', 'IPPPD', 'DGPDS2', 'AEM', 'FEST']
print("Asignaturas " + str (alumno1.asignaturas_matriculadas()))

plan_de_estudios_MDS={"DGPDS1":3,"DGPDS2":6,"IPPPD":4,"FEST":4,"AEM":6,"APCD":4,
                   "APBD":5,"ML1":5,"ML2":5,"TMO":4,"ICSR":3,"MDTE":3,"DSBI":3,
                   "PLNCD1":2,"PLNCD2":2,"VD":2,"VI":2,"TFM":6} 

# >>> alumno1.media_expediente(plan_de_estudios_MDS)
# 0.9724637681159419
print("Nota media " + str(alumno1.media_expediente(plan_de_estudios_MDS)))

        

# Apartado 13.2
# -------------


# Supongamos que tenemos un archivo de texto en los que cada línea corresponde
# a un alumno con sus asignaturas y notas, con el siguiente formato:

# NOMBRE APELLIDO1 APELLIDO2 A1 N1 A2 N2 .... An Nn

# Por ejemplo, podríamos tener un archivo alumno_notas.txt con las siguientes
# líneas:

# Juan Pérez Quirós DGPDS1 7.4 DGPDS2 8.4 IPPPD 9.1 FEST 7.5 AEM 6.2 APCD 8.2 APBD 5.3 ML1 8.8 ML2 7.5 TMO 8.7 ICSR 6.1 MDTE 7.3 DSBI 10.0 PLNCD1 5.0 PLNCD2 6.2 VD 6.4 VI 7.1 TFM 8.5
# María González Peña DGPDS1 5.4 DGPDS2 9.3 IPPPD 8.7 FEST 7.6 APCD 9.2 APBD 6.6 ML1 .8 ML2 7.7 TMO 5.2 MDTE 5.3 DSBI 8.2 PLNCD1 6.0 PLNCD2 9.2 VD 6.4 VI 7.1 
# Pedro Moncada Escobar DGPDS1 6.4 IPPPD 9.5 FEST 7.8 AEM 5.2 APCD 7.2 APBD 5.8 ML1 8.8 TMO 7.2 ICSR 8.8 DSBI 5.0 PLNCD1 7.0 VD 8.4 VI 6.1 
# Salvador Gutiérrez Sánchez DGPDS1 7.7 DGPDS2 8.0 IPPPD 7.3 FEST 7.9 AEM 8.2 APCD 8.6 APBD 5.3 TMO 5.2 ICSR 8.1 MDTE 5.3 PLNCD1 5.3 PLNCD2 7.5 VD 8.4
# Rocío Cotán Sánchez DGPDS2 8.2 FEST 7.1 APCD 6.2 ML1 5.8 ML2 7.9 TMO 5.2 ICSR 9.1 MDTE 6.3 DSBI 6.6 PLNCD1 5.6 PLNCD2 6.5 VI 6.1 TFM 9.5
# Gabriel Mejías Cifuentes DGPDS1 6.9 DGPDS2 7.3 IPPPD 9.0 FEST 6.5 AEM 6.5 APBD 5.7 ML1 7.8 ICSR 8.1 MDTE 5.3 PLNCD1 5.1 PLNCD2 8.0 
# Josefa Cabrera León DGPDS1 7.4 DGPDS2 8.4 IPPPD 9.1 FEST 7.5 

# Por simplificar, ni los nombres de pila ni los apellidos serán compuestos.

# Se pide definir una función lee_notas(archivo), que recibiendo el nombre del
# archivo, devuelva una lista de objetos de la clase Alumno, cada uno
# conteniendo toda la información de la correspondiente línea del archivo de
# texto. 

# Ejemplo:
def lee_notas (fichero):
    infile = open("./" + fichero, 'r')
    line = [i for i in infile]
    infile.close()
    l = [i.replace("\n","").split(' ') for i in line]
    lista_alumnos = []
    for i in range(len(l)):
        nombre = ""
        asignaturas = {}
        for j in range(len(l[i])):     
            if j == 0 or j == 1:
                nombre += l[i][j] + " "
            elif j == 2:
                nombre += l[i][j]
            elif j % 2 != 0:
               asignaturas[l[i][j]] = l[i][j+1]
        alumno = Alumno(nombre, [i for i in asignaturas])
        [alumno.pon_nota(i, asignaturas[i]) for i in asignaturas]                    
        lista_alumnos.append(alumno)
    return lista_alumnos

# >>> lista_alumnos=lee_notas("alumno_notas.txt")
fichero = "alumno_notas.txt"

lista_alumnos = lee_notas(fichero)
# >>> lista_alumnos
# [Juan Pérez Quirós,
#  María González Peña,
#  Pedro Moncada Escobar,
#  Salvador Gutiérrez Sánchez,
#  Rocío Cotán Sánchez,
#  Gabriel Mejías Cifuentes,
#  Josefa Cabrera León]
print("Lista de alumnos " + str(lista_alumnos))

# >>> lista_alumnos[2].nombre
# 'Pedro Moncada Escobar'
print("Alumno " + lista_alumnos[2].nombre)
# >>> lista_alumnos[2].consulta_nota("APCD")
# 7.2
asignatura = "APCD"
try:
    print("Consulta nota " + str(lista_alumnos[2].consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> lista_alumnos[2].consulta_nota("DSBI")
# 5.0
asignatura = "DSBI"
try:
    print("Consulta nota " + str(lista_alumnos[2].consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
# >>> lista_alumnos[2].consulta_nota("TFM")
# Traceback (most recent call last):

#   File "<ipython-input-56-b068fc897dbd>", line 1, in <module>
#     lista_alumnos[2].consulta_nota("TFM")

#   File "......", line 26, in consulta_nota
#     raise AsignaturaNoMatriculada("Asignatura no matriculada para este alumno")

# AsignaturaNoMatriculada: Asignatura no matriculada para este alumno
asignatura = "TFM"
try:
    print("Consulta nota " + str(lista_alumnos[2].consulta_nota(asignatura)))
except AsignaturaNoMatriculada as e:
    print(e)
    
# >>> lista_alumnos[3].asignaturas_matriculadas() 
# ['DGPDS1',
#  'PLNCD2',
#  'PLNCD1',
#  'ICSR',
#  'IPPPD',
#  'DGPDS2',
#  'AEM',
#  'VD',
#  'TMO',
#  'APCD',
#  'APBD',
#  'MDTE',
#  'FEST']
print("Asignaturas " + str(lista_alumnos[3].asignaturas_matriculadas()))

# Apartado 13.3
# --------------

# Definir una función mejor_expediente(lista_alumnos, plan_de_estudiso), que
# recibiendo como entrada:
#  - Una lista lista_de_alumnos de objetos de la clase Alumno
#  - Un diccionario plan_de_estudios, que asigna a cada asignatura del plan de
#    estudios su número de créditos. 
# devuelve el objeto de la clase Alumno (o lista de objetos, si hay más de uno), con la mejor
# nota media

# Ejemplo:

# >>> mejor_expediente(lista_alumnos,plan_de_estudios_MDS)
# Juan Pérez Quirós
def mejor_expediente(lista_alumnos, plan_de_estudios):
    maxi = 0
    media = 0
    alumnos = []
    for i in range(len(lista_alumnos)):
        media = lista_alumnos[i].media_expediente(plan_de_estudios)
        if maxi <= media:
            maxi = media
            alumnos.append(lista_alumnos[i])
    if len(alumnos) == 1:
        return alumnos[0]
    else:
        return alumnos
print("Mejor expediente " + str(mejor_expediente(lista_alumnos, plan_de_estudios_MDS)))




























