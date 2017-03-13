import numpy as np
import random

#from __builtin__ import list

class mom:


    """
        @param a: Matriz de probab
        @param b: Probabilidad de observación x error para cada estado
        @param pi: Vector de posibles estados iniciales
        @markov Instancia un modelo oculto de markov
    """
    def __init__(self,pi, a, b, o, sensor, posibilities, states):
        if a or b or pi or sensor:
            self.pi = pi
            self.a = a
            self.b = b
            self.o = o #observaciones posibles
            self.sensor = sensor
        else:
            self.pi = []
            self.a = []
            self.b = []
            self.o = o
            self.sensor = []
            self.posibility = posibilities
            self.states = states #posibles estados
            if len(states) != 0:
                self.samples(len(states))
            else:
                self.__create_markov_instance()


    def create_b_matrix(self, states,epsilon):
        total = []
        for i in range(len(self.o)):
            total.append(1)
            for j in range(len(self.posibility)):
                error = 1
                if (self.posibility[j] in states and self.posibility[j] in self.o[i]) or \
                        (self.posibility[j] not in  self.o[i] and self.posibility[j] not in states):
                    total[i] *= (1 - epsilon)
                else:
                    total[i] *= epsilon
            self.b.append(total)

    def create_transition_matrix(self, states, condition):
        self.a = []
        for i in range(len(self.states)):
            a_ = []
            for j in range(len(self.states)):
                if condition[i][j]== True:
                    cont = 0
                    if self.m != 1:
                        for k in range(len(states)):
                            if not None in list(states[k]):
                                cont += 1
                    else:
                        cont +=1
                    if cont != 0:
                        a_.append(1/cont)
                    else:
                        a_.append(0)
                else:
                    a_.append(0)
            self.a.append(a_)

    def create_pi(self, states):
        for i in range(len(states)):
            self.pi.append(1/len(states))

    def __create_markov_instance(self):
        conditions = []
        epsilon = 1 / 16
        conditions_ = []
        for i in range(len(self.states)):
            conditions_.append(True)
        conditions.append(conditions_)
        #self.states = current_states(way)
        self.create_pi(self.states)
        self.create_transition_matrix(self.posibility,conditions)
        self.create_b_matrix(self.states, epsilon)


    """def current_states(self,way):
        # Para todos los elementos del tablero
        states = []
        for k1 in range(self.n):
            for k2 in range(self.m):
                total = []
                for i in range(len(self.o)):
                    states = []
                    #way([k1, k2])
                    for j in range(self.posibility):
                        if way[j] != [None, None]:
                            if not None in way[j]:
                                states.append(self.posibility[j])
        return states

    vector = []
    def look_way(self, current_grid):
        list_ways = []
        # check the contiguos grid
        x1 = current_grid[0] - 1
        y1 = current_grid[1]

        x2 = current_grid[0]
        y2 = current_grid[1] - 1

        x3 = current_grid[0] + 1
        y3 = current_grid[1]

        x4 = current_grid[0]
        y4 = current_grid[1] + 1
        #O = ["N", "S", "E", "O"]
        #N,S,E,O:
        #states = []
        list_ways_length = len(list_ways)
        if (self.check_in_bounds(x2, y2) and self.random_map[x2][y2] == 0):
            list_line = [x2, y2]
            list_ways.append(list_line)
            #states.append("N")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x4, y4) and self.random_map[x4][y4] == 0):
            list_line = [x4, y4]
            list_ways.append(list_line)
            #states.append("S")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x1, y1) and self.random_map[x1][y1] == 0):
            list_line = [x1, y1]
            list_ways.append(list_line)
            #states.append("E")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x3, y3) and self.random_map[x3][y3] == 0):
            list_line = [x3, y3]
            list_ways.append(list_line)
            #states.append("O")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        return list_ways


    def check_in_bounds(self,i,j):
        res = False
        if(i in range(self.n) and j in range(self.m)):
            res = True
        return res"""
    def forward(self, observation_prima):
        alpha = []
        indice = self.o.index(observation_prima[0])
        print("Observacion: /n", observation_prima.observacion[indice])
        rangoEstados = range(len(self.states))  # Rango con todos los posibles estados
        # Generacion de la probabilidad del primer instante(paso 1)
        for i in rangoEstados:
            alpha.append(self.sensor[i][indice] * self.pi[i])
        # print(alpha)
        # Generacion de la probabilidad de los siguientes instantes
        rangoPosObs = range(1, len(observation_prima))  # Rango de las posibles observaciones tomadas
        for obs in rangoPosObs:
            indice = self.o.index(observation_prima[obs])
            print("Observacion", self.o[indice])
            antAlpha = alpha[:]
            # print("Observacion: ", obs)
            for i in rangoEstados:
                valor = 0.0
                # print("Estado: ", i)
                for estado in rangoEstados:
                    valor += self.a[i][estado] * antAlpha[estado]
                    # print(modOculMarkov.transiciones[i][estado],'*' , antAlpha[estado])
                alpha[i] = self.sensor[i][indice] * valor
                # print("Valor nuevo alpha: ", alpha)
        acum = 0
        val = 0.0
        for i in alpha:
            if acum < i:
                acum = i
                val = self.a[alpha.index(i)]

        return val
#dada una secuencias de observaciones dar una estimación de la casilla donde está:
    def fordward_ant(self,alfa,t):
        sum = 0.0
        for k in range(1, t):
            for j in range(0, len(self.a[0])):
                for i in range(0, len(self.pi)):
                    if k == 1:
                        alfa[i][j] = self.b[i][j] * self.pi[i]
                    else:
                        if j != t - 1:
                            sum += self.a[i][j] * alfa[i][k-1]
                        else:
                            self.fordward(alfa, t - 1)
                            sum += self.a[i][j] * alfa[i][k - 1] * self.b[j][k]
        return alfa




#La secuencia de estados más probable para llegar de un punto a otro
    def viterbi(self,v,pr,t):
        list_elements = []
        for k in range(1, t):
            for j in range(0, len(self.a[0])):
                for i in range(0, len(self.pi)):
                    if k == 0:
                        v[i][j] = self.b[i][0] * self.pi[i]
                    else:
                        if j != t - 1:
                            list_elements.append(self.a[i][j] * v[i][k-1])
                        else:
                            self.viterbi(v, pr,t - 1)
                            list_elements.append(self.a[i][j] * v[i][k - 1])
                            v[i][j] = max(list_elements) * self.b[j][k]
                            pr.append(np.argmax(list_elements) * self.b[j][k])

        return pr


    def random_generator(self):
        return random.random()

    def __str__(self):
        return "a= " + str(self.a) + "\nb= " + str(self.b) + "\npi= " + str(self.pi) + "\no= " + str(self.o) + "\n"

    def samples(self, t):

        self.pi = []
        self.a = []
        self.b = []
        sum_a = 0
        sum_b = 0
        sum_pi = 0
        ponderacion = 1 / t
        for i in range(t):
            value_pi = self.random_generator() * ponderacion
            if i == t-1:
                value_pi = 1.0 - sum_pi
            sum_pi += value_pi
            self.pi.append(value_pi)
            a_lin = []
            b_lin = []
            sum_b = 0.0
            sum_a = 0.0
            for j in range(t):
                value_a = self.random_generator() * ponderacion
                value_b =self.random_generator() * ponderacion
                if j == t - 1:
                    value_a = 1.0 - sum_a
                    value_b = 1.0 - sum_b

                sum_a += value_a
                sum_b += value_b
                a_lin.append(value_a)
                b_lin.append(value_b)
            self.a.append(a_lin)
            self.b.append(b_lin)


class localization_item(mom):



    def __init__(self,pi,a,b,o, posibilities, states,n,m, walls):
        super().__init__(pi, a, b, o, posibilities, states)
        if walls>=n*m:
            print("El número de walls no puede ser mayor que n*m")
        else:
            tam = (n * m) - walls
            prob = 1/tam
            for i in range(tam):
                self.pi.append(prob)
            self.random_map = self.random_map(n,m, walls)
            self.n = n
            self.m = m
            self.iteration = 0
            #self.states = ["N","S","E","O","NS","NE","NO","SE","SO","EO","NSE","NSO","NEO","SEO","NSEO","-"]
            self.started_grid = self.way_grid(n,m)
            self.current_grid = self.started_grid
            self.list_ways_grid = self.look_way(self.current_grid)
            self.o = o
            self.__create_markov_instance()
            #self.samples2()

    def matrizB(self, epsilon=1 / 50):
        lista = []
        horizontal = len(self.tablero) - 1
        vertical = len(self.tablero) - 1
        # Se generan los vecinos de cada estado
        for i in range(len(self.tablero)):
            for j in range(len(self.tablero[i])):
                a = []
                if (self.tablero[i][j] != 0):
                    if ((i - 1 >= 0) and (j <= vertical) and (i <= horizontal) and self.tablero[i - 1][j] == 1):
                        a.append([i - 1, j])  # N
                    if ((i >= 0) and (j <= vertical) and (i < horizontal) and self.tablero[i + 1][j] == 1):
                        a.append([i + 1, j])  # S
                    if ((j > 0) and (j <= vertical) and (i <= horizontal) and self.tablero[i][j - 1] == 1):
                        a.append([i, j - 1])  # O
                    if ((j >= 0) and (j < vertical) and (i <= horizontal) and self.tablero[i][j + 1] == 1):
                        a.append([i, j + 1])  # E
                    lista.append(a)
        # Se genera una lista del valores de las observaciones para cada estado
        listaDeObservaciones = []
        for i in range(len(self.estados)):
            estadoIn = []
            for j in range(len(self.observaciones)):
                observacion = self.observaciones[j]
                error = 1
                estado = []
                # Se recorren los distintos vecinos y se comprueba si la observacion es equivocada o no
                for direccion in 'NSOE':
                    if direccion == 'N':
                        estado = [self.estados[i][0] - 1, self.estados[i][1]]
                    elif direccion == 'S':
                        estado = [self.estados[i][0] + 1, self.estados[i][1]]
                    elif direccion == 'O':
                        estado = [self.estados[i][0], self.estados[i][1] - 1]
                    elif direccion == 'E':
                        estado = [self.estados[i][0], self.estados[i][1] + 1]

                    if estado in lista[i]:
                        if direccion in observacion:
                            error = error * epsilon
                        else:
                            error = error * (1 - epsilon)
                    else:
                        if direccion in observacion:
                            error = error * (1 - epsilon)
                        else:
                            error = error * epsilon
                estadoIn.append(error)
            listaDeObservaciones.append(estadoIn)
        return listaDeObservaciones


    def __create_markov_instance(self):
        conditions = []
        epsilon = 1 / 16
        for i in range(self.n):
            conditions_ = []
            for j in range(self.m):
                if self.random_map[i][j] == 0:
                    conditions_.append(True)
                else:
                    conditions_.append(False)
            conditions.append(conditions_)
        states = self.look_way(self.current_grid)
        print(states)
        self.pi = []
        self.create_pi(states)
        self.a = []
        self.create_transition_matrix(states, conditions)
        self.b = []
        self.create_b_matrix(states, epsilon)

    def create_observation_and_b_matrix(self):
        # Para todos los elementos del tablero
        posibility = ["N","S","E","O"]
        epsilon = 1 / 16
        for k1 in range(self.n):
            for k2 in range(self.m):
                total = []
                for i in range(len(self.o)):
                    states = []

                    aux = self.look_way([k1, k2])
                    if aux[0] != [None, None]:
                        states.append("N")
                    if aux[1] != [None, None]:
                        states.append("S")
                    if aux[2] != [None, None]:
                        states.append("E")
                    if aux[3] != [None, None]:
                        states.append("O")

                    total.append(1)
                    for j in range(len(posibility)):
                        if (posibility[j] in  states and posibility[j] in self.o[i]) or (posibility[j] not in  states and posibility[j] not in self.o[i]):
                            total[i] *= (1 - epsilon)
                        else:
                            total[i] *= epsilon
                self.b.append(total)


        """for i in self.list_ways_grid:
            o_ = []
            for j in i:
                if j is not None:
                    o_.append(1 / len(self.list_ways_grid))
                else:
                    o_.append(0)
            o.append(o_)"""
    def move_dron(self):
        self.iteration += 1
        item = random.randint(0,len(self.list_ways_grid)-1)
        self.current_grid = self.list_ways_grid[item]

    #Random grid
    def way_grid(self, n, m):
        n_aux = random.randint(0, n - 1)
        m_aux = random.randint(0, m - 1)
        while (self.random_map[n_aux][m_aux] == 1):
            n_aux = random.randint(0, n - 1)
            m_aux = random.randint(0, m - 1)
        return [n_aux,m_aux]

    def random_map(self,n,m, walls):
        #0: Camino
        grid = 0
        self.random_map = []
        if n*m<=walls:
            print("Tienes que introducir un número de walls menor que n*m")
            return self.random_map
        for i in range(n):
            map_line = []
            for j in range(m):
                map_line.append(0)
            self.random_map.append(map_line)
        for i in range(walls):
            way_grid = self.way_grid(n,m)
            self.random_map[way_grid[0]][way_grid[1]] = 1
        return self.random_map

    def __str__(self):
        return super().__str__() + "random map " + str(self.random_map) + " \nstarted grid " + str(self.started_grid) + "\n"
    #current_grid = [i,j]; return matrix of possibles grids to go



    def look_way(self, current_grid):
        list_ways = []
        # check the contiguos grid
        x1 = current_grid[0] - 1
        y1 = current_grid[1]

        x2 = current_grid[0]
        y2 = current_grid[1] - 1

        x3 = current_grid[0] + 1
        y3 = current_grid[1]

        x4 = current_grid[0]
        y4 = current_grid[1] + 1
        #O = ["N", "S", "E", "O"]
        #N,S,E,O:
        #states = []
        list_ways_length = len(list_ways)
        if (self.check_in_bounds(x2, y2) and self.random_map[x2][y2] == 0):
            list_line = [x2, y2]
            list_ways.append(list_line)
            #states.append("N")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x4, y4) and self.random_map[x4][y4] == 0):
            list_line = [x4, y4]
            list_ways.append(list_line)
            #states.append("S")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x1, y1) and self.random_map[x1][y1] == 0):
            list_line = [x1, y1]
            list_ways.append(list_line)
            #states.append("E")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        if (self.check_in_bounds(x3, y3) and self.random_map[x3][y3] == 0):
            list_line = [x3, y3]
            list_ways.append(list_line)
            #states.append("O")
        else:
            list_line = [None, None]
            list_ways.append(list_line)
        return list_ways


    def check_in_bounds(self,i,j):
        res = False
        if(i in range(self.n) and j in range(self.m)):
            res = True
        return res

A = []
A_ = [[0.7,0.3],[0.3,0.7]]
A.append(A_)
A_ = [[0.5,0.5],[0.5,0.5]]
A.append(A_)

# Lista: [Lluvia[paraguas, no-paraguas], no-lluvia[paraguas, no-paraguas]]
B = []
B_ = [[0.9,0.1],[0.2,0.8]]
B.append(B_)
B_ = [[0.8,0.2],[0.1,0.9]]
B.append(B_)
pi = [0.5,0.5]
observation_posibilities = ["AA","AB","AC","AD","BB","BC","BD","CC","CD","DD"]
ob = mom(pi,A,B,observation_posibilities,[],[],[])
alfa = []
v = []
pr = []
for i in range(len(ob.a[0])):
    v_lin = []
    al_lin = []
    for j in range(len(ob.a[0])):
        al_lin.append(float(0.0))
        v_lin.append(float(0.0))
    alfa.append(al_lin)
    v.append(v_lin)

#ob = mom(A,B, pi,["LL","LN","NN"],["L","N"],["L","N","L","L"],0)

print("Obj0\n{}".format(ob))
#print("Obj0 fordward {}".format(ob.fordward(alfa,len(ob.a[0]))))
#print("Obj0 viterbi {}".format(ob.viterbi(v,pr,len(ob.a[0]))))
#ob_item = localization_item([],[],[],0,2,5,6)



ob1 = mom([],[], [],observation_posibilities,[],["A","B","C","C"],["state1","state2","state3"])
alfa = []
v = []
pr = []
for l in range(len(ob1.states)):
    v_lin = []
    al_lin = []
    for m in range(len(ob1.states)):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)

observation_prima = observation_posibilities[3]
print("Obj1\n{}".format(ob1))
print("Obj1 fordward {}".format(ob1.fordward()))
#print("Obj1 viterbi {}".format(ob1.viterbi(v,pr,3)))



"""
PL = A
PN = B
XL = C
XN = D
"""
ob2 = mom([],[], [], [],observation_posibilities,["A","B","C","D"],["A","B","C","C"])

alfa = []
v = []
pr = []
for i in range(len(ob1.states)):
    v_lin = []
    al_lin = []
    for j in range(len(ob1.states)):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)
print("Obj2\n{}".format(ob2))
#print("Obj2 fordward {}".format(ob2.fordward(alfa,3)))
#print("Obj2 viterbi {}".format(ob2.viterbi(v,pr,3)))


alfa = []
v = []
pr = []
for i in range(len(ob1.states)):
    v_lin = []
    al_lin = []
    for j in range(len(ob1.states)):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)

#ob_item = localization_item([],[], [],["N", "S", "E", "O", "NS", "NE", "NO", "SE", "SO", "EO", "NSE", "NSO", "NEO", "SEO", "NSEO", ""],["N","S","E","O"],[],3,5,6)
#print("Obj3\n{}".format(ob_item))
#print("Obj3 fordward {}".format(ob_item.fordward(alfa,len(ob_item.states))))
#print("Obj3 viterbi {}".format(ob_item.viterbi(v,pr,len(ob_item.states))))





#print(ob_item.viterbi(v,pr,t))

#print(ob_item.viterbi (v,pr,t))
#print("Casilla inicial: "+ str(ob_item.started_grid))
#print(ob_item.look_way(ob_item.started_grid))
#print(ob_item.samples2())
#print(ob_item.pi)
#print(ob_item.a)
#print(ob_item.b)
#print(ob_item.a)
#ob.samples(t)
#print(random_map(6,9,10))
#print(ob)
#print(ob.fordward(alfa,t))
#print("Sum alfa {} ".format(sum([sum(i) for i in alfa])))
#print(ob.viterbi(v,pr,t))
#print("Sum v {} ".format(sum([sum(i) for i in v])))

#ob = mom(pi,a,b)
#print(str(ob.randomNumber()))
#print("alpha: "+str(ob.fordward(alfa,t)))
#print("pr: "+str(ob.viterbi(v,pr,t)) + " v: "+str(v))
#print("a:"+str(a))
#print("b: "+str(b))
#print("pi: "+str(b))
"""def init_probability(self, current_grid):
if len(self.look_way(current_grid)) == 0:
    return 0
return (1/(len(self.look_way(current_grid))))


def samples2(self):
    cont = 0
    pi = []
    b = []
    for i in self.random_map:
        for j in i:
            if not j:
                cont += 1
    for i in range(len(self.random_map)):
        b_ = []
        for j in range(len(self.random_map[i])):
            if not self.random_map[i][j]:
                pi.append(1/cont)
            else:
                pi.append(0)
            b_.append(self.init_probability([i,j]))
        b.append(b_)
    self.b = b

def samples3(self):
    cont = 0
    b = []
    listWayGrid = self.look_way(self.current_grid)
    for i in range(2):
        b_ = []
        for j in range(2):
            #[[N,S],[E,O]]
            if listWayGrid[i][j] is not None:
                b_.append(1 / cont)
            else:
                b_.append(0)
        b.append(b_)
    self.b = b
"""
