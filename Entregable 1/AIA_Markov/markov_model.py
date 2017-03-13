import numpy as np
import random

#from __builtin__ import list

class mom:


    """
        @param a: Matriz de probab    def look_way(self,current_grid):
        list_ways = []
        #check the contiguos grid
        for i in range(2):
            for j in range(2):
                i_aux = current_grid[0]-i
                i_aux2 = current_grid[0]+i
                j_aux = current_grid[1]-j
                j_aux2 = current_grid[1]+j
                if(i !=0 or j !=0):
                    if(self.check_in_bounds(i_aux, j_aux) and self.random_map[i_aux][j_aux]==0):
                        list_line = [i_aux,j_aux]
                        list_ways.append(list_line)
                    if (self.check_in_bounds(i_aux2,j_aux2) and self.random_map[i_aux2][j_aux2] == 0):
                        list_line = [i_aux2, j_aux2]
                        list_ways.append(list_line)
        return list_waysilidad de transicción
        @param b: Probabilidad de observación
        @param pi: Vector de posibles estados de transicción
        @markov Instancia un modelo oculto de markov
    """
    def __init__(self, pi, a, b, o, sensor, posibilities, states):
        if a or b or pi or sensor:
            self.pi = pi
            self.a = a
            self.b = b
            self.o = o  # observaciones posibles
            self.sensor = sensor
            self.states = states
            self.posibility = posibilities
            self.sensorCalculado = self.Samples(len(states))

        else:
            self.pi = []
            self.a = []
            self.b = []
            self.o = o
            self.sensor = []
            self.posibility = posibilities
            self.states = states  # posibles estados
            if len(states) != 0:
                self.Samples(len(states))
            else:
                self.__create_markov_instance()



    # Currentstates
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
                            if None in list(states[k]):
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
        n_states = len(states) -1
        for i in range(n_states):
            self.pi.append(1/n_states)

    def __create_markov_instance(self):
        conditions = []
        epsilon = 1 / 16
        for i in range(len(self.states)):
            conditions_ = [True]
            conditions.append(conditions_)
        #self.states = current_states(way)
        #self.create_pi(self.states)
        #self.create_transition_matrix(self.posibility,conditions)
        #self.create_b_matrix(self.states, epsilon)
        self.sensor = self.b


    def fordward(self, observation_prima):
        alpha = []
        index = self.o.index(observation_prima[0])
        print("Observacion:" + str(self.o[index]))
        rangoEstados = range(len(self.states))  # Rango con todos los posibles estados
        # Generacion de la probabilidad del primer instante(paso 1)
        for i in rangoEstados:
            print(index)
            print(self.sensor[i][index])
            alpha.append(self.sensor[i][index] * self.pi[i])
        # print(alpha)
        # Generacion de la probabilidad de los siguientes instantes
        rangoPosObs = range(1, len(observation_prima))  # Rango de las posibles observaciones tomadas
        for obs in rangoPosObs:
            indice = self.o.index(observation_prima[obs])
            print("Observacion", self.o[indice])
            antAlpha = alpha[:]
            for i in rangoEstados:
                valor = 0.0
                for estado in rangoEstados:
                    valor += self.a[i][estado] * antAlpha[estado]
                    # print(modOculMarkov.transiciones[i][estado],'*' , antAlpha[estado])
                alpha[i] = self.sensor[i][indice] * valor
                # print("Valor nuevo alpha: ", alpha)
        sum = 0
        value = 0.0
        for i in alpha:
            if sum < i:
                sum = i
                value = self.states[alpha.index(i)]

        return value

    # Devuelve la secuencia de estados mas probables para las observaciones tomadas
    def viterbi(self, observation_prima):
        viterbi = []
        prob = {}  # Diccionario con: "instante - Estado" = probabilidad de ese estado en ese instante
        index = self.o.index(observation_prima[0])
        statesRange = range(len(self.states))  # Rango con todos los posibles estados
        # Generacion de la probabilidad del primer instante(paso 1)
        for i in statesRange:
            viterbi.append(self.sensor[i][index] * self.pi[i])
            prob['0-' + str(i)] = ''
        # print(viterbi)
        # print(prob)
        observationPosibilitiesRange = range(1, len(observation_prima))  # Rango de las posibles observaciones tomadas
        # Generamos los valores de viterbi y las probabilidades de cada estado
        lastState = 0
        states = []
        for observation in observationPosibilitiesRange:
            index = self.o.index(observation_prima[observation])
            antViterbi = viterbi[:]
            # print("Observacion: ", obs)
            for i in statesRange:
                value = 0
                # print("Estado: ", i)
                for state in statesRange:
                    currentValue = self.a[i][state] * antViterbi[state]
                    if value < currentValue:
                        value = currentValue
                        prob[str(observation) + '-' + str(i)] = state
                        # print(modOculMarkov.transiciones[i][estado],'*' , antViterbi[estado])
                viterbi[i] = self.sensor[i][index] * value
                # print("Valor nuevo viterbi: ", viterbi)
                # print("Valor estados: ", prob)
            # Calculamos la probabilidad de la ultima observacion y extraemos los estados mas probables para llegar a ese estado
            if observation == len(observation_prima) - 1:
                value = 0

                for state in statesRange:
                    if value < viterbi[state]:
                        value = viterbi[state]
                        lastState = state
                states.append(self.states[lastState])
                for n in range(len(observation_posibilities) - 1, 0, -1):
                    lastState = prob[str(n) + '-' + str(lastState)]
                    states.append(self.states[lastState])
                    # print(list(reversed(estados)))
        return list(reversed(states))


    def random_generator(self):
        return random.random()

    def __str__(self):
        return "a= " + str(self.a) + "\nb= " + str(self.b) + "\npi= " + str(self.pi) + "\n"

    def generateRandomMarkovModel(self):

        self.pi = []
        self.a = []
        self.b = []
        sum_a = 0
        sum_b = 0
        sum_pi = 0
        t= len(states)
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
            for j in range():
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
        self.sensor = self.b

    def generaObservaciones(self, estado):
        numeroAleatorio = random.random()
        print("Numero aleatorio observacion:", numeroAleatorio)
        prob = 0
        valor = ""
        for observacion in range(len(self.sensor[estado])):
            prob += self.sensor[estado][observacion]
            if numeroAleatorio < prob:
                valor = self.o[observacion]
                break

        return valor

    def Samples(self, numStates):
        secEstados = []
        secObservaciones = []
        estadoActual = 0
        for estado in range(numStates):
            numeroAleatorio = random.random()
            # print("Numero aleatorio estado:",numeroAleatorio)
            if estado == 0:
                valor = 0
                for i in range(len(self.pi)):
                    valor += self.pi[i]
                    if numeroAleatorio < valor:
                        secEstados.append(self.states[i])
                        secObservaciones.append(self.generaObservaciones(i))
                        estadoActual = i
                        break
                print("Valores de estado 0:", secEstados, secObservaciones)
            else:
                valor = 0
                for i in range(len(self.a[estadoActual])):
                    valor += self.a[estadoActual][i]
                    if numeroAleatorio < valor:
                        secEstados.append(self.states[i])
                        secObservaciones.append(self.generaObservaciones(i))
                        estadoActual = i
                        break
                        print(secEstados,secObservaciones)
        return secEstados, secObservaciones

        # Calcular la proporción de estados que coinciden

    def evaluaViterbi(self, estados, estadosCalculados):
        val = 0
        for i in range(len(estados)):
            if estados[i] != estadosCalculados[i]:
                val += 1
        return (val / len(estados))


class localization_item(mom):



    def __init__(self,pi,a,b,o, posibilities, states,t, n,m, walls):
        super().__init__(pi, a, b, o, posibilities, states, t)
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

t = 3
alfa = []
v = []
pr = []
for i in range(t):
    v_lin = []
    al_lin = []
    for j in range(t):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)

observation_posibilities = ["A","B","C","D","AA","AB","AC","AD","BB","BC","BD","CC","CD","DD"]
states = []
A = [[0.7,0.3],[0.3,0.7]]
B = [[0.9,0.1],[0.2,0.8]]
pi = [0.5,0.5]
ob = mom(pi,A,B,observation_posibilities,[],[],[])

print("Obj1\n{}".format(ob))
#print("Obj1 fordward {}".format(ob.fordward(observation_posibilities)))
#print("Obj1 viterbi {}".format(ob.viterbi(observation_posibilities)))
#ob_item = localization_item([],[],[],0,2,5,6)


alfa = []
v = []
pr = []
for i in range(t):
    v_lin = []
    al_lin = []
    for j in range(t):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)

states  = ["A","B","C","D"]
observation_prima = [observation_posibilities[3],observation_posibilities[4]]

ob2 = mom(pi,A,B,observation_posibilities,B,["A","B","C","D"],states)

print("Obj2\n{}".format(ob2))
print("Obj2 fordward {}".format(ob2.fordward(observation_posibilities)))
#print("Obj2 viterbi {}".format(ob2.viterbi(observation_posibilities)))



"""
PL = A
PN = B
XL = C
XN = D
"""
observation_prima = [observation_posibilities[3],observation_posibilities[4]]
#ob3 = mom([],[], [], observation_posibilities,[],["A","B","C","D"],["A","B","C","C"])

#ob2 = mom([],[], [],["AA","AB","AC","AD","BB","BC","BD","CC","CD","DD"],["A","B","C","D"],["A","B","C","C"],0)

"""t = 4
alfa = []
v = []
pr = []
for i in range(t):
    v_lin = []
    al_lin = []
    for j in range(len(ob3.a)):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)
#print("Obj3\n{}".format(ob2))
#print("Obj3 fordward {}".format(ob3.fordward(alfa,2)))
#print("Obj3 viterbi {}".format(ob3.viterbi(v,pr,2)))
"""

t = 4
alfa = []
v = []
pr = []
for i in range(t):
    v_lin = []
    al_lin = []
    for j in range(len(ob2.a[0])):
        al_lin.append(float(1.0))
        v_lin.append(float(1.0))
    alfa.append(al_lin)
    v.append(v_lin)

#ob_item = localization_item([],[], [],["N", "S", "E", "O", "NS", "NE", "NO", "SE", "SO", "EO", "NSE", "NSO", "NEO", "SEO", "NSEO", ""],["N","S","E","O"],[],1,3,5,6)
#print("Obj4\n{}".format(ob_item))
#print("Obj4 fordward {}".format(ob_item.fordward(alfa,2)))
#print("Obj4 viterbi {}".format(ob_item.viterbi(v,pr,2)))


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

