import numpy as np 
import time

class Agent(object):
    def __init__(self, states_size, actions_size, inverse_learning=False, gamma=0.8, k=1.2):
        '''
        states_size         número total de estados
        actions_size        número total de acciones posibles
        inverse_learning    aprendizaje por refuerzo inverso
        gamma               fáctor de descuento  # EL VALOR POR DEFECTO NO TIENE POR QUE SER VÁLIDO
        k                   valor de k para el método k # EL VALOR POR DEFECTO NO TIENE POR QUE SER VÁLIDO
        '''

        self.qTable = np.zeros((states_size,actions_size))
        self.gamma = gamma
        self.k = k
        self.inverse = inverse_learning
        self.current_state = None
        self.StateActionPair = []


    def set_initial_state(self, state):
        '''
        Definir el estado inicial y otros posibles valores iniciales en cada reintento.
        '''
        self.current_state = state
        self.StateActionPair = []

    def act(self):
        '''
        A partir del estado actual self.current_state devolver la acción adecuada
        Acciones posibles
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        '''
        #
        actions = [0,1,2,3]
        actionWeights = self.getProbabilities()

        #Elige la siguiente accion en funcion de los pesos proporcionados por getProbabilities()
        action = np.random.choice(actions, p=actionWeights)

        self.lastAction = action
        self.lastState = self.current_state

        return action

    def learn(self, state, reward):
        '''
        Llama a la funcion de aprendizaje inversa o clasica en funcion de self.inverse
        '''
        if(self.inverse):
            self.inverseLearn(state,reward)
        else:
            self.stdLearn(state,reward)
        return

    def stdLearn(self, state, reward):
        '''
        Aprendizaje por refuerzo clasico
        '''

        q_target = reward + self.gamma * np.max(self.qTable[state, :])

        self.qTable[self.lastState, self.lastAction] = q_target
        self.current_state = state

        return

    def updateQTable(self,StateActionPairList,index=0):
        '''
        Funcion recursiva que implementa la actualizacion de la
        tabla Q en funcion de los pares (Estado,accion) transcurridos
        hasta llegar al estado meta.
        '''
        if(index == len(StateActionPairList)):
            return
        else:
            self.updateQTable(StateActionPairList, index+1)

            q_target = StateActionPairList[index-1][2] + self.gamma * np.max(self.qTable[StateActionPairList[index][0], :])

            self.qTable[StateActionPairList[index-1][0], StateActionPairList[index-1][1]] = q_target

            return

    def inverseLearn(self, state, reward):

        # Si el estado es terminal
        if (reward == 100):

            #Añadimos el estado actual y el ultimo a la lista de nodos
            self.StateActionPair.append([self.current_state, self.lastAction, reward])
            self.StateActionPair.append([state, -1, 0])

            #Actualizamos recursivamente los valores de Q en la tabla para los pares (Estado,Accion) que hemos transitado
            self.updateQTable(self.StateActionPair)

            #Vaciamos la estructura [Estado,accion,recompensa]
            self.StateActionPair.clear()


        else:
            # Guardamos [Estado,accion,recompensa] para actualizar los valores de la tabla Ql cuando alcancemos la meta
            self.StateActionPair.append([self.current_state,self.lastAction,reward])


        self.current_state = state

        return



    def getProbabilities(self):

        state = self.current_state
        val = np.asarray(self.qTable[state,:]).flatten(1)

        #Obtener acciones para estado actual
        qValforActioninState = np.asarray(self.qTable[state, :])

        #Calcular probabilidad de cada accion en funcion de k
        probs = np.power(self.k, val)/np.sum(np.power(self.k, val))

        return probs
        sys.exit(2)




class ParameterOptimizer():
    def optimizeParams(self):
        gamma = 0.0
        k = 0.9
        av = []

        for gamma in np.arange(1.51, 2.01, 0.04):
            for k in np.arange(1.01,2.01,0.04):
                for times in range(3):
                    env = LostInSpace()
                    agent = Agent(env.get_states_size(),env.get_actions_size(), gamma=gamma, k=k,inverse_learning=True)  # LOS VALORES POR DEFECTO NO TIENE POR QUE SER VÁLIDOS

                    episode_count = 10000  # TODO Modificar si es necesario para estudiar si el agente aprende
                    reward = 0
                    done = False

                    for _ in range(episode_count):
                        agent.set_initial_state(env.reset())
                        while True:
                            action = agent.act()
                            state, reward, done = env.step(action)
                            # Out of space, no aprender
                            if state >= env.get_states_size():
                                break
                            agent.learn(state, reward)
                            if done:
                                break

                    av.append(env.total / env.times)

                av.sort()
                #print('For gamma = %f , k = %f the average is %s:\t' % (gamma, k , str(env.total / env.times)))
                print('"%f","%f","%f"' % ( av[1], gamma, k,))
                av.clear()

        print("No Inverse")

        for gamma in np.arange(0.05, 2.0, 0.01):
            for k in np.arange(1.05,3.0,0.01):
                for times in range(3):
                    env = LostInSpace()
                    agent = Agent(env.get_states_size(),env.get_actions_size(), gamma=gamma, k=k,inverse_learning=False)  # LOS VALORES POR DEFECTO NO TIENE POR QUE SER VÁLIDOS

                    episode_count = 10000  # TODO Modificar si es necesario para estudiar si el agente aprende
                    reward = 0
                    done = False

                    for _ in range(episode_count):
                        agent.set_initial_state(env.reset())
                        while True:
                            action = agent.act()
                            state, reward, done = env.step(action)
                            # Out of space, no aprender
                            if state >= env.get_states_size():
                                break
                            agent.learn(state, reward)
                            if done:
                                break

                    av.append(env.total / env.times)

                av.sort()
                #print('For gamma = %f , k = %f the average is %s:\t' % (gamma, k , str(env.total / env.times)))
                print('"%f","%f","%f"' % ( av[1], gamma, k,))
                av.clear()





# NO MODIFICAR LA CLASE LostInSpace ##########################################
class LostInSpace(object):
    def __init__(self, max_steps=150, space_size=10):
        '''
        Posición inicial X=0, Y=0
        Posición objectivo X=4, Y=2
        Acciones: Codificadas con un entero
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        Solo se puede realizar hasta max_step pasos
        El tamaño del tablero es de space_size x space_size
        '''
        self.position=np.array([0.,0.]) # X, Y
        self.actions = np.array([[0.,1.],[0.,-1.],[-1.,0.],[1.,0.]]) # UP 0, DOWN 1, LEFT 2, RIGHT 3
        self.target=np.array([7.,7.])
        self.max_steps = max_steps
        self.space_size = space_size
        self.steps = 0
        self.total = 0
        self.times = 0

    def step(self, action):
        '''
        El agente avanza una posición en la dirección indicada
        Acciones: Codificadas con un entero
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        '''
        reward = 0
        done = False
        printY = True

        if(printY==True):
             for x in range(20):
                 for y in range(20):
                     if(int(self.position[0]) == x) and ((int(self.position[1])) == y):
                         print("# ",end='')
                     elif( int(self.target[0]) == x) and ((int(self.target[1])) == y):
                         print("T ",end='')
                     else:
                         print("_ ",end='')
                 print("\n")

             print('\x1b[2K\r')
        time.sleep(0.064)
        
        self.steps+=1
        self.position += self.actions[action]
        state = self.get_state()

        if self.position.max() > self.space_size-1 or self.position.min() < 0:
            done = True
            self.total += self.max_steps # Añadimos al total el número máximo de pasos porque nos salimos
            self.times += 1
            state = -1
            print('Fin. Pasos:\t'+ str(self.steps) +'\tMedia:\t' + str(self.total/self.times) + '\tOut of space')

        if self.steps >= self.max_steps:
            done = True
            self.total += self.steps
            self.times += 1
            print(self.total/self.times)
            print('Fin. Pasos:\t'+ str(self.steps) +'\tMedia:\t' + str(self.total/self.times) + '\tOut of steps')


        if np.absolute(self.position - self.target).sum() == 0.0:
            self.total += self.steps
            self.times += 1
            print('Fin. Pasos:\t'+ str(self.steps) +'\tMedia:\t' + str(self.total/self.times) + '\tWell done little boy')
            reward = 100
            done = True

        return state, reward, done

    def reset(self):
        '''
        Volvemos a la posición inicial
        '''
        self.position=np.array([0.0,0.0])
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        '''
        Devuelve un entero representando el estado en el que se encuentra el agente.
        El número de estados será igual max_steps al cuadrado
        '''
        return int(self.position[0]*self.space_size+self.position[1])

    def get_states_size(self):
        return self.space_size**2

    def get_actions_size(self):
        return self.actions.shape[0]
# NO MODIFICAR LA CLASE LostInSpace ##########################################








if __name__ == '__main__':
    # Lost in Space
    #
    #
    env = LostInSpace()
    agent = Agent(env.get_states_size(), env.get_actions_size(),inverse_learning=True) # LOS VALORES POR DEFECTO NO TIENE POR QUE SER VÁLIDOS
    
    episode_count = 100000 # TODO Modificar si es necesario para estudiar si el agente aprende
    reward = 0
    done = False
    
    for _ in range(episode_count):
        agent.set_initial_state(env.reset())
        while True:
            action = agent.act()
            state, reward, done = env.step(action)
            # Out of space, no aprender
            if state >= env.get_states_size():
                break
            agent.learn(state, reward)
            if done:
                break
    
    #
