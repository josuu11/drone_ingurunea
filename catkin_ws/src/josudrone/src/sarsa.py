import random
import numpy as np # Q balioak kanpo fitxategi batean gordetzeko erabiliko den liburutegia inportatu

class Sarsa:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward 
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)

    def save(self, fitxategia):
    	np.save(fitxategia, self.q)
    	print(fitxategia, "Fitxategia gorde da")
    	print(self.q)
    	
    def load(self, fitxategia):
    	# self.q = np.load(fitxategia).item() Object array izanik allow.. =True jarri
    	self.q = np.load(fitxategia, allow_pickle=True).item()
    	print(fitxategia, "Fitxategia kargatu da")
