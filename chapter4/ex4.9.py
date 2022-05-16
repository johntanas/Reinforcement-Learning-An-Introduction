import numpy as np
import matplotlib.pyplot as plt


def train(ph=0.4, Theta=0.000001):
    V = [0]*100
    for i in range(0, 100):
        V[i] = np.random.random() * 1000
    V[0] = 0
    pi = [0]*100
    counter = 1
    while True:
        Delta = 0
        for s in range(1, 100):  # for each state
            old_v = V[s]
            v = [0] * 51
            for a in range(1, min(s, 100 - s) + 1):
                v[a] = 0
                if a + s < 100:
                    v[a] += ph * (0 + V[s + a])
                    v[a] += (1 - ph) * (0 + V[s - a])
                elif a + s == 100:
                    v[a] += ph
                    v[a] += (1 - ph) * (0 + V[s - a])
            op_a = np.argmax(v)
            pi[s] = op_a
            V[s] = v[op_a]
            Delta = max(Delta, abs(old_v - V[s]))
        counter += 1
        if counter % 1000 == 0:
            print("train loop" + str(counter))
            print("Delta =" + str(Delta))
        if Delta < Theta:
            break
    return [V[1:100], pi[1:100]]
