#a = [0.5, 0.3, 0.2]
#b = [0.9, 0.8, 0.7]
#c = np.random.choice(b, 5, a)
import sys

import numpy as np
import matplotlib.pyplot as plt
#print(np.append([[1,2,3],[4,5,6]], [[7,8,9]] , axis = 1))
##a = np.array([[1, 1], [2, 2], [3, 3]])
##print(a)
#b = np.append(a, [[1,5]], axis=0)
#print(b)

#a=[2,45,45,64,0,698,2,45]
#print(np.sqrt(np.var(a)))

#a = [1,2,3,4,5,6,7]
#b = [2,3,4,5,6,7,8]
#df = pd.DataFrame(a, columns = ['cn'])
#df['cn1'] = b
#print(df)
#df.loc['mean','cn'] = df['cn'].mean()
#print(df)
#rnd = np.random
#rnd.seed(0)

#d = [0.0 for i in range(0, 1000)]
#d2 = [0.0 for i in range(0,1000)]
#for i in range(0,1000):
#    d[i] = (rnd.rand()*40000000+ 680000000)/1000
#    d2[i] = (rnd.rand()*12000 + 1980000)/1000000
#print(d)
#print(d2)
#sce = 100
#a1 = [199950000, 199990000]
#a2 = [0.4, 0.6]
#d3 = numpy.random.choice(a1, 1000, a2)
#print(d3)
#d4 = ["true" for i in range(0, 1000)]
#da = {'obj':d, 'budgut':d3, 'gap': d2, 'feasibility':d4}
#df = pd.DataFrame(da)
#df.loc['mean','obj'] = np.mean(d)
#df.loc['variance','obj'] = math.sqrt(np.var(d))
#df.to_csv('./result/SAA_1000_80M_10y_1000sce.csv')

#a = math.sqrt(math.sqrt(16))

#print(a)#
#print
#parameter
#a = set()
##a.add(5)
#a.add(10)
#a.add(5)
#print(a)



class dijkstra():
    def __init__(self, vertice):
        self.V = vertice
        self.graph = [[0.0 for row in range(vertice)] for column in range(vertice)]
    def minnode(self, dist, spt):
        minnode = 0
        min = sys.maxsize

        for node in range(self.V):
            if dist[node] < min and spt[node] == False:
                min = dist[node]
                minindex = node

        return minindex

    def algorithm(self, src):
        s = set()
        dist = [sys.maxsize] * self.V
        dist[src] = 0.0
        for i in range(self.V):
            s.add(i)
        while len(s) != 0:
            a = self.minnode(dist)

            for j in range(self.V):
                if self.graph[a][j] !=0 and a in s == False and dist[j] > dist[i] + self.graph[a][j]:
                    dist[j] = dist[i] + self.graph[a][j]
            s.remove(a)

