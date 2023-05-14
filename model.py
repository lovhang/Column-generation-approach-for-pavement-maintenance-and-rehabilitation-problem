import numpy as np
import math
import sys
import time

import numpy.random
from docplex.mp.model import Model
import itertools
import numpy as np
import pandas as pd
from docplex.cp import expression
import matplotlib.pyplot as plt

rnd = np.random
rnd.seed(0)
#S = int(sys.argv[1]) #number of road
#B = float(sys.argv[2])  # budget
#T = int(sys.argv[3]) #time period
#g = int(sys.argv[4]) #maximum treatment for single pavement
#scenario = int(sys.argv[5]) # number of scenario to generate SAA result
#casenum = int(sys.argv[6]) # Number of scenario for different case running
#casecheck = int(sys.argv[7])  # Number of scenarios for different case to check SAA result
S = 10
R = 4
T = 10
B = 2000000
g = 5
scenario = 20
casenum = 1
casecheck = 20
Qmax = 100  # maximum capacity
Qmin = 20  # minimum capacity

AADT = {}  # AADT set
q0 = {}  # initial capacity
s = [i for i in range(1, S + 1)]  # road segment set
r = [i for i in range(1, R + 1)]  # treatment set include donothing
rs = [i for i in range(2, R + 1)]  # treatment set not include do nothing
t = [i for i in range(0, T + 1)]  # time step set
dr = 0.9  # deterioration rate
e = {1: 0.0, 2: 15.0, 3: 25.0, 4: 40.0}  # improvement value
cost = {1: 0.0, 2: 50000.0, 3: 170000.0, 4: 285000.0}  # treatment cost
for i in s:
    # AADT[i] = rnd.rand() * 100
    AADT[i] = 1.0
for i in s:
    q0[i] = rnd.rand() * (100 - Qmin - 10) + Qmin + 10

unit = ''
pa = B
if B / 1000000 < 1:
    unit = str(B)
elif 1000000 < B < 1000000000:
    unit = str(B / 1000000) + 'M'
elif B / 1000000000 >= 1:
    unit = str(B / 1000000000) + 'B'
address = './result/output_S' + str(S) + '_B' + unit + '_T' + str(T) + '_G' + str(g) + '.txt'
with open(address,'w') as file1:
    file1.write("# of road segments: {}".format(S))
    file1.write(" \n")
    file1.write("Planning period(year): {}".format(T))
    file1.write(" \n")
    file1.write("Budget: {}".format(B))
    file1.write(" \n")
    file1.write("minimum capac  ity: {}".format(Qmin))
    file1.write(" \n")
    file1.write("maximum treatment for a segment: {}".format(g))


class milp:
    mdl = Model('PMS')
    x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
    q = mdl.continuous_var_dict([(i, j) for i in s for j in t], lb=Qmin, ub=Qmax, name='q')
    #y = mdl.integer_var_dict()
    obj_value = 0.0
    totalcost = 0.0
    solvetime = 0.0

    def model1(self):
        self.mdl.maximize(self.mdl.sum(AADT[i] * self.q[i, j] for i in s for j in t))
        self.mdl.add_constraints(
            self.q[i, k + 1] == self.q[i, k] * dr + self.mdl.sum(self.x[i, j, k] * e[j] for j in r) for i in s
            for k in range(0, T))
        self.mdl.add_constraint(self.mdl.sum(self.x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        # mdl.add_constraints(q[i,k] >= Qm for i in s for k in t)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs for k in t) <= g for i in s)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in r) == 1 for i in s for k in t)
        # Initial state
        self.mdl.add_constraints(self.q[i, 0] == q0[i] for i in s)
        self.mdl.print_information()
        #self.mdl.export_as_lp("milp_1")

    def model2(self):
        self.mdl.maximize(self.mdl.sum(AADT[i] * q0[i] * (dr ** j) +self.mdl.sum(self.x[i, k, l] * e[k] * (dr ** (j-l-1))
                                                                                 for k in r for l in range(0, j) )for i in s for j in t))
        self.mdl.add_constraints(q0[i]*(dr**j)+self.mdl.sum(self.x[i, k, l] * e[k] * (dr ** (j-l-1)) for k in r
                                                            for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1))
        self.mdl.add_constraints(q0[i]*(dr**j)+self.mdl.sum(self.x[i, k, l] * e[k] * (dr ** (j-l-1)) for k in r
                                                            for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1))
        self.mdl.add_constraint(self.mdl.sum(self.x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        # mdl.add_constraints(q[i,k] >= Qm for i in s for k in t)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs for k in t) <= g for i in s)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in r) == 1 for i in s for k in t)
        self.mdl.print_information()

        #self.mdl.export_as_lp("milp_2")

    def solve(self):
        # self.mdl.time_limit = 72000
        solution = self.mdl.solve(log_output=True)
        if solution is None:
            print(self.mdl.solve_details)
            self.mdl.end()
        else:
            print("model solved")
            print(solution)
            # print(self.mdl.solve_details)
            self.solvetime = round(self.mdl.solve_details.time, 3)
            self.obj_value = solution.get_objective_value()
            print("model objective value: " + str(self.obj_value))
            for i in s:
                for j in r:
                    for k in t:
                        if self.x[(i, j, k)].solution_value > 0.9:
                            self.totalcost = self.totalcost + cost[j]
            # print(str(self.totalcost))

    def end(self):
        self.mdl.end()

    def runmodel1(self):
        self.model1()
        self.solve()
        # self.plot()
        self.output()
        self.end()

    def runmodel2(self):
        self.model2()
        self.solve()
        # self.output()
        self.end()

    def output(self):
        # file2 = open("output_"+sys.argv[3]+".txt", "w")
        with open(address, 'a') as file2:
            file2.write("=============MILP solution=============")
            file2.write(" \n")
            file2.write(" \n")
            file2.write("Solving time: " + str(self.solvetime) + "s")
            file2.write(" \n")
            file2.write(" \n")
            file2.write("objective value: " + str(self.obj_value))
            file2.write(" \n")
            file2.write(" \n")
            file2.write("Total cost: " + str(self.totalcost))
            file2.write(" \n")
            file2.write(" \n")
            file2.write("S ")
            for k in t:
                file2.write(" " + str(k) + " ")

            file2.write(" \n")

            for i in s:
                file2.write(str(i) + " ")
                for k in t:
                    # cer = False
                    # file2.write(" 0 ")
                    for j in r:
                        # print(str(i)+"."+str(j)+"."+str(k)+": "+str(x[(i,j,k)].solution_value))
                        if self.x[(i, j, k)].solution_value > 0.9:

                            if j == 1:
                                file2.write(" 0 ")
                            elif j == 2:
                                file2.write(" P ")
                            elif j == 3:
                                file2.write(" M ")
                            elif j == 4:
                                file2.write(" R ")
                file2.write(" \n")

            file2.write(" \n")

            file2.write("{0:<5}".format("q"))

            for k in t:
                file2.write("{0:<5}".format(k))

            file2.write(" \n")
            for i in s:
                file2.write("{0:<5}".format(i))
                for k in t:
                    file2.write("{0:<5}".format(str(math.floor(self.q[(i, k)].solution_value))))
                file2.write(" \n")

            file2.write(" \n")

        # i=1; j=2; k=3
        # print(solution.get_value("x_"+str(i)+"_"+str(j)+"_"+str(k)))


class cg():
    # obj_value = 0
    pcl = [[[0.0 for k in range(T + 1)] for j in range(R + 1)] for i in range(S + 1)]  # primal model cost coefficient list
    cll = [[[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in range(0, S + 1)]]  # column list
    ldl = [0]  # solution lambda list
    ptc = 0.0  #primal problem constant value
    subObj = 0.0 #objective value of subproblem
    masObj = 0.0 #objective value of masterproblem
    w = 0.0  # dual value for budget constraint
    al = 0.0  # dual value for lambda constraint
    Totalcost = 0.0
    solvetime = 0
    itrnum = 0
    bestsol = []
    bestobj = 0
    bestbud = 0
    gap = 0
    feasibility = False
    def getcoe(self):
        mdl = Model('coefi model')
        x = mdl.continuous_var_dict([(i, j, k) for i in s for j in r for k in t], lb=0, ub=1, name='x')
        mdl.minimize(-1*mdl.sum(AADT[i] * q0[i] * (dr ** j) + mdl.sum(x[i, k, l] * e[k] * (dr ** (j-l-1))for k in r for l in range(0, j) )for i in s for j in t))
        mdl.add_constraints(x[i, j, k] == 0 for i in s for j in r for k in t)
        #mdl.export_as_lp("coefficient_problem")
        for i in s:
            for j in r:
                for k in t:
                    self.pcl[i][j][k] = mdl.objective_coef(x[i, j, k])
        #print("pcl check",self.pcl)
        solution = mdl.solve()
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
        else:
            self.ptc = solution.get_objective_value()
            #'ptc: ' + str(self.ptc))
            mdl.end()

    def initiate(self, pavnum):
        #Initiate problem for single pavement
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]   # create new column by solution
        sepobj = 0.0
        fsb = False
        mdl = Model('initiate problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(-self.pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            >= Qmin for j in range(1, T+1))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            <= Qmax for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('initiate problem_' + str(pavnum))
        solution = mdl.solve(log_output = False)
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            #print('initiate problem solved')
            sepobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.99:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb,nclp, sepobj

    def runiniate(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tobj = 0.0
        for i in s:
            #print('=========initial problem for pavement ' + str(i) +' =============')
            fsb, nclp, sepobj = self.initiate(i)
            if fsb is True:
                ncl[i] = nclp
                tobj = tobj + sepobj

            else:
                print('initial problem '+str(i)+' infeasible')
                return False
        print('obj: ' + str(tobj))
        self.cll.append(ncl)
        self.subObj = tobj
        print('initial problem solved')

    def masterp(self):
        cnum =len(self.cll) #column number
        cn =[i for i in range(0,cnum)] #column index
        self.ldl = [0.0 for i in range(0, cnum)]
        mdl = Model('master problem')
        ld = mdl.continuous_var_dict([i for i in cn], lb=0, ub=1, name='ld') # variables lambda
        mdl.minimize(self.ptc+mdl.sum(self.pcl[i][j][k]*self.cll[m][i][j][k]*ld[m] for i in s for j in r for k in t
                                      for m in cn))
        mdl.add_constraint(mdl.sum(cost[j]*self.cll[l][i][j][k]*ld[l] for i in s for j in r for k in t for l in cn) <= B, 'ct1')
        mdl.add_constraint(mdl.sum(ld[l] for l in cn) == 1.0, 'ct2')
       # mdl.export_as_lp('master_problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print('master problem infeasible')
            print(mdl.solve_details)
            mdl.end()
            return False
        else:
            print('master problem solved')
            self.w = mdl.dual_values(mdl.find_matching_linear_constraints('ct1'))[0]
            self.al = mdl.dual_values(mdl.find_matching_linear_constraints('ct2'))[0]
            self.masObj = solution.get_objective_value()
            print('master objective: ' + str(self.masObj))
            for i in range(0,cnum):
                self.ldl[i] = ld[i].solution_value
                print(str(round(ld[i].solution_value,3)), end = " ")
            print(' ')
            print('w: '+str(self.w)+'\n'+'alpha: '+str(self.al))
            mdl.end()
            return True

    def sub(self):
        mdl = Model('sub problem')
        x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        mdl.maximize(mdl.sum((self.w*cost[j]-self.pcl[i][j][k]) * x[i, j, k] for i in s for j in r for k in t)+self.al)
        mdl.add_constraints(q0[i]*(dr**j)+mdl.sum(x[i, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            >= Qmin for i in s for j in range(1, T+1))
        mdl.add_constraints(q0[i]*(dr**j)+mdl.sum(x[i, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            <= Qmax for i in s for j in range(1, T+1))
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in rs for k in t) <= g for i in s)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in r) == 1 for i in s for k in t)
        #mdl.export_as_lp("subproblem")
        solution = mdl.solve(log_output=False)

        if solution is None:
            print('sub problem infeasible')
            print(mdl.solve_details)
            mdl.end()
            return False
        else:
            print('sub problem solved')
            #print(solution)
            self.subObj = solution.get_objective_value()
            print("reduce cost: "+str(self.subObj))
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]  # create new column based on solution
            for i in s:
                for j in r:
                    for k in t:
                        if x[i, j, k].solution_value > 0.98:
                            ncl[i][j][k] = 1.0
                        else:
                            ncl[i][j][k] = 0.0
            #print(ncl[1])
            self.cll.append(ncl)  # add new column to column list
            #print(len(self.cll))
            mdl.end()
            return True

    def subp(self, pavnum):
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]
        subobj = 0.0
        fsb = False
        mdl = Model('sub problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum((self.w*cost[j]-self.pcl[pavnum][j][k]) * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            >= Qmin for j in range(1, T+1))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            <= Qmax for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('sub problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            subobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.98:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb, nclp, subobj

    def runsubproblem(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tempobj = self.al
        for i in s:
            #print('=========sub problem for pavement ' + str(i) +' =============')
            fsb, nclp, subobj = self.subp(i)
            if fsb == True:
                ncl[i] = nclp
                tempobj = tempobj + subobj
            else:
                print('sub  problem' + str(i) +' infeasible')
                return False
        self.cll.append(ncl)
        self.subObj = tempobj
        print('sub problem solved')
        print("reduce cost: "+str(tempobj))
        return True

    def costcalculate(self, plan):
        plancost = 0.0
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        plancost = plancost + cost[j]*1.0

        return plancost

    def objectivecalculate(self, plan):
        objectivevalue = self.ptc
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        objectivevalue = objectivevalue + self.pcl[i][j][k]
        return -objectivevalue

    def result(self):
        tempbestobj = 0.0
        tempbestsol = []
        tempbestbud = 0
        tempfeasibility = False
        for i in range(1,len(self.cll)):
            temp = self.cll[i]
            tempcost = self.costcalculate(temp)
            tempobj = self.objectivecalculate(temp)
            #print("=== "+str(i)+" ====")
            #print(tempcost)
            #print(tempobj)
            if tempcost < B and tempobj>tempbestobj:
                tempbestsol = temp
                tempbestobj = self.objectivecalculate(temp)
                tempbestbud = self.costcalculate(temp)
                tempfeasibility = True
        #print(tempbestobj)
        self.bestsol = tempbestsol
        self.bestobj = tempbestobj
        self.bestbud = tempbestbud
        self.gap = -(-self.masObj-self.bestobj)/self.masObj
        #(hint) bestobj 8116 masobj -8468
        self.feasibility = tempfeasibility
        print("bestobj: ", self.bestobj, "masterobj: ", self.masObj, "gap", self.gap)
    def runcg(self):
        self.getcoe()
        now = time.time() #calculate time at the beginning of iteration
        if self.runiniate() == False:
            file1 = open(address, "a")
            file1.write(" initiate infeasible")
        else:
            itr = 0
            #print('sub obj: '+ str(self.subObj))
            while self.subObj > 0.0001 and itr <50:
                print('=========itration '+str(itr)+'===========')
                if not self.masterp():
                    print(" masterproblem infeasible at iteration "+str(itr))
                    file1 = open(address, "a")
                    file1.write(" masterproblem infeasible at iteration "+str(itr))
                    break
                if not self.runsubproblem():
                #if not self.sub():
                    print(address+str(itr))
                    file1 = open("./result/output_"+str(S)+"_"+str(B)+".txt", "a")
                    file1.write(" subproblem infeasible at iteration "+str(itr))
                    break
                #print("reduce cost: "+ str(self.subobj))
                itr += 1
            print("=====end of iteration=====")
            later = time.time() #stop calculation at the end of iteration
            self.solvetime = int(later - now)
            self.itrnum = itr
            self.result()

    def output(self):
        #print(self.ldl)
        #file1 = open("output_"+sys.argv[3]+".txt", "a")
        with open(address, 'a') as file1:
        #file1 = open(address, "a")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("=============CG solution=============")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("Solving time: "+str(self.solvetime)+"s")
            file1.write(" \n")
            file1.write("iteration: "+str(self.itrnum))
            file1.write(" \n")
            file1.write("Feasibility: "+str(self.feasibility))
            file1.write(" \n")
            file1.write(" \n")
            file1.write("objective value: "+str(-self.masObj))
            file1.write(" \n")
            file1.write(" \n")
            for i in range(0,len(self.ldl)):
                if self.ldl[i] >0.001:
                    print(self.ldl[i])
                    file1.write("lambda("+str(i)+"): "+str(self.ldl[i]))
                    file1.write(" \n")
                    tempobjective = self.objectivecalculate(self.cll[i])
                    print(tempobjective)
                    print('---------------')
                    file1.write("Objective value: "+str(tempobjective))
                    file1.write(" \n")
                    tempcost = self.costcalculate(self.cll[i])
                    print(tempcost)
                    print('---------------')
                    file1.write("Total cost: "+str(tempcost))
                    file1.write(" \n")
                    file1.write(" \n")
            file1.write("Objective value of best solution: "+str(self.bestobj))
            file1.write(" \n")
            file1.write("Budget of best solution: "+str(self.bestbud))
            file1.write(" \n")
            file1.write("Gap: "+ str(round(self.gap*100,3))+"%")
            file1.write(" \n")
            file1.write(" \n")
            file1.close()




class ScModel():
    mdl = Model('PMS')
    x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
    q = mdl.continuous_var_dict(1)
    obj_value = 0.0
    totalcost = 0.0
    solvetime = 0.0
    prob = [0.8, 0.2]
    dt = [0.9, 0.8]
    dts = []
    problist = []
    scenum = 0.0
    par = [] # parameter for q0, not lambda
    par1 = [] # parameter for variables x

    def __init__(self):
        #
        sce = [] #list of possible deterioration rate for each pavement
        prob1 = [] #probability list
        scet = [] #list of possible deterioration rate for each pavement for each time
        prob2 = []
        for i in s:
            sce.append(self.dt)
            prob1.append(self.prob)
        #print(sce)
        so = list(itertools.product(*sce))
        #print(so)
        #print(len(so))
        prob1a = list(itertools.product(*prob1))
        for k in range(0, T): #deterioration happen at the start of year
            scet.append(so)
            prob2.append(prob1a)
        #print(len(scet))
        sot = list(itertools.product(*scet))
        #print(len(sot))
        prob2a = list(itertools.product(*prob2))
        #print(sot[0])
        #print("prob2a", prob2a)
        self.scenum = len(sot) #length of the scenarios
        print('scenarios number', self.scenum)
        self.dts = [[[0.0 for k in range(0,self.scenum)] for j in range(0, T+1)] for i in range(0, S+1)] #every set start with 0

        for i in range(0, self.scenum):
            for j in range(0, T):
                for k in s:
                    self.dts[k][j][i] = sot[i][j][k-1]
        self.problist = [0.0 for i in range(0, self.scenum)] #probability for each scenarios
        print(self.dts[1][1])
        for i in range(0, self.scenum):
            temp = 1.0
            ele = prob2a[i]
            for j in ele:
                for k in j:
                    temp = temp * k
            self.problist[i] = temp
        # create q variable based on scenarios

        # calculate lambda**t
        self.par = [[[1.0 for k in range(0, self.scenum)] for j in range(0, T+1)] for i in range(0, S+1)] # parameter for q0, not lambda
        for k in range(0, self.scenum):
            for i in range(1, S+1):
                for j in range(1, T+1):
                    temp = 1.0
                    for l in range(0,j):
                        temp = temp * self.dts[i][l][k]
                        #print(i, j, temp)
                    self.par[i][j][k] = temp
                    #print("par ", self.par[i][j][k])
        #print("check ", self.dts[1][0][3])
        #print("check0" ,self.par[1][1])
        self.par1 = [[[[1.0 for k in range(0, self.scenum)]for l in range(0, T)] for j in range(0, T+1)] for i in range(0, S+1)]

        for k in range(0, self.scenum):
            for j in range(1, T+1):
                for i in s:
                    for l in range(0, j):
                        temp = 1.0
                        for m in range(0, j-l-1):
                            temp = temp * self.dts[i][j-m-1][k]
                        self.par1[i][j][l][k] = temp
        print("check1", self.dts[1][1][0])
        print("check2", self.par1[1][2][0][0])

    def model1(self):
        self.mdl = Model('PMS')
        self.x = self.mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        self.q = self.mdl.continuous_var_dict([(i, j, k) for i in s for j in t for k in range(0, self.scenum)], lb=Qmin, ub=Qmax, name='q') # q variables for linear model
        self.mdl.maximize(self.mdl.sum(AADT[i] * self.q[i, j, k] * self.problist[k] for i in s for j in t for k in range(0, self.scenum)))
        self.mdl.add_constraints(
            self.q[i, k + 1, m] == self.q[i, k, m] * self.dts[i][k][m] + self.mdl.sum(self.x[i, j, k] * e[j] for j in r) for i in s
            for k in range(0, T) for m in range(0, self.scenum))
        self.mdl.add_constraint(self.mdl.sum(self.x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        # mdl.add_constraints(q[i,k] >= Qm for i in s for k in t)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs for k in t) <= g for i in s)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in r) == 1 for i in s for k in t)
        # Initial state
        self.mdl.add_constraints(self.q[i, 0, m] == q0[i] for i in s for m in range(0, self.scenum))
        self.mdl.print_information()
        #self.mdl.export_as_lp("SCmilp_1")

    def model2(self):
        self.mdl = Model('PMS')
        self.x = self.mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        self.mdl.maximize(self.mdl.sum(self.mdl.sum(AADT[i] * q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l]
                         * e[k] * (self.par1[i][j][l][m]) for k in r for l in range(0, j)) for i
                                       in s for j in range(0, T+1) )*self.problist[m] for m in range(0, self.scenum)))
        self.mdl.add_constraints(q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                for k in r for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1) for m in range(0, self.scenum))
        self.mdl.add_constraints(q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                for k in r for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1) for m in range(0, self.scenum))
        self.mdl.add_constraint(self.mdl.sum(self.x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        # mdl.add_constraints(q[i,k] >= Qm for i in s for k in t)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs for k in t) <= g for i in s)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in r) == 1 for i in s for k in t)
        self.mdl.print_information()
        #self.mdl.export_as_lp("SCmilp_2")

    def solve(self):
        # self.mdl.time_limit = 72000
        solution = self.mdl.solve(log_output=False)
        if solution is None:
            print(self.mdl.solve_details)
            self.mdl.end()
        else:
            print("model solved")
            #print(solution)
            # print(self.mdl.solve_details)
            self.solvetime = round(self.mdl.solve_details.time, 3)
            self.obj_value = solution.get_objective_value()
            print("model objective value: " + str(self.obj_value))
            for i in s:
                for j in r:
                    for k in t:
                        if self.x[(i, j, k)].solution_value > 0.9:
                            self.totalcost = self.totalcost + cost[j]
            # print(str(self.totalcost))

    def end(self):
        self.mdl.end()

    def runmodel1(self):
        self.model1()
        self.solve()
        self.output()
        self.end()

    def runmodel2(self):
        self.model2()
        self.solve()
        #self.output()
        self.end()

    def output(self):
        with open('./result/output_' + str(S) + '_' + unit + '_' + str(T) + 'y_SC.txt','w') as file1:
            file1.write("# of road segments: {}".format(S))
            file1.write(" \n")
            file1.write("Planning period(year): {}".format(T))
            file1.write(" \n")
            file1.write("Budget: {}".format(B))
            file1.write(" \n")
            file1.write("minimum capac  ity: {}".format(Qmin))
            file1.write("=============MILP solution=============")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("Solving time: " + str(self.solvetime) + "s")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("objective value: " + str(self.obj_value))
            file1.write(" \n")
            file1.write(" \n")
            file1.write("Total cost: " + str(self.totalcost))
            file1.write(" \n")
            file1.write(" \n")
            file1.write("S ")
            for k in t:
                file1.write(" " + str(k) + " ")

            file1.write(" \n")

            for i in s:
                file1.write(str(i) + " ")
                for k in t:
                    # cer = False
                    # file2.write(" 0 ")
                    for j in r:
                        # print(str(i)+"."+str(j)+"."+str(k)+": "+str(x[(i,j,k)].solution_value))
                        if self.x[(i, j, k)].solution_value > 0.9:

                            if j == 1:
                                file1.write(" 0 ")
                            elif j == 2:
                                file1.write(" P ")
                            elif j == 3:
                                file1.write(" M ")
                            elif j == 4:
                                file1.write(" R ")
                file1.write(" \n")

            file1.write(" \n")

            file1.write("{0:<5}".format("q"))

            for k in t:
                file1.write("{0:<5}".format(k))

            file1.write(" \n")

            file1.write(" \n")

class SAAmodel():
    prob = [0.5, 0.5]
    dt = [0.9, 0.8]
    scnum = 0
    dts = []
    par = [] # parameter for q0, not lambda
    par1 = [] # parameter for variables x
    totalcost = 0.0
    solvetime = 0.0
    mdl = Model('PMS')
    x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
    #print("q0 ", q0[1])
    pcl = [[[0.0 for k in range(T + 1)] for j in range(R + 1)] for i in range(S + 1)]  # primal model cost coefficient list
    cll = [[[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in range(0, S + 1)]]  # column list
    ldl = [0]  # solution lambda list
    ptc = 0.0  #primal problem constant value
    subObj = 0.0 #objective value of subproblem
    masObj = 0.0 #objective value of masterproblem
    w = 0.0  # dual value for budget constraint
    al = 0.0  # dual value for lambda constraint
    iidobj = [] #single scenario objective value
    iidgap = []
    iidfeasi = []
    bestobj = -1.0
    bestbud = -1.0
    bestsol = []
    gap = -1.0
    feasibility = False
    def __init__(self , num):
        self.scnum = num
        proary = numpy.random.choice(self.dt, self.scnum*S*T, p=self.prob)
        self.dts = numpy.reshape(proary, (S,T,self.scnum)).tolist()
        #print(self.dts)
        # calculate lambda**t
        self.par = [[[1.0 for k in range(0, self.scnum)] for j in range(0, T+1)] for i in range(0, S+1)] # parameter for q0, not lambda
        for k in range(0, self.scnum):
            for i in range(1, S+1):
                for j in range(1, T+1):
                    temp = 1.0
                    for l in range(0,j):
                        temp = temp * self.dts[i-1][l][k] # the generated deterioration matrix start with index 0
                        #print(i, j, temp)
                    self.par[i][j][k] = temp
                    #print("par ", self.par[i][j][k])
        #print("check ", self.dts[1][0][3])
        #print("check0" ,self.par[1][1])
        #print("par check", self.par[1])
        self.par1 = [[[[1.0 for k in range(0, self.scnum)]for l in range(0, T)] for j in range(0, T+1)] for i in range(0, S+1)]

        for k in range(0, self.scnum):
            for j in range(1, T+1):
                for i in s:
                    for l in range(0, j):
                        temp = 1.0
                        for m in range(0, j-l-1):
                            temp = temp * self.dts[i-1][j-m-1][k]
                        self.par1[i][j][l][k] = temp
        #print("check2", self.par1[1][2])

    def model1(self, m):
        obj = -1
        mdl = Model('single model')
        x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(mdl.sum(AADT[i] * q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l]
                                * e[k] * (self.par1[i][j][l][m]) for k in r for l in range(0, j)) for i
                                in s for j in range(0, T+1))))
        mdl.add_constraints(q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                    for k in r for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1) )
        mdl.add_constraints(q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                    for k in r for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in rs for k in t) <= g for i in s)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in r) == 1 for i in s for k in t)
        mdl.print_information()
        #mdl.export_as_lp("SAAmilp_1")
        solution = mdl.solve(log_output=False)
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            return obj
        else:
            #print("model1 solved")
            #print(solution)
            # print(self.mdl.solve_details)
            solvetime = round(mdl.solve_details.time, 3)
            obj = solution.get_objective_value()
            mdl.end()
            return obj

    def iidsolve(self):
        self.iidobj = [0.0 for i in range(0,self.scnum)]
        for i in range(0, self.scnum):
            temp = self.model1(i)
            self.iidobj[i] = temp
        #print(self.iidobj)
        print("mean " , np.mean(self.iidobj))
        print("variance ", np.var(self.iidobj) )

    class cgs():
        # run column generation for scenarios sn
        def __init__(self, sn, par, par1):
            self.__sn = sn
            self.__par = par
            self.__par1 = par1
            self.__pcl = [[[0.0 for k in range(T + 1)] for j in range(R + 1)] for i in range(S + 1)]  # primal model cost coefficient list
            self.__cll = [[[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in range(0, S + 1)]]  # column list
            self.__ldl = [0]  # solution lambda list
            self.__ptc = 0.0  #primal problem constant value
            self.__subObj = 0.0 #objective value of subproblem
            self.__masObj = 0.0 #objective value of masterproblem
            self.__w = 0.0  # dual value for budget constraint
            self.__al = 0.0  # dual value for lambda constraint

        def getco(self):
            mdl = Model('coefi model')
            #print("__sn check", self.__sn)
            #print("__par check", self.__par)
            #print("__par1 check", self.__par1)
            x = mdl.continuous_var_dict([(i, j, k) for i in s for j in r for k in t], lb=0, ub=1, name='x')
            mdl.minimize(-1*mdl.sum(mdl.sum(AADT[i] * q0[i] * (self.__par[i][j][self.__sn]) + mdl.sum(x[i, k, l]
                         * e[k] * (self.__par1[i][j][l][self.__sn]) for k in r for l in range(0, j)) for i
                        in s for j in range(0, T+1))))
            mdl.add_constraints(x[i, j, k] == 0 for i in s for j in r for k in t)
            #mdl.export_as_lp("SAA_coefficient_problem")
            for i in s:
                for j in r:
                    for k in t:
                        self.__pcl[i][j][k] = mdl.objective_coef(x[i, j, k])
            solution = mdl.solve()
            if solution is None:
                print(mdl.solve_details)
                mdl.end()
            else:
                self.__ptc = solution.get_objective_value()
                #print('ptc: ' + str(self.__ptc))
                #print("pcl", self.__pcl)
                mdl.end()

        def initiate(self,pavnum):
            nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]   # create new column by solution
            sepobj = 0.0
            fsb = False
            mdl = Model('SAA_initiate problem')
            x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
            mdl.maximize(mdl.sum(-self.__pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                        for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                        for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1))
            mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
            mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
            #mdl.export_as_lp('initiate problem_' + str(pavnum))
            solution = mdl.solve(log_output = False)
            if solution is None:
                print(mdl.solve_details)
                mdl.end()
                fsb = False
            else:
                #print('initiate problem solved')
                sepobj = solution.get_objective_value()
                for j in r:
                    for k in t:
                        if x[pavnum, j, k].solution_value > 0.99:
                            nclp[j][k] = 1.0
                        else:
                            nclp[j][k] = 0.0
                mdl.end()
                fsb = True
            return fsb,nclp, sepobj

        def runiniate(self):
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]
            tobj = 0.0
            for i in s:
                #print('=========initial problem for pavement ' + str(i) +' =============')
                fsb, nclp, sepobj = self.initiate(i)
                if fsb is True:
                    ncl[i] = nclp
                    tobj = tobj + sepobj

                else:
                    print('initial problem '+str(i)+' infeasible')
                    return False
            print('obj: ' + str(tobj))
            self.__cll.append(ncl)
            self.__subObj = tobj

        def masterp(self):
            cnum =len(self.__cll) #column number
            cn =[i for i in range(0,cnum)] #column index
            self.__ldl = [0.0 for i in range(0, cnum)]
            mdl = Model('master problem')
            ld = mdl.continuous_var_dict([i for i in cn], lb=0, ub=1, name='ld') # variables lambda
            mdl.minimize(self.__ptc+mdl.sum(self.__pcl[i][j][k]*self.__cll[m][i][j][k]*ld[m] for i in s for j in r for k in t
                                          for m in cn))
            mdl.add_constraint(mdl.sum(cost[j]*self.__cll[l][i][j][k]*ld[l] for i in s for j in r for k in t for l in cn) <= B, 'ct1')
            mdl.add_constraint(mdl.sum(ld[l] for l in cn) == 1.0, 'ct2')
            #mdl.export_as_lp('master_problem')
            solution = mdl.solve(log_output = False)

            if solution is None:
                print('master problem infeasible')
                print(mdl.solve_details)
                mdl.end()
                return False
            else:
                print('master problem solved')
                self.__w = mdl.dual_values(mdl.find_matching_linear_constraints('ct1'))[0]
                self.__al = mdl.dual_values(mdl.find_matching_linear_constraints('ct2'))[0]
                self.__masObj = solution.get_objective_value()
                print('master objective: ', self.__masObj)
                for i in range(0,cnum):
                    self.__ldl[i] = ld[i].solution_value
                    print(str(round(ld[i].solution_value, 3)), end=" ")
                print(' ')
                print('w: '+str(self.__w)+'\n'+'alpha: '+str(self.__al))
                mdl.end()
                return True

        def subp(self, pavnum):
            nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]
            subobj = 0.0
            fsb = False
            mdl = Model('sub problem')
            x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
            mdl.maximize(mdl.sum((self.__w*cost[j]-self.__pcl[pavnum][j][k]) * x[pavnum, j, k] for j in r for k in t))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                         for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1) for m in range(0, self.__sn))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                         for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1) for m in range(0, self.__sn))
            mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
            mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
            #mdl.export_as_lp('sub problem')
            solution = mdl.solve(log_output = False)

            if solution is None:
                print(mdl.solve_details)
                mdl.end()
                fsb = False
            else:
                subobj = solution.get_objective_value()
                for j in r:
                    for k in t:
                        if x[pavnum, j, k].solution_value > 0.98:
                            nclp[j][k] = 1.0
                        else:
                            nclp[j][k] = 0.0
                mdl.end()
                fsb = True
            return fsb, nclp, subobj
        def runsubproblem(self):
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]
            tempobj = self.__al
            for i in s:
                #print('=========sub problem for pavement ' + str(i) +' =============')
                fsb, nclp, subobj = self.subp(i)
                if fsb is True:
                    ncl[i] = nclp
                    tempobj = tempobj + subobj
                else:
                    print('sub  problem' + str(i) +' infeasible')
                    return False
            self.__cll.append(ncl)
            self.__subObj = tempobj
            print('sub problem solved')
            print("reduce cost: "+str(tempobj))
            return True

        def costcalculate(self, plan):
            plancost = 0.0
            for i in s:
                for j in r:
                    for k in t:
                        if plan[i][j][k] > 0.99:
                            plancost = plancost + cost[j]*1.0

            return plancost

        def objectivecalculate(self, plan):
            objectivevalue = self.__ptc
            for i in s:
                for j in r:
                    for k in t:
                        if plan[i][j][k] > 0.99:
                            objectivevalue = objectivevalue + self.__pcl[i][j][k]
            return -objectivevalue

        def result(self):
            tempbestobj = -1.0
            tempbestsol = []
            tempbestbud = -1.0
            tempbestgap = -1.0
            tempbestfeasibility = False
            #print(self.__cll)
            for i in range(1,len(self.__cll)):
                temp = self.__cll[i]
                tempcost = self.costcalculate(temp)
                tempobj = self.objectivecalculate(temp)
                #print("=== "+str(i)+" ====")
                #print(tempcost)
                #print(tempobj)
                if tempcost < B and tempobj>tempbestobj:
                    tempbestsol = temp
                    tempbestobj = tempobj
                    tempbestbud = tempcost
                    tempbestfeasibility = True
            #print(tempbestobj)
            tempbestgap = -(tempbestobj-self.__masObj)/self.__masObj
            #print('result obj', tempbestobj)
            return tempbestobj, tempbestbud, tempbestgap, tempbestfeasibility


        def runcg(self):
            self.getco()
            now = time.time() #calculate time at the beginning of iteration
            if self.runiniate() == False:
                filen1 = open(address, "a")
                filen1.write(" initiate infeasible")

            else:
                itr = 0
                print('sub obj: '+ str(self.__subObj))
                while self.__subObj > 0.1 and itr <50:
                    print('=========itration '+str(itr)+'===========')
                    if not self.masterp():
                        print(" masterproblem infeasible at iteration "+str(itr))
                        with open(address, "a") as filen1:
                            filen1.write(" masterproblem infeasible at iteration "+str(itr))
                    if not self.runsubproblem():
                        #if not self.sub():
                        print(address+str(itr))
                        with open(address, "a") as filen1:
                            filen1.write(" subproblem infeasible at iteration "+str(itr))
                    #print("reduce cost: "+ str(self.subobj))
                    itr += 1
                print("=====end of iteration=====")
                later = time.time() #stop calculation at the end of iteration
                self.__solvetime = int(later - now)
                self.__itrnum = itr

    def iidcg(self):
        iidobj = [-1.0 for i in range(0, self.scnum)]
        iidbud = [-1.0 for i in range(0, self.scnum)]
        iidgap = [-1.0 for i in range(0, self.scnum)]
        iidfeasi = [False for i in range(0, self.scnum)]
        for i in range(0, self.scnum):
            #print("scnum check",i)
            temp = self.cgs(i, self.par, self.par1)
            temp.runcg()
            tempobj, tempbud, tempgap, tempfeasi = temp.result()

            iidobj[i] = tempobj
            iidbud[i] = tempbud
            iidgap[i] = tempgap
            iidfeasi[i]= tempfeasi
        #print("feasibility", self.feasibility)
        #print("objset ", self.iidobj)
        d = {'obj':iidobj, 'budgut':iidbud, 'gap': iidgap, 'feasibility':iidfeasi}
        df = pd.DataFrame(d)
        df.loc['mean','obj'] = np.mean(iidobj)
        df.loc['variance','obj'] = math.sqrt(np.var(iidobj))
        #print(df)
        addr = './result/SAA_' + str(S) + '_' + unit + '_' + str(T) + 'y_' + str(self.scnum) + 'snr.csv'
        df.to_csv(addr)

    def model2(self):
        # objective x variable index (-1) since for t = t, the qt = q0 + sum(x0...xt-1)
        self.mdl = Model('PMS')
        self.x = self.mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        self.mdl.maximize(self.mdl.sum(self.mdl.sum(AADT[i] * q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l]
                                                    * e[k] * (self.par1[i][j][l][m]) for k in r for l in range(0, j)) for i
                                                    in s for j in range(0, T+1) )/self.scnum for m in range(0, self.scnum)))
        self.mdl.add_constraints(q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                for k in r for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1) for m in range(0, self.scnum))
        self.mdl.add_constraints(q0[i] * (self.par[i][j][m]) + self.mdl.sum(self.x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                for k in r for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1) for m in range(0, self.scnum))
        self.mdl.add_constraint(self.mdl.sum(self.x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        # mdl.add_constraints(q[i,k] >= Qm for i in s for k in t)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs for k in t) <= g for i in s)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in r) == 1 for i in s for k in t)
        #self.mdl.print_information()
        #self.mdl.export_as_lp("SAAmilp_2")

    def directsolve(self):
        solution = self.mdl.solve(log_output=False)
        if solution is None:
            print(self.mdl.solve_details)
            self.mdl.end()
        else:
            print("model solved")
            #print(solution)
            # print(self.mdl.solve_details)
            self.solvetime = round(self.mdl.solve_details.time, 3)
            self.obj_value = solution.get_objective_value()
            print("model objective value: " + str(self.obj_value))
            for i in s:
                for j in r:
                    for k in t:
                        if self.x[(i, j, k)].solution_value > 0.9:
                            self.totalcost = self.totalcost + cost[j]
            # print(str(self.totalcost))

    def runmodel2(self):
        self.model2()
        self.directsolve()
        self.end()

    def end(self):
        self.mdl.end()

    def getcoe(self):
        mdl = Model('coefi model')
        x = mdl.continuous_var_dict([(i, j, k) for i in s for j in r for k in t], lb=0, ub=1, name='x')
        mdl.minimize(-1*mdl.sum(mdl.sum(AADT[i] * q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l]
                                                * e[k] * (self.par1[i][j][l][m]) for k in r for l in range(0, j)) for i
                                               in s for j in range(0, T+1) )/self.scnum for m in range(0, self.scnum)))
        mdl.add_constraints(x[i, j, k] == 0 for i in s for j in r for k in t)
        #mdl.export_as_lp("SAA_coefficient_problem")
        for i in s:
            for j in r:
                for k in t:
                    self.pcl[i][j][k] = mdl.objective_coef(x[i, j, k])
        solution = mdl.solve()
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
        else:
            self.ptc = solution.get_objective_value()
            print('ptc: ' + str(self.ptc))
            #print("pcl", self.pcl)
            mdl.end()

    def initiate(self, pavnum):
        #Initiate problem for single pavement
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]   # create new column by solution
        sepobj = 0.0
        fsb = False
        mdl = Model('SAA_initiate problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(-self.pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum] * (self.par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.par1[pavnum][j][l][m])
                for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraints(q0[pavnum] * (self.par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.par1[pavnum][j][l][m])
                for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('initiate problem_' + str(pavnum))
        solution = mdl.solve(log_output = False)
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            #print('initiate problem solved')
            sepobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.99:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb,nclp, sepobj

    def runiniate(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tobj = 0.0
        for i in s:
            #print('=========initial problem for pavement ' + str(i) +' =============')
            fsb, nclp, sepobj = self.initiate(i)
            if fsb is True:
                ncl[i] = nclp
                tobj = tobj + sepobj

            else:
                print('initial problem '+str(i)+' infeasible')
                return False
        print('obj: ' + str(tobj))
        self.cll.append(ncl)
        self.subObj = tobj

    def masterp(self):
        cnum =len(self.cll) #column number
        cn =[i for i in range(0,cnum)] #column index
        self.ldl = [0.0 for i in range(0, cnum)]
        mdl = Model('master problem')
        ld = mdl.continuous_var_dict([i for i in cn], lb=0, ub=1, name='ld') # variables lambda
        mdl.minimize(self.ptc+mdl.sum(self.pcl[i][j][k]*self.cll[m][i][j][k]*ld[m] for i in s for j in r for k in t
                                      for m in cn))
        mdl.add_constraint(mdl.sum(cost[j]*self.cll[l][i][j][k]*ld[l] for i in s for j in r for k in t for l in cn) <= B, 'ct1')
        mdl.add_constraint(mdl.sum(ld[l] for l in cn) == 1.0, 'ct2')
        #mdl.export_as_lp('master_problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print('master problem infeasible')
            print(mdl.solve_details)
            mdl.end()
            return False
        else:
            print('master problem solved')
            self.w = mdl.dual_values(mdl.find_matching_linear_constraints('ct1'))[0]
            self.al = mdl.dual_values(mdl.find_matching_linear_constraints('ct2'))[0]
            self.masObj = solution.get_objective_value()
            print('master objective: ' ,self.masObj)
            for i in range(0,cnum):
                self.ldl[i] = ld[i].solution_value
                print(str(round(ld[i].solution_value,3)), end = " ")
            print(' ')
            print('w: '+str(self.w)+'\n'+'alpha: '+str(self.al))
            mdl.end()
            return True

    def sub(self):
        mdl = Model('sub problem')
        x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        mdl.maximize(mdl.sum((self.w*cost[j]-self.pcl[i][j][k]) * x[i, j, k] for i in s for j in r for k in t)+self.al)
        mdl.add_constraints(q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                            for k in r for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraints(q0[i] * (self.par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (self.par1[i][j][l][m])
                            for k in r for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in rs for k in t) <= g for i in s)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in r) == 1 for i in s for k in t)
        #mdl.export_as_lp("subproblem")
        solution = mdl.solve(log_output=False)

        if solution is None:
            print('sub problem infeasible')
            print(mdl.solve_details)
            mdl.end()
            return False
        else:
            print('sub problem solved')
            #print(solution)
            self.subObj = solution.get_objective_value()
            print("reduce cost: "+str(self.subObj))
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]  # create new column based on solution
            for i in s:
                for j in r:
                    for k in t:
                        if x[i, j, k].solution_value > 0.98:
                            ncl[i][j][k] = 1.0
                        else:
                            ncl[i][j][k] = 0.0
            #print(ncl[1])
            self.cll.append(ncl)  # add new column to column list
            #print(len(self.cll))
            mdl.end()
            return True

    def subp(self, pavnum):
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]
        subobj = 0.0
        fsb = False
        mdl = Model('sub problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum((self.w*cost[j]-self.pcl[pavnum][j][k]) * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum] * (self.par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.par1[pavnum][j][l][m])
            for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraints(q0[pavnum] * (self.par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.par1[pavnum][j][l][m])
            for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1) for m in range(0, self.scnum))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('sub problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            subobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.98:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb, nclp, subobj

    def runsubproblem(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tempobj = self.al
        for i in s:
            #print('=========sub problem for pavement ' + str(i) +' =============')
            fsb, nclp, subobj = self.subp(i)
            if fsb is True:
                ncl[i] = nclp
                tempobj = tempobj + subobj
            else:
                print('sub  problem' + str(i) +' infeasible')
                return False
        self.cll.append(ncl)
        self.subObj = tempobj
        print('sub problem solved:', self.subObj )
        print("reduce cost: "+str(tempobj))
        return True

    def costcalculate(self, plan):
        plancost = 0.0
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        plancost = plancost + cost[j]*1.0

        return plancost

    def objectivecalculate(self, plan):
        objectivevalue = self.ptc
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        objectivevalue = objectivevalue + self.pcl[i][j][k]
        return -objectivevalue

    def result(self):
        tempbestobj = 0.0
        tempbestsol = []
        tempbestbud = 0
        tempfeasibility = False
        for i in range(1,len(self.cll)):
            temp = self.cll[i]
            tempcost = self.costcalculate(temp)
            tempobj = self.objectivecalculate(temp)
            #print("=== "+str(i)+" ====")
            #print(tempcost)
            #print(tempobj)
            #print(temp)
            if tempcost < B and tempobj>tempbestobj:
                tempbestsol = temp
                tempbestobj = self.objectivecalculate(temp)
                tempbestbud = self.costcalculate(temp)
                tempfeasibility = True
        #print(tempbestobj)
        self.bestsol = tempbestsol
        self.bestobj = tempbestobj
        self.bestbud = tempbestbud
        self.gap = -(-self.masObj-self.bestobj)/self.masObj
        #bestobj 6977 mastobj -7042 (hint)
        #print(self.bestsol)
        print("bestobj: ", self.bestobj, "masterobj: ", -self.masObj, "gap:", self.gap)
        self.feasibility = tempfeasibility

    def getresult(self)-> list:
        return self.bestsol

    def runcg(self):
        print("===start to run cg====")
        self.getcoe()
        now = time.time() #calculate time at the beginning of iteration
        if self.runiniate() == False:
            print("initiate failed")
            with open(address, "a") as file1:
                file1.write(" initiate infeasible")
        else:
            itr = 0
            print('sub obj: '+ str(self.subObj))
            while self.subObj > 0.1 and itr <50:
                print('=========itration '+str(itr)+'===========')
                if not self.masterp():
                    print(" masterproblem infeasible at iteration "+str(itr))
                    file1 = open(address, "a")
                    file1.write(" masterproblem infeasible at iteration "+str(itr))
                    file1.close()
                    break
                if not self.runsubproblem():
                    #if not self.sub():
                    print(address+str(itr))
                    file1 = open(address, "a")
                    file1.write(" subproblem infeasible at iteration "+str(itr))
                    file1.close()
                    break
                #print("reduce cost: "+ str(self.subobj))
                itr += 1
            print("=====end of iteration=====")
            later = time.time() #stop calculation at the end of iteration
            self.solvetime = int(later - now)
            self.itrnum = itr
            self.result()

    def recordSAA(self):
        adres = './result/SAA_' + str(S) + '_' + unit + '_' + str(T) + 'y_' + str(self.scnum) + 'snr.csv'
        df = pd.read_csv(adres)
        df.loc[len(df.index)] = ['SAA', self.bestobj, self.bestbud, self.gap, self.feasibility]
        df.to_csv(adres)
    def test(self):
        print(self.bestobj)
    def output(self):
        #print(self.ldl)
        #file1 = open("output_"+sys.argv[3]+".txt", "a")
        with open(address, 'a') as file1:
            #file1 = open(address, "a")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("=============CG Scenario solution=============")
            file1.write(" \n")
            file1.write(" \n")
            file1.write("Solving time: "+str(self.solvetime)+"s")
            file1.write(" \n")
            file1.write("iteration: "+str(self.itrnum))
            file1.write(" \n")
            file1.write("Feasibility: "+str(self.feasibility))
            file1.write(" \n")
            file1.write(" \n")
            file1.write("objective value: "+str(self.masObj))
            file1.write(" \n")
            file1.write(" \n")
            for i in range(0,len(self.ldl)):
                if self.ldl[i] >0.001:
                    print(self.ldl[i])
                    file1.write("lambda("+str(i)+"): "+str(self.ldl[i]))
                    file1.write(" \n")
                    tempobjective = self.objectivecalculate(self.cll[i])
                    print(tempobjective)
                    print('---------------')
                    file1.write("Objective value: "+str(tempobjective))
                    file1.write(" \n")
                    tempcost = self.costcalculate(self.cll[i])
                    print(tempcost)
                    print('---------------')
                    file1.write("Total cost: "+str(tempcost))
                    file1.write(" \n")
                    file1.write(" \n")
            file1.write("Objective value of best solution: "+str(self.bestobj))
            file1.write(" \n")
            file1.write("Budget of best solution: "+str(self.bestbud))
            file1.write(" \n")
            file1.write("Gap: "+ str(round(self.gap*100,3))+"%")
            file1.write(" \n")
            file1.write(" \n")
            file1.close()



# =======================this is the function of model================

def runmilp():
    a = milp()
    a.runmilp()
def runcg():
    a = cg()
    a.runcg()
    a.output()

#========================================================================
def runiidSAA():
    ex = SAAmodel(scenario)
    ex.iidcg() # run iid model and save to csv file
    ex.runcg() # run SAA model
    ex.recordSAA() #save to csv file

#===============run SAA with n scenarios m times=========================
def runSAANM(n, m):
    objlist = [-1.0 for i in range(0, m)]
    budlist = [-1.0 for i in range(0, m)]
    gaplist = [-1.0 for i in range(0, m)]
    feasiblelist = [False for i in range(0, m)]

    for i in range(0,m):
        tmp = SAAmodel(n)
        tmp.runcg()
        objlist[i] = tmp.bestobj
        budlist[i] = tmp.bestbud
        gaplist[i] = tmp.gap
        feasiblelist[i] = tmp.feasibility
    d = {'obj' : objlist, 'bud' : budlist, 'gap' : gaplist, 'feasibility' : feasiblelist}
    df = pd.DataFrame(d)
    df.to_csv('./result/SAA_' + str(S) + '_' + unit + '_' + str(T) + 'y_' + str(n) + 'snr_'+str(m)+'times.csv')

def checifesi(sol: list, case: int): # sol solution to check, case check case number
    prob = [0.5, 0.5]; dt = [0.9, 0.8]
    ra = scenario + case # this number used to generate more random variable, while we only choose one of array
    proary = numpy.random.choice(dt, S*T*ra, p=prob)
    dts = numpy.reshape(proary, (S,T,ra)).tolist()
    #print(dts.dtype)
    #print(np.shape(dts))
    #print(dts[:][:][10])
    par = [[[1.0 for k in range(0, case)] for j in range(0, T+1)] for i in range(0, S+1)] # parameter for q0, not lambda
    #print(np.shape(par))
    for k in range(0, case):
        for i in range(1, S+1):
            for j in range(1, T+1):
                temp = 1.0
                for l in range(0,j):
                    temp = temp * dts[i-1][l][k+scenario]# the generated deterioration matrix start with index 0
                    #print(i, j, temp)
                par[i][j][k] = temp
    #print(par[1][5][4])
    par1 = [[[[1.0 for k in range(0, case)] for l in range(0, T)] for j in range(0, T+1)] for i in range(0, S+1)]

    for k in range(0, case):
        for j in range(1, T+1):
            for i in s:
                for l in range(0, j):
                    temp = 1.0
                    for m in range(0, j-l-1):
                        temp = temp * dts[i-1][j-m-1][k+scenario]
                    par1[i][j][l][k] = temp

    def runfeasi(m): # run model for scenario m to check feasibility and its corresponding objective value
        obj = -1
        mdl = Model('single model')
        x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(mdl.sum(AADT[i] * q0[i] * (par[i][j][m]) + mdl.sum(x[i, k, l]
                                                                                     * e[k] * (par1[i][j][l][m]) for k in r for l in range(0, j)) for i
                                     in s for j in range(0, T+1))))
        mdl.add_constraints(q0[i] * (par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (par1[i][j][l][m])
                                                                  for k in r for l in range(0, j)) >= Qmin for i in s for j in range(1, T+1) )
        mdl.add_constraints(q0[i] * (par[i][j][m]) + mdl.sum(x[i, k, l] * e[k] * (par1[i][j][l][m])
                                                                  for k in r for l in range(0, j)) <= Qmax for i in s for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[i, j, k] * cost[j] for i in s for j in r for k in t) <= B)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in rs for k in t) <= g for i in s)
        mdl.add_constraints(mdl.sum(x[i, j, k] for j in r) == 1 for i in s for k in t)
        mdl.add_constraints(x[i,j,k] == sol[i][j][k] for i in s for j in r for k in t)
        #mdl.print_information()
        #mdl.export_as_lp("SAAmilp_1")
        solution = mdl.solve(log_output=False)
        if solution is None:
            #print(mdl.solve_details)
            print("infeasible")
            mdl.end()
            return -1, False
        else:
            #print("model1 solved")
            #print(solution)
            # print(self.mdl.solve_details)
            #solvetime = round(mdl.solve_details.time, 3)
            obj = solution.get_objective_value()
            mdl.end()
            print("feasible with obj:", str(obj))
            return obj, True

    def getco(self):
        mdl = Model('coefi model')
        #print("__sn check", self.__sn)
        #print("__par check", self.__par)
        #print("__par1 check", self.__par1)
        x = mdl.continuous_var_dict([(i, j, k) for i in s for j in r for k in t], lb=0, ub=1, name='x')
        mdl.minimize(-1*mdl.sum(mdl.sum(AADT[i] * q0[i] * (self.__par[i][j][self.__sn]) + mdl.sum(x[i, k, l]
                                                                                                  * e[k] * (self.__par1[i][j][l][self.__sn]) for k in r for l in range(0, j)) for i
                                        in s for j in range(0, T+1))))
        mdl.add_constraints(x[i, j, k] == 0 for i in s for j in r for k in t)
        #mdl.export_as_lp("SAA_coefficient_problem")
        for i in s:
            for j in r:
                for k in t:
                    self.__pcl[i][j][k] = mdl.objective_coef(x[i, j, k])
        solution = mdl.solve()
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
        else:
            self.__ptc = solution.get_objective_value()
            #print('ptc: ' + str(self.__ptc))
            #print("pcl", self.__pcl)
            mdl.end()

    def initiate(self,pavnum):
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]   # create new column by solution
        sepobj = 0.0
        fsb = False
        mdl = Model('SAA_initiate problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(-self.__pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                                                                      for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1))
        mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                                                                      for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('initiate problem_' + str(pavnum))
        solution = mdl.solve(log_output = False)
        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            #print('initiate problem solved')
            sepobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.99:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb,nclp, sepobj

    def runiniate(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tobj = 0.0
        for i in s:
            #print('=========initial problem for pavement ' + str(i) +' =============')
            fsb, nclp, sepobj = self.initiate(i)
            if fsb is True:
                ncl[i] = nclp
                tobj = tobj + sepobj

            else:
                print('initial problem '+str(i)+' infeasible')
                return False
        print('obj: ' + str(tobj))
        self.__cll.append(ncl)
        self.__subObj = tobj

    def masterp(self):
        cnum =len(self.__cll) #column number
        cn =[i for i in range(0,cnum)] #column index
        self.__ldl = [0.0 for i in range(0, cnum)]
        mdl = Model('master problem')
        ld = mdl.continuous_var_dict([i for i in cn], lb=0, ub=1, name='ld') # variables lambda
        mdl.minimize(self.__ptc+mdl.sum(self.__pcl[i][j][k]*self.__cll[m][i][j][k]*ld[m] for i in s for j in r for k in t
                                        for m in cn))
        mdl.add_constraint(mdl.sum(cost[j]*self.__cll[l][i][j][k]*ld[l] for i in s for j in r for k in t for l in cn) <= B, 'ct1')
        mdl.add_constraint(mdl.sum(ld[l] for l in cn) == 1.0, 'ct2')
        #mdl.export_as_lp('master_problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print('master problem infeasible')
            print(mdl.solve_details)
            mdl.end()
            return False
        else:
            print('master problem solved')
            self.__w = mdl.dual_values(mdl.find_matching_linear_constraints('ct1'))[0]
            self.__al = mdl.dual_values(mdl.find_matching_linear_constraints('ct2'))[0]
            self.__masObj = solution.get_objective_value()
            print('master objective: ', self.__masObj)
            for i in range(0,cnum):
                self.__ldl[i] = ld[i].solution_value
                print(str(round(ld[i].solution_value, 3)), end=" ")
            print(' ')
            print('w: '+str(self.__w)+'\n'+'alpha: '+str(self.__al))
            mdl.end()
            return True

    def subp(self, pavnum):
        nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]
        subobj = 0.0
        fsb = False
        mdl = Model('sub problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum((self.__w*cost[j]-self.__pcl[pavnum][j][k]) * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                                                              for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1) for m in range(0, self.__sn))
        mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                                                              for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1) for m in range(0, self.__sn))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        #mdl.export_as_lp('sub problem')
        solution = mdl.solve(log_output = False)

        if solution is None:
            print(mdl.solve_details)
            mdl.end()
            fsb = False
        else:
            subobj = solution.get_objective_value()
            for j in r:
                for k in t:
                    if x[pavnum, j, k].solution_value > 0.98:
                        nclp[j][k] = 1.0
                    else:
                        nclp[j][k] = 0.0
            mdl.end()
            fsb = True
        return fsb, nclp, subobj
    def runsubproblem(self):
        ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
               range(0, S + 1)]
        tempobj = self.__al
        for i in s:
            #print('=========sub problem for pavement ' + str(i) +' =============')
            fsb, nclp, subobj = self.subp(i)
            if fsb is True:
                ncl[i] = nclp
                tempobj = tempobj + subobj
            else:
                print('sub  problem' + str(i) +' infeasible')
                return False
        self.__cll.append(ncl)
        self.__subObj = tempobj
        print('sub problem solved')
        print("reduce cost: "+str(tempobj))
        return True

    def costcalculate(self, plan):
        plancost = 0.0
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        plancost = plancost + cost[j]*1.0

        return plancost

    def objectivecalculate(self, plan):
        objectivevalue = self.__ptc
        for i in s:
            for j in r:
                for k in t:
                    if plan[i][j][k] > 0.99:
                        objectivevalue = objectivevalue + self.__pcl[i][j][k]
        return -objectivevalue

    def result(self):
        tempbestobj = -1.0
        tempbestsol = []
        tempbestbud = -1.0
        tempbestgap = -1.0
        tempbestfeasibility = False
        #print(self.__cll)
        for i in range(1,len(self.__cll)):
            temp = self.__cll[i]
            tempcost = self.costcalculate(temp)
            tempobj = self.objectivecalculate(temp)
            #print("=== "+str(i)+" ====")
            #print(tempcost)
            #print(tempobj)
            if tempcost < B and tempobj>tempbestobj:
                tempbestsol = temp
                tempbestobj = tempobj
                tempbestbud = tempcost
                tempbestfeasibility = True
        #print(tempbestobj)
        tempbestgap = -(tempbestobj-self.__masObj)/self.__masObj
        #print('result obj', tempbestobj)
        return tempbestobj, tempbestbud, tempbestgap, tempbestfeasibility


    def runcg(self):
        self.getco()
        now = time.time() #calculate time at the beginning of iteration
        if self.runiniate() == False:
            filen1 = open(address, "a")
            filen1.write(" initiate infeasible")

        else:
            itr = 0
            print('sub obj: '+ str(self.__subObj))
            while self.__subObj > 0.1 and itr <50:
                print('=========itration '+str(itr)+'===========')
                if not self.masterp():
                    print(" masterproblem infeasible at iteration "+str(itr))
                    with open(address, "a") as filen1:
                        filen1.write(" masterproblem infeasible at iteration "+str(itr))
                if not self.runsubproblem():
                    #if not self.sub():
                    print(address+str(itr))
                    with open(address, "a") as filen1:
                        filen1.write(" subproblem infeasible at iteration "+str(itr))
                #print("reduce cost: "+ str(self.subobj))
                itr += 1
            print("=====end of iteration=====")
            later = time.time() #stop calculation at the end of iteration
            self.__solvetime = int(later - now)
            self.__itrnum = itr
    class cgs():
        # run column generation for scenarios sn
        def __init__(self, sn, par, par1):
            self.__sn = sn
            self.__par = par
            self.__par1 = par1
            self.__pcl = [[[0.0 for k in range(T + 1)] for j in range(R + 1)] for i in range(S + 1)]  # primal model cost coefficient list
            self.__cll = [[[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in range(0, S + 1)]]  # column list
            self.__ldl = [0]  # solution lambda list
            self.__ptc = 0.0  #primal problem constant value
            self.__subObj = 0.0 #objective value of subproblem
            self.__masObj = 0.0 #objective value of masterproblem
            self.__w = 0.0  # dual value for budget constraint
            self.__al = 0.0  # dual value for lambda constraint

        def getco(self):
            mdl = Model('coefi model')
            #print("__sn check", self.__sn)
            #print("__par check", self.__par)
            #print("__par1 check", self.__par1)
            x = mdl.continuous_var_dict([(i, j, k) for i in s for j in r for k in t], lb=0, ub=1, name='x')
            mdl.minimize(-1*mdl.sum(mdl.sum(AADT[i] * q0[i] * (self.__par[i][j][self.__sn]) + mdl.sum(x[i, k, l]
                                            * e[k] * (self.__par1[i][j][l][self.__sn]) for k in r for l in range(0, j)) for i
                                            in s for j in range(0, T+1))))
            mdl.add_constraints(x[i, j, k] == 0 for i in s for j in r for k in t)
            #mdl.export_as_lp("SAA_coefficient_problem")
            for i in s:
                for j in r:
                    for k in t:
                        self.__pcl[i][j][k] = mdl.objective_coef(x[i, j, k])
            solution = mdl.solve()
            if solution is None:
                print(mdl.solve_details)
                mdl.end()
            else:
                self.__ptc = solution.get_objective_value()
                #print('ptc: ' + str(self.__ptc))
                #print("pcl", self.__pcl)
                mdl.end()

        def initiate(self,pavnum):
            nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]   # create new column by solution
            sepobj = 0.0
            fsb = False
            mdl = Model('SAA_initiate problem')
            x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
            mdl.maximize(mdl.sum(-self.__pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                                                                          for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][self.__sn]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][self.__sn])
                                                                                          for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1))
            mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
            mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
            #mdl.export_as_lp('initiate problem_' + str(pavnum))
            solution = mdl.solve(log_output = False)
            if solution is None:
                print(mdl.solve_details)
                mdl.end()
                fsb = False
            else:
                #print('initiate problem solved')
                sepobj = solution.get_objective_value()
                for j in r:
                    for k in t:
                        if x[pavnum, j, k].solution_value > 0.99:
                            nclp[j][k] = 1.0
                        else:
                            nclp[j][k] = 0.0
                mdl.end()
                fsb = True
            return fsb,nclp, sepobj

        def runiniate(self):
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]
            tobj = 0.0
            for i in s:
                #print('=========initial problem for pavement ' + str(i) +' =============')
                fsb, nclp, sepobj = self.initiate(i)
                if fsb is True:
                    ncl[i] = nclp
                    tobj = tobj + sepobj

                else:
                    print('initial problem '+str(i)+' infeasible')
                    return False
            print('obj: ' + str(tobj))
            self.__cll.append(ncl)
            self.__subObj = tobj

        def masterp(self):
            cnum =len(self.__cll) #column number
            cn =[i for i in range(0,cnum)] #column index
            self.__ldl = [0.0 for i in range(0, cnum)]
            mdl = Model('master problem')
            ld = mdl.continuous_var_dict([i for i in cn], lb=0, ub=1, name='ld') # variables lambda
            mdl.minimize(self.__ptc+mdl.sum(self.__pcl[i][j][k]*self.__cll[m][i][j][k]*ld[m] for i in s for j in r for k in t
                                            for m in cn))
            mdl.add_constraint(mdl.sum(cost[j]*self.__cll[l][i][j][k]*ld[l] for i in s for j in r for k in t for l in cn) <= B, 'ct1')
            mdl.add_constraint(mdl.sum(ld[l] for l in cn) == 1.0, 'ct2')
            #mdl.export_as_lp('master_problem')
            solution = mdl.solve(log_output = False)

            if solution is None:
                print('master problem infeasible')
                print(mdl.solve_details)
                mdl.end()
                return False
            else:
                print('master problem solved')
                self.__w = mdl.dual_values(mdl.find_matching_linear_constraints('ct1'))[0]
                self.__al = mdl.dual_values(mdl.find_matching_linear_constraints('ct2'))[0]
                self.__masObj = solution.get_objective_value()
                print('master objective: ', self.__masObj)
                for i in range(0,cnum):
                    self.__ldl[i] = ld[i].solution_value
                    print(str(round(ld[i].solution_value, 3)), end=" ")
                print(' ')
                print('w: '+str(self.__w)+'\n'+'alpha: '+str(self.__al))
                mdl.end()
                return True

        def subp(self, pavnum):
            nclp = [[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)]
            subobj = 0.0
            fsb = False
            mdl = Model('sub problem')
            x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
            mdl.maximize(mdl.sum((self.__w*cost[j]-self.__pcl[pavnum][j][k]) * x[pavnum, j, k] for j in r for k in t))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                                                                  for k in r for l in range(0, j)) >= Qmin for j in range(1, T+1) for m in range(0, self.__sn))
            mdl.add_constraints(q0[pavnum] * (self.__par[pavnum][j][m]) + mdl.sum(x[pavnum, k, l] * e[k] * (self.__par1[pavnum][j][l][m])
                                                                                  for k in r for l in range(0, j)) <= Qmax for j in range(1, T+1) for m in range(0, self.__sn))
            mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
            mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
            #mdl.export_as_lp('sub problem')
            solution = mdl.solve(log_output = False)

            if solution is None:
                print(mdl.solve_details)
                mdl.end()
                fsb = False
            else:
                subobj = solution.get_objective_value()
                for j in r:
                    for k in t:
                        if x[pavnum, j, k].solution_value > 0.98:
                            nclp[j][k] = 1.0
                        else:
                            nclp[j][k] = 0.0
                mdl.end()
                fsb = True
            return fsb, nclp, subobj
        def runsubproblem(self):
            ncl = [[[0.0 for k in range(0, T + 1)] for j in range(0, R + 1)] for i in
                   range(0, S + 1)]
            tempobj = self.__al
            for i in s:
                #print('=========sub problem for pavement ' + str(i) +' =============')
                fsb, nclp, subobj = self.subp(i)
                if fsb is True:
                    ncl[i] = nclp
                    tempobj = tempobj + subobj
                else:
                    print('sub  problem' + str(i) +' infeasible')
                    return False
            self.__cll.append(ncl)
            self.__subObj = tempobj
            print('sub problem solved')
            print("reduce cost: "+str(tempobj))
            return True

        def costcalculate(self, plan):
            plancost = 0.0
            for i in s:
                for j in r:
                    for k in t:
                        if plan[i][j][k] > 0.99:
                            plancost = plancost + cost[j]*1.0

            return plancost

        def objectivecalculate(self, plan):
            objectivevalue = self.__ptc
            for i in s:
                for j in r:
                    for k in t:
                        if plan[i][j][k] > 0.99:
                            objectivevalue = objectivevalue + self.__pcl[i][j][k]
            return -objectivevalue

        def result(self):
            tempbestobj = -1.0
            tempbestsol = []
            tempbestbud = -1.0
            tempbestgap = -1.0
            tempbestfeasibility = False
            #print(self.__cll)
            for i in range(1,len(self.__cll)):
                temp = self.__cll[i]
                tempcost = self.costcalculate(temp)
                tempobj = self.objectivecalculate(temp)
                #print("=== "+str(i)+" ====")
                #print(tempcost)
                #print(tempobj)
                if tempcost < B and tempobj>tempbestobj:
                    tempbestsol = temp
                    tempbestobj = tempobj
                    tempbestbud = tempcost
                    tempbestfeasibility = True
            #print(tempbestobj)
            tempbestgap = -(tempbestobj-self.__masObj)/self.__masObj
            #print('result obj', tempbestobj)
            return tempbestobj, tempbestbud, tempbestgap, tempbestfeasibility


        def runcg(self):
            self.getco()
            now = time.time() #calculate time at the beginning of iteration
            if self.runiniate() == False:
                filen1 = open(address, "a")
                filen1.write(" initiate infeasible")

            else:
                itr = 0
                print('sub obj: '+ str(self.__subObj))
                while self.__subObj > 0.1 and itr <50:
                    print('=========itration '+str(itr)+'===========')
                    if not self.masterp():
                        print(" masterproblem infeasible at iteration "+str(itr))
                        with open(address, "a") as filen1:
                            filen1.write(" masterproblem infeasible at iteration "+str(itr))
                    if not self.runsubproblem():
                        #if not self.sub():
                        print(address+str(itr))
                        with open(address, "a") as filen1:
                            filen1.write(" subproblem infeasible at iteration "+str(itr))
                    #print("reduce cost: "+ str(self.subobj))
                    itr += 1
                print("=====end of iteration=====")
                later = time.time() #stop calculation at the end of iteration
                self.__solvetime = int(later - now)
                self.__itrnum = itr
    saaobj = [-1.0 for i in range(0, case)]
    saafeasi = [False for i in range(0, case)]
    scobj = [-1.0 for i in range(0, case)]

    for i in range(0, case):
        saaobj[i], saafeasi[i] =  runfeasi(i)
        #sc = cgs(i, par, par1)
        #sc.runcg()
        #scobj[i], tempbud, tempgap, tempfeasi = sc.result()
    d = {'saafeasi':saafeasi}
    #d = {'saafeasi':saafeasi, 'saaobj':saaobj, 'scobj': scobj}
    df = pd.DataFrame(d)
    #print(df)
    #addr = './result/SAAANDSC_' + str(S) + '_' + unit + '_' + str(T) + 'y_' + str(scenario) + 'snr.csv'
    addr = './result/FESASIBLECHECK_' + str(S) + '_' + unit + '_' + str(T) + 'y_' + str(scenario) + 'snr.csv'
    df.to_csv(addr)




exm = milp()
exm.runmodel1()

#runSAANM(scenario,casenum)
#runcg()
#tmp = SAAmodel(scenario)
#tmp.runcg()
#sol = tmp.getresult()
#print(sol)
#tmp.output()
#checifesi(sol, casecheck)