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
#S = int(sys.argv[1])
#B = float(sys.argv[2])  # budget
#T = int(sys.argv[3])
#g = int(sys.argv[4])
#scenario = int(sys.argv[5])
#casenum = int(sys.argv[6])
S = 5
R = 4
T = 10
B = 2000000
g = 4
scenario = 1
casenum = 1
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
address = './result/seasonalplan_S' + str(S) + '_B' + unit + '_T' + str(T) + '_G' + str(g) + '.txt'
T = T*4
t = [i for i in range(0, T + 1)]
yeart = [n for n in t if n % 4 != 1] #season that is not the first can not be assigned treatment for yealy plan
#print(yeart)
dr = math.sqrt(math.sqrt(dr))
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
    file1.write(" \n")
    file1.write("deterioration rate per season: {}".format(dr))

class milp:
    mdl = Model('PMS')
    x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
    q = mdl.continuous_var_dict([(i, j) for i in s for j in t], lb=Qmin, ub=Qmax, name='q')
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

    def solve(self):
        # self.mdl.time_limit = 72000
        solution = self.mdl.solve(log_output=True)
        if solution is None:
            print(self.mdl.solve_details)
            self.mdl.end()
        else:
            print("milp model solved")
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
        # self.plot()
        self.output()
        self.tocsv()
        self.end()

    def tocsv(self):
        tp = [[0.0 for j in range(0, T+1)] for i in range(0, S+1)]
        for i in s:
            for j in t:
                tp[i][j] = self.q[(i, j)].solution_value
        df = pd.DataFrame(tp)
        df.to_csv('./result/seas.csv')

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


class yearmilp:
    mdl = Model('PMS')
    x = mdl.binary_var_dict([(i, j, k) for i in s for j in r for k in t], name='x')
    q = mdl.continuous_var_dict([(i, j) for i in s for j in t], lb=Qmin, ub=Qmax, name='q')
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
        self.mdl.add_constraints(self.mdl.sum(self.x[i, j, k] for j in rs) == 0 for i in s for k in yeart)
        # Initial state
        self.mdl.add_constraints(self.q[i, 0] == q0[i] for i in s)
        self.mdl.print_information()
        #self.mdl.export_as_lp("milp_1")

    def solve(self):
        # self.mdl.time_limit = 72000
        solution = self.mdl.solve(log_output=True)
        if solution is None:
            print(self.mdl.solve_details)
            self.mdl.end()
        else:
            print("milp model solved")
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
        # self.plot()
        self.output()
        self.tocsv()
        self.end()


    def tocsv(self):
        tp = [[0.0 for j in range(0, T+1)] for i in range(0, S+1)]
        for i in s:
            for j in t:
                tp[i][j] = self.q[(i, j)].solution_value
        df = pd.DataFrame(tp)
        df.to_csv('./result/year.csv')

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
        #mdl.export_as_lp('seasonal initiate problem_' + str(pavnum))
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
        print("bestobj: ", self.bestobj, "masterobj: ", self.masObj)
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
    def writesol(self): # problem here not fix, x value is not saved here
        with open(address, 'a') as file2:
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
                        #if self.x[(i, j, k)].solution_value > 0.9:
                        if self.bestsol[i][j][k] > 0.9:

                            if j == 1:
                                file2.write(" 0 ")
                            elif j == 2:
                                file2.write(" P ")
                            elif j == 3:
                                file2.write(" M ")
                            elif j == 4:
                                file2.write(" R ")
                file2.write(" \n")

class cgyear():
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
        mdl = Model('year initiate problem')
        x = mdl.binary_var_dict([(pavnum, j, k) for j in r for k in t], name='x')
        mdl.maximize(mdl.sum(-self.pcl[pavnum][j][k] * x[pavnum, j, k] for j in r for k in t))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            >= Qmin for j in range(1, T+1))
        mdl.add_constraints(q0[pavnum]*(dr**j)+mdl.sum(x[pavnum, k, l] * e[k] * (dr ** (j-l-1)) for k in r for l in range(0, j))
                            <= Qmax for j in range(1, T+1))
        mdl.add_constraint(mdl.sum(x[pavnum, j, k] for j in rs for k in t) <= g)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in r) == 1 for k in t)
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in rs) == 0 for k in yeart)
        #mdl.add_constraint(x[pavnum, 3, 1] == 1)
        #mdl.export_as_lp('year initiate problem_' + str(pavnum))
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
                print('year initial problem '+str(i)+' infeasible')
                return False
        print('obj: ' + str(tobj))
        self.cll.append(ncl)
        self.subObj = tobj
        print('year initial problem solved')

    def masterp(self):
        cnum =len(self.cll) #column number
        cn =[i for i in range(0,cnum)] #column index
        self.ldl = [0.0 for i in range(0, cnum)]
        mdl = Model('year master problem')
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
            print('year master problem solved')
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
        mdl.add_constraints(mdl.sum(x[pavnum, j, k] for j in rs) == 0 for k in yeart)
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
        print("bestobj: ", self.bestobj, "masterobj: ", self.masObj)
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
            file1.write("=============CG year plan solution=============")
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
    def writesol(self):
        with open(address, 'a') as file2:
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
                        #if self.x[(i, j, k)].solution_value > 0.9:
                        if self.bestsol[i][j][k] > 0.9:

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


def plancompare():
    yp = cgyear()
    yp.runcg()
    yp.output()
    #yp.writesol()
    sp = cg()
    sp.runcg()
    sp.output()
    #sp.writesol()
def milpcompare():
    yp = milp()
    yp.runmodel1()
    sp = yearmilp()
    sp.runmodel1()

plancompare()
#milpcompare()