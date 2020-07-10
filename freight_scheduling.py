#Importing necessary libraries
import numpy as np
import pandas as pd
from itertools import combinations
from more_itertools import locate
import random
import copy
#Defining our network

    #list of all stations
stations = list(range(0,16))

    #n-d list of all connected stations for station as index
adjacent_stations = [[1,4],[0,5,2],[1,6,3],[2,7],[0,5,8],[1,4,9,6],[2,5,7,10],[3,6,11],[4,9,12],
                     [5,8,10,13],[6,9,11,14],[7,10,15],[8,13],[12,9,14],[13,10,15],[14,11]]

    #all segments

segments =  combinations(stations,2)
segments = list(segments)

#Trains Info
    
    #Index as train number
origins = [0,15,2]
destinations = [15,0,14]
no_of_trains = len(origins)

#Generating the population

def individual_generator(origin,destination):
    neighbours = copy.deepcopy(adjacent_stations)       #only copy.deepcopy will work for 2-D lists to copy it in another variable without refrencing
    temp = [] 
    i = origin
    while(i!=destination):
        temp.append(i)   
        for l in neighbours[i]:
            if i in neighbours[l]:
                neighbours[l].remove(i)
        if len(neighbours[i]) == 0:
            temp = []
            break
        i = random.choice(neighbours[i])
        
    temp.append(destination)
    return(temp)
    
    
def population_generator(origin,destination,count):
    P = []
    i=0
    while i< count:
        P.append(individual_generator(origin,destination))
        if len(P[i]) == 1:
            P.pop(i)
        else:   
            i = i+1
    return(P)
    
    #population is specific for every train w.r.t its origin and destination

population = []
count = 10

for i in range(no_of_trains):
    population.append(population_generator(origins[i],destinations[i],count))
    
population = np.asarray(population)
#Function for making population matrix as per our need, i.e either trainwise or according to individual



#Genetic operator and fitness, probabilities

generations = 10
prob_crossover = 0.8
prob_mutation = 0.1
    
    #Elitism, best individuals selected
    
def selection(population,no_of_trains,count):
    pop_fitness = population.transpose()
    
    c1 = list(range(len(pop_fitness)))
    c2=[]
    for i in range(len(pop_fitness)):
        c2.append(fitness(pop_fitness[i]))
    
    DC = pd.DataFrame([c1,c2]).transpose()
    DC.sort_values(by=[1])
        
    for i in range(count,len(pop_fitness)):
        del pop_fitness[DC[0][i]]

    
    return(pop_fitness.transpose())

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def swap(parent_1,parent_2,index_1,index_2):
    X_1 = parent_1[index_1[0]:index_1[1]]
    Y_1 = parent_2[index_2[0]:index_2[1]]

    del parent_1[index_1[0]:index_1[1]]
    del parent_2[index_2[0]:index_2[1]]
    for i in range(len(Y_1)):
        parent_1.insert(i+index_1[0],Y_1[i])
    for i in range(len(X_1)):
        parent_2.insert(i+index_2[0],X_1[i])
    return(parent_1,parent_2)


def crossover(X,Y):
    # CX = []
    # CY = []
    common_stations = intersection(X,Y)
    
    nodes = random.sample(common_stations,2)
    nodes = list(nodes)
    index_1 = []
    index_2 = []
        
    index_1.append(max(list(locate(X, lambda a: a == nodes[0]))))
    index_1.append(max(list(locate(X, lambda a: a == nodes[1]))))
    index_2.append(max(list(locate(X, lambda a: a == nodes[0]))))
    index_2.append(max(list(locate(X, lambda a: a == nodes[1]))))

    index_1.sort()    
    index_2.sort()
 
    X_copy = X.copy()
    Y_copy = Y.copy()
    CX,CY = swap(X_copy,Y_copy,index_1,index_2)
    return(CX,CY)
    
def mutation(X):
    temp = np.random.choice(X) 
    new_origin = temp          #the node(index) for mutation
    temp = X.index(new_origin)
    destination_ori = X[-1]
    del X[temp:len(X)]
    X1=(individual_generator(new_origin,destination_ori))
    return(X+X1)
    
def generation(population,no_of_trains):
    random_crossover = np.random.uniform(0,1,1)    
    for i in range(no_of_trains):
        if random_crossover <= prob_crossover:
            tc1,tc2 = crossover(random.choice(population[i]),random.choice(population[i]))
            # population[i].append(tc1)
            # population[i].append(tc2)
            np.append(population[i],tc1)
            np.append(population[i],tc2)

    random_mutation = np.random.uniform(0,1,1)    
    for i in range(no_of_trains):
        if random_mutation <= prob_mutation:
            t2 = mutation(random.choice(population[i]))
            # population[i].append(t2)
            np.append(population[i],t2)
            
def fitness(individual):
    fitness = 0  
    j=0
    for i in range(len(individual)):
         fitness += len(individual[i])-1      
    for i in combinations(individual,2):
        if(len(i[0])>=len(i[1])):
            s = 1
            b = 0
        else:
            s = 0
            b = 1
        while(j < len(i[s])-2 and j<30):
             
             if(i[s][j]==i[b][j]):
                 
                 if(j==0 and i[s][1]==i[b][1]):
                     i[s].insert(0,i[s][0])
                     fitness +=1
                     j=0
                     continue

                 if(i[b][j+1]==i[s][j-1]):
                     i[b].insert(0,i[b][0])
                     fitness +=1
                     j=0
                     continue
 
                 else:
                     i[s].insert(0,i[s][0])
                     fitness +=1
                     j=0
                     continue
                     
             if(i[s][j]==i[b][j+1] and i[s][j+1]==i[b][j]):
                 i[s].insert(0,i[s][0])
                 fitness +=1
                 j=0
                 continue
              
             j=j+1
                 
    
    return(fitness)
    
#Run over generations
    
    #first generation with the initial population without selection

generation(population,no_of_trains)

    #Further generations
for i in range(generations):
    selection(population,no_of_trains,count)
    generation(population,no_of_trains)

#priting the final answer
final_fitness = []
final_population = population.transpose()
for i in range(count):
    final_fitness.append(fitness(final_population[i]))

print('\n Minimized total running time of all trains (in units)  = ',min(final_fitness),'\n for the train paths: \n')

for i in range(no_of_trains):
    
    print('train ',i,' follows the path- ',final_population[final_fitness.index(min(final_fitness))][i])

