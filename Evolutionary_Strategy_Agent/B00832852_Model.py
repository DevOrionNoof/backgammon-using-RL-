from B00832852_Agent import MLP_M
from backgammon import backgammon
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import pickle

#This function is to get the winning rate for the white player in 100 games
def get_fittest(agent):
    wins=[]
    for episode in range(100):
        game = backgammon()
        while game.get_winner() is None:
            moves=game.moves
            if moves[0][-1]==1:
                scores=[]
                for m in moves:
                    scores.append(agent.predict(m))
                mov=np.argmax(scores)
                state=moves[mov]
                game.make_move(moves[mov])
            else:
                for m in moves:
                    #against random
                    game.score_move(m, np.random.rand())
            if game.get_winner() is not None:
                wins.append(game.get_winner())
    return wins.count("WHITE")/len(wins)

#population list of 50 of MLP model
population = [MLP_M(28, 50, 50, 1) for i in range(50)]
# create list to store each generation fitness
generation_fitness = []

#print initial preformance of the population to evaluate training later
initial_fitness=np.mean([get_fittest(i) for i in population])
print("initial fittness without traning is : {}".format(initial_fitness))

#define sigma
sig = 0.05

# loop for 100 generations to calc fitness for agent in population
for i in tqdm(range(0, 100)):
    fit_ness = [get_fittest(i) for i in population]
    #print the average of generation fitness
    print("Fitness of Generation : {} is :{:.2f} ".format(i + 1, np.mean(fit_ness)))
    #sort the top 10 population fitness for next generation
    population = [population[i] for i in np.argsort(fit_ness)[::-1]][:10]
    #parent will be the first agent in the sorted population
    parent = population[0]
    #store it in variable best_fit
    best_fit = np.argsort(fit_ness)[::-1][0]
    this_gen_fit = []
    new_population = []
    #40 children loop
    for k in range(40):
        #create a child by using a deepcopy of parent
        child = deepcopy(parent)
        #Mutation for child's weight by adding random values and this way learned form  Pollack and Blair(1998)-A3 paper
        for row in range(len(child.w1)):
            for col in range(len(child.w1[row])):
                if np.random.randint(100) < 2:
                    #create random values in random value ranged from -1 to 1
                    child.w1[row][col] += ((np.random.rand() * 1 * 2) - 1)

        for row in range(len(child.w2)):
            for col in range(len(child.w2[row])):
                if np.random.randint(100) < 2:
                    child.w2[row][col] += ((np.random.rand() * 1 * 2) - 1)

        for row in range(len(child.w3)):
            for col in range(len(child.w3[row])):
                if np.random.randint(100) < 2:
                    child.w3[row][col] += ((np.random.rand() * 1 * 2) - 1)
        #update bias
        child.b1 += sig * ((np.random.rand() * 1 * 2) - 1)
        child.b2 += sig * ((np.random.rand() * 1 * 2) - 1)
        child.b3 += sig * ((np.random.rand() * 1 * 2) - 1)

        #save the child fitness
        child_fit = get_fittest(child)

        #if the child fitness is greater then update perent weight
        if child_fit > best_fit:
            # using the equation from
            for row in range(len(child.w1)):
                for col in range(len(child.w1[row])):
                    # updating weights of parent according to equation Pollack and Blair(1998)-A3 paper
                    parent.w1[row][col] = 0.95 * parent.w1[row][col] + 0.05 * child.w1[row][col]
            for row in range(len(child.w2)):
                for col in range(len(child.w2[row])):
                    parent.w2[row][col] = 0.95 * parent.w2[row][col] + 0.05 * child.w2[row][col]
            for row in range(len(child.w3)):
                for col in range(len(child.w3[row])):
                    parent.w3[row][col] = 0.95 * parent.w3[row][col] + 0.05 * child.w3[row][col]
            #update the parent agent bias
            parent.b1 = 0.95 * parent.b1 + 0.05 * child.b1
            parent.b2 = 0.95 * parent.b2 + 0.05 * child.b2
            parent.b3 = 0.95 * parent.b3 + 0.05 * child.b3
        # add parent agent to the the next or new poplulation
        new_population.append(parent)
    #append average
    generation_fitness.append(np.mean(fit_ness))
    #extend the list of population with new population
    population.extend(new_population)

best_agent=population[1]

#create pickle file
with open(f'B00832852_Model.pkl', 'wb') as file:
    pickle.dump(best_agent, file)
