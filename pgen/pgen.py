import argparse
import time

import numpy as np
import gym
import pybullet
import pybullet_envs

import pickle

def relu(x):
    x[x<0.0] = 0.0
    return x

def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x

class MLP():
    def __init__(self, weights, input_dim, hid_dim, \
            output_dim, discrete=False):

        dim_x2h = input_dim*hid_dim
        weights = weights.squeeze()
        self.x2h = weights[:dim_x2h].reshape(input_dim, hid_dim)
        self.h2y = weights[dim_x2h:].reshape(hid_dim,output_dim)
        self.discrete = discrete

    def forward(self,obs):
        if len(obs.shape) is 1:
            obs = obs.reshape(1,obs.shape[0])

        h = np.tanh(np.matmul(obs, self.x2h))
        y = np.matmul(h,self.h2y)

        return y

    def get_action(self,obs):
        y = self.forward(obs)

        if self.discrete:
            action = np.argmax(y)[np.newaxis]
        else:
            action = np.tanh(y)

        return action

def get_fitness(env, population, epds=4):

    pop_size = len(population)
    total_steps = 0
    fitness = []

    for indy in range(pop_size):
        epd_reward = 0.0

        for epd in range(epds):
            obs = env.reset()
            done = False
            while not done:
                act = population[indy].get_action(obs)

                if len(act.shape) > 1:
                    act = act[0]

                if act.shape[0] == 1 and population[indy].discrete:
                    # fix for CartPole
                    act = act[0]

                obs, reward, done, info = env.step(act)
                epd_reward += reward
                total_steps += 1

        epd_reward /= epds

        fitness.append(epd_reward)

    return fitness, total_steps


def get_gen_fitness(env, generators, epds=3, \
        input_dim=5, hid_dim=8, output_dim=1, dim_latent=2, discrete=False):

    len_generators = len(generators)
    total_total_steps = 0

    pop_fitness = []
    num_weights = input_dim*hid_dim + hid_dim*output_dim
    epd_epds = 1
    for ii in range (len_generators):
        gen_fitness = None
        for epd in range(epds): 
            # generate a set of policy weights from a latent vector,
            # using the generator
            latent_space = np.random.randn(1, dim_latent) / 3.
            agent_params = generators[ii].forward(latent_space)
            #for epd_epd in range(epd_epds):
            agent_mean = np.tanh(agent_params[:1,0:num_weights])

            #agent_var = np.abs(agent_params[:1,num_weights:])
            #agent_weights = np.random.normal(agent_mean, agent_var)
            #agent = [MLP(agent_weights, input_dim, hid_dim, output_dim, discrete=discrete)]

            
            agent = [MLP(agent_mean, input_dim, hid_dim, output_dim, discrete=discrete)]
            
            fitness, total_steps = get_fitness(env, agent, epds=epds)

            total_total_steps += total_steps
            
            gen_fitness = fitness if gen_fitness is None else \
                    np.array(gen_fitness) + np.array(fitness)
                    
        pop_fitness.append(gen_fitness[0]/(epd_epds*epds))

    return pop_fitness, total_total_steps

def get_generators(gen_mean, gen_var, num_generators=32, latent_dim=2, \
        hid_dim=16, output_dim=96):

    generators = []
    for gen in range(num_generators):
        
        weights = np.random.normal(gen_mean, gen_var)
        generators.append(MLP(weights, latent_dim, hid_dim, output_dim))

    return generators

def update_gen_dist(generators, fitness, mean=None, \
        var=None, lr=0.75):


    sort_indices = list(np.argsort(fitness))
    sort_indices.reverse()

    sorted_fitness = np.array(fitness)[sort_indices]
    
    num_elite = 8

    elite_fit = np.mean(sorted_fitness[:num_elite])
    
    num_weights = generators[0].x2h.shape[0]*generators[0].x2h.shape[1] \
            + generators[1].h2y.shape[0]*generators[1].h2y.shape[1] \

    elite_mean = np.zeros((num_weights))

    for jj in range(num_elite):
        
        weights = generators[sort_indices[jj]].x2h.ravel()
        weights = np.append(weights, generators[sort_indices[jj]].h2y.ravel())
        elite_mean += weights 

    elite_mean /= num_elite

    if mean is not None:
        new_mean = (1-lr) * mean + lr * elite_mean
    else:
        new_mean = elite_mean

    new_var =  1e-1*np.ones((num_weights))

    return new_mean, new_var, generators[sort_indices[jj]]


def train_generators(env, max_generations, input_dim, hid_dim, output_dim,\
        tag="default", fit_threshold=float("Inf"), discrete=False):

    latent_dim, gen_hid = 2, 16
    gen_out = 2*(input_dim*hid_dim + output_dim*hid_dim) 

    gen_mean = np.zeros((latent_dim*gen_hid + gen_hid*gen_out))
    gen_var = np.ones((latent_dim*gen_hid + gen_hid*gen_out))
    num_generators = 128
    epds = 3 

    smooth_fit = 0.0
    
    fitnesses = []
    means_and_vars = []
    best_generators = []
    total_interacts = 0
    t0 = time.time()
    for generation in range(max_generations):
        
        generators = get_generators(gen_mean, gen_var, num_generators,
                latent_dim=2, hid_dim=gen_hid, output_dim=gen_out)

        fitness, env_interacts = get_gen_fitness(env, generators, \
                epds=epds, input_dim=input_dim, hid_dim=hid_dim, \
                output_dim=output_dim, discrete=discrete)
        total_interacts += env_interacts
        
        gen_mean, gen_var, best_generator =\
                update_gen_dist(generators, fitness, gen_mean)

        fitnesses.append(fitness)
        means_and_vars.append([gen_mean, gen_var])
        best_generators.append(best_generator)

        smooth_fit = 0.75 * smooth_fit + 0.25 * np.max(fitness)
        print("generation {} max/mean fitness +/- std/min {:.2e}/{:.2e}+/-{:.2e}/{:.2e}".format(generation, np.max(fitness), np.mean(fitness),\
                np.std(fitness), np.min(fitness)))
        print("wall time {:.2f}, env interacts {}".format(\
                time.time()-t0, total_interacts))
        
        if smooth_fit > fit_threshold:
            print("task solved, ending evolution")
            break
        if generation % 50 == 0:
            with open("results/{}_fitness.pickle".format(tag), "wb") as f:
                pickle.dump(fitnesses, f)
            with open("results/{}_dists.pickle".format(tag), "wb") as f:
                pickle.dump(means_and_vars, f)
            with open("results/{}_generators.pickle".format(tag), "wb") as f:
                pickle.dump(best_generators, f)


    with open("results/{}_fitness.pickle".format(tag), "wb") as f:
        pickle.dump(fitnesses, f)
    with open("results/{}_dists.pickle".format(tag), "wb") as f:
        pickle.dump(means_and_vars, f)
    with open("results/{}_generators.pickle".format(tag), "wb") as f:
        pickle.dump(best_generators, f)

def train():

    parser = argparse.ArgumentParser("PGENs parameters")
    parser.add_argument("-n", "--env_name", default="CartPole-v1",\
            nargs="+", type=str, help="environment to train on")
    parser.add_argument("-g", "--max_generations", default=10,\
            type=int)

    args = parser.parse_args()
    generations = args.max_generations

    if type(args.env_name) != list:
        env_names = [args.env_name]
    else:
        env_names = args.env_name

    for env_name in env_names:
        env = gym.make(env_name)
        print("making {} env".format(env_name))

        input_dim = env.observation_space.sample().shape[0]
        try:
            output_dim = env.action_space.n
            discrete = True
        except:
            output_dim = env.action_space.sample().shape[0]
            discrete = False

        hid_dim = 16

        tag = env_name[0:8] + str(int(time.time()))[-5:]

        train_generators(env, generations, input_dim, hid_dim, output_dim, \
                tag=tag, fit_threshold=1500., discrete=discrete)

if __name__ == "__main__":
    train()
