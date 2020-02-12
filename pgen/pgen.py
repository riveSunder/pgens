import numpy as np
import gym
import pybullet
import pybullet_envs
import time

import pickle

def relu(x):
    x[x<0.0] = 0.0
    return x

def sigmoid(x):
    x = 1 / (1 + np.exp(-x)
    return x

class MLP():
    def __init__(self, weights, input_dim, hid_dim, output_dim):

        dim_x2h = input_dim*hid_dim
        weights = weights.squeeze()
        self.x2h = weights[:dim_x2h].reshape(input_dim, hid_dim)
        self.h2y = weights[dim_x2h:].reshape(hid_dim,output_dim)
        self.discrete = False

    def forward(self,obs):
        if len(obs.shape) is 1:
            obs = obs.reshape(1,obs.shape[0])

        h = np.tanh(np.matmul(obs, self.x2h))
        y = np.matmul(h,self.h2y)

        return y

    def get_action(self,obs):
        y = self.forward(obs)

        if self.discerete:
            action = np.argmax(y)
        else:
            action = y

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
                act = population[indy].forward(obs)

                act = np.tanh(act)
                if act.shape[1] > 1:
                    act = act.squeeze()
                obs, reward, done, info = env.step(act)
                epd_reward += reward
                total_steps += 1

        epd_reward /= epds

        fitness.append(epd_reward)

    return fitness, total_steps


def get_gen_fitness(env, generators, epds=8, \
        input_dim=5, hid_dim=8, output_dim=1, dim_latent=2):

    len_generators = len(generators)
    total_total_steps = 0

    pop_fitness = []
    num_weights = input_dim*hid_dim + hid_dim*output_dim
    epd_epds = 4
    for ii in range (len_generators):
        gen_fitness = None
        for epd in range(epds): 
            # generate a set of policy weights from a latent vector,
            # using the generator
            latent_space = np.random.randn(1, dim_latent)
            agent_params = generators[ii].forward(latent_space)
            for epd_epd in range(epd_epds):
                agent_mean = np.tanh(agent_params[:1,0:num_weights])
                agent_var = np.abs(agent_params[:1,num_weights:])

                agent_weights = np.random.normal(agent_mean, agent_var)
                
                agent = [MLP(agent_weights, input_dim, hid_dim, output_dim)]
                
                fitness, total_steps = get_fitness(env, agent, epds=1)

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
        var=None, lr=0.25):


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

    new_var =  np.ones((num_weights))

    return new_mean, new_var, generators[sort_indices[jj]]


def train_generators(env, max_generations, \
        input_dim, output_dim, hid_dim, tag="default", fit_threshold=float("Inf")):

    latent_dim, gen_hid = 2, 16
    gen_out = 2*(input_dim*hid_dim + output_dim*hid_dim) 

    gen_mean = np.zeros((latent_dim*gen_hid + gen_hid*gen_out))
    gen_var = np.ones((latent_dim*gen_hid + gen_hid*gen_out))
    num_generators = 64
    epds = 4 

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
                output_dim=output_dim)
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
            with open("results/dists.pickle".format(tag), "wb") as f:
                pickle.dump(means_and_vars, f)
            with open("results/generators.pickle".format(tag), "wb") as f:
                pickle.dump(best_generators, f)

    import pdb; pdb.set_trace()

    with open("results/{}_fitness.pickle".format(tag), "wb") as f:
        pickle.dump(fitnesses, f)
    with open("results/dists.pickle".format(tag), "wb") as f:
        pickle.dump(means_and_vars, f)
    with open("results/generators.pickle".format(tag), "wb") as f:
        pickle.dump(best_generators, f)

def main():

    # make env
    env_name = "InvertedPendulumBulletEnv-v0"
    #env_name = "InvertedPendulumSwingupBulletEnv-v0"
    #env_name = "InvertedDoublePendulumBulletEnv-v0"
    #env_name = "HalfCheetahBulletEnv-v0"
    #env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    print("making {} env".format(env_name))

    input_dim = env.observation_space.sample().shape[0]
    output_dim = env.action_space.sample().shape[0]

    if "Swingup" in env_name:
        tag = "Swingup"+ str(int(time.time()))[-5:]
    elif "Double" in env_name:
        tag = "Double"+ str(int(time.time()))[-5:]
    else:
        tag = "InvPend" + str(int(time.time()))[-5:]

    train_generators(env, 20000, input_dim, output_dim, hid_dim=16, tag=tag, fit_threshold=999.)

if __name__ == "__main__":
    main()
