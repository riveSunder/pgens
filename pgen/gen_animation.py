import argparse
import pickle
import gym

from pgen import MLP

import numpy as np
import skimage
import skimage.io
import matplotlib.pyplot as plt

def gen_animation(env_name, generator_file, latent_steps=10):

    env = gym.make(env_name)

    with open(generator_file, "rb") as file:
        pgen = pickle.load(file)[-1]
    
    input_dim = env.observation_space.shape[0]
    hid_dim = 16 
    output_dim = env.action_space.shape[0]

    num_weights = input_dim*hid_dim + hid_dim*output_dim

    #pgen  = MLP(weights, input_dim, hid_dim, num_weights*2)
    
    # walk the latent space, dimension 0
    latent_range = 1.
    latent_walk = np.arange(-latent_range/2, latent_range/2, latent_range / latent_steps)[:,np.newaxis]
    
    fitness_landscape = []
    epds = 8

    for latent_walk0 in latent_walk:
        for latent_walk1 in latent_walk:
            latent_space = np.array([[latent_walk0, latent_walk1]]).reshape(1,2)


            agent_params = pgen.forward(latent_space)
            agent_mean = np.tanh(agent_params[:1, 0:num_weights])

            agent = MLP(agent_mean, input_dim, hid_dim, output_dim=1)
            for epd in range(epds):
                obs = env.reset()
                done = False
                epd_reward = 0.0
                step = 0
                
                while not done:
                    act = agent.forward(obs)
                    act = np.tanh(act)

                    if act.shape[1] > 1:
                        act = act.squeeze()
                    obs, reward, done, info = env.step(act)
                    epd_reward += reward
                    
                    if epd == 0:
                        img = env.render(mode="rgb_array")
                        skimage.io.imsave("./results/figs/{}_step{}_latent{}.png".format(\
                                        env_name, str(step).zfill(3), \
                                        str(int(10*latent_walk0)) + str(int(latent_walk1))),\
                                        np.uint8(img))
                    step += 1
            fitness_landscape.append(epd_reward/epds)
            print("reward : {}".format(epd_reward), " latent space ", latent_space)

    fitness_landscape_plot = np.array(fitness_landscape).reshape(latent_steps, latent_steps)
    plt.figure(figsize=(10,10))
    plt.imshow(fitness_landscape_plot)
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--env_name", type=str, default="InvertedPendulumBulletEnv-v0")
    parser.add_argument("-g", "--generator_file", type=str, default="results/InvPend_generators.pickle")

    args = parser.parse_args()

    env_name = args.env_name

    generator_file = args.generator_file

    gen_animation(env_name, generator_file, latent_steps=32)
