## Continous cart pole using evolutionary strategy (ES)
## This is an implementation of the paper https://arxiv.org/pdf/1703.03864.pdf
## Running this script should do the trick

import gym
import numpy as np
from gym import wrappers
env = gym.make('Pendulum-v0')
#env = wrappers.Monitor(env, './home/sid/ccp_pg', force= True)

total_runs = 0
def simulate(policy, steps, graphics = False):
#method used for testing parameters that seem to be the best so far.
    observation = env.reset()
    R = 0
    for i in range(steps):
        if graphics: env.render()
        a = policy(observation)

        observation, reward, done, info = env.step(a)
        R += reward
        if done:
            break
    return R


def approx_policy_eval(policy, n_samples = 1):
#method used for testing to fine optimized parameters
    R = 0
    global total_runs
    total_runs += 1
    for _ in range(n_samples):
        observation = env.reset()
        #reset testing situation and run the trial n_samples times
        for i in range(1000):
          #number of steps the trial is run for
            a = policy(observation)
              #choose an action based on whatever policy ive chosen and the observed screen state

            observation, reward, done, info = env.step(a)

            R += reward
            if done:
                break


    return R/float(n_samples)




def esw1():

    npop = 8     # population size, the number of episodes it runs before evaluating itself
    sigma = 0.1    # noise standard deviation
    alpha = 0.1  # learning rate
    n = env.observation_space.shape[0]
    w = np.random.randn(n) # initial guess

    max_U = -float('inf')
    for i in range(1000):
      #make each worker do what follows
      N = np.random.randn(npop, n) #line3 for algorithm1
      #^ a list of length npop (10)  10 random numbers per every row in the observation
      R = np.zeros(npop)
      for j in range(npop):
        w_try = w + sigma*N[j]
        #w_try is the tweaked parameters from the randomized initial value. 
        R[j] = approx_policy_eval(lambda s: ([w_try.dot(s)]) )
        #R[j] is the list that stores 10 episodes of randomized chanigng parameters

#here we made w_t ry.dot(s) a list. 
#This is because approx_policy_eval takes an action as parameter, and that action has to be a list

      s = np.std(R)
      b = np.mean(R)

      if b > max_U:
          max_U = b
          max_w = w.copy()

      print ("this is b",b)

#below if statements compute the returns
      if s != 0:
        A = (R - b) / s
        #the difference between the score that it got R and the average score (average deviation) divided by the standard deviation (s)
      else:
          A = R
      if b > (-1000): return w
      w += alpha/(npop*sigma) * np.dot(N.T, A)
      #above is line 5 in algorithm1
      #npop->n   alpha->greek letter alpha     F->A   epsilonlookingthing -> N.T or the noise   w -> theta

    print ('max', max_U)
    return max_w



w = esw1()

 #w is the current parameters that the es method is dealing with facilitate the solve score (higher than 950) or its highest 
r = 0
for i in range(100):
    r+=simulate(lambda s: [(w.dot(s))], 1000)

print ('average_return over 100 trials:', r/100.0)
print ('total episodes', total_runs)
