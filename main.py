import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
# import torchga

from experts.PG import PG
from cost import CostNN2
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ENV SETUP
env_name = 'CartPole-v0'
env = gym.make(env_name).unwrapped
if seed is not None:
    env.seed(seed)
n_actions = env.action_space.n
state_shape = env.observation_space.shape
state = env.reset()

# LOADING EXPERT/DEMO SAMPLES
demo_trajs = np.load('expert_samples/pg_cartpole.npy', allow_pickle=True)
print(len(demo_trajs))

# INITILIZING POLICY AND REWARD FUNCTION
policy = PG(state_shape, n_actions)
cost_f = CostNN2(state_shape[0] + 1)
policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-2)
# cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)



import torch
import pygad.torchga
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    costs_samp = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=samp_input)

    costs_demo = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=demo_input)

    # abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))

    # solution_fitness = 1.0 / abs_error

    return -loss_IOC.detach().numpy()

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Create the PyTorch model.
input_layer = torch.nn.Linear(state_shape[0] + 1, 10)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(10, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.L1Loss()


# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 10 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights


# ga_instance = pygad.GA(num_generations=num_generations,
#                     num_parents_mating=num_parents_mating, 
#                     fitness_func=fitness_function,
#                     sol_per_pop=sol_per_pop, 
#                     num_genes=num_genes,
#                     parent_selection_type=parent_selection_type,
#                     keep_parents=keep_parents,
#                     crossover_type=crossover_type,
#                     mutation_type=mutation_type


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                    #    sol_per_pop=1000,
                    #    num_genes=1,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)







samp_input = None
demo_input = None
mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), np.array([])

# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            probs = np.array(traj[1]).reshape(-1, 1)
        actions = np.array(traj[2]).reshape(-1, 1)
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)

D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
return_list, sum_of_cost_list = [], []
for i in range(1000):
    trajs = [policy.generate_session(env) for _ in range(EPISODES_TO_PLAY)]
    sample_trajs = trajs + sample_trajs
    D_samp = preprocess_traj(trajs, D_samp)

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    loss_rew = []
    for _ in range(REWARD_FUNCTION_UPDATE):
        selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
        selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

        D_s_samp = D_samp[selected_samp]
        D_s_demo = D_demo[selected_demo]

        #D̂ samp ← D̂ demo ∪ D̂ samp
        D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)

        states, probs, actions = D_s_samp[:,:-2], D_s_samp[:,-2], D_s_samp[:,-1]
        states_expert, actions_expert = D_s_demo[:,:-2], D_s_demo[:,-1]

        # Reducing from float64 to float32 for making computaton faster
        states = torch.tensor(states, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        states_expert = torch.tensor(states_expert, dtype=torch.float32)
        actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

        samp_input = torch.cat((states, actions.reshape(-1, 1)), dim=-1)
        demo_input = torch.cat((states_expert, actions_expert.reshape(-1, 1)), dim=-1)




        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        # UPDATING THE COST FUNCTION
        # cost_optimizer.zero_grad()
        # loss_IOC.backward()
        # cost_optimizer.step()


        ga_instance.run()
        
        loss_IOC = ga_instance.best_solutions_fitness[-1]

        loss_rew.append(-loss_IOC)

    for traj in trajs:
        states, actions, rewards = traj
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
            
        # costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()


        data_inputs = torch.cat((states, actions.reshape(-1, 1)), dim=-1)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        # Make predictions based on the best solution.
        predictions = pygad.torchga.predict(model=model,
                                            solution=solution,
                                            data=data_inputs)
        print("Predictions : \n", predictions.detach().numpy())

        costs=predictions.detach().numpy()


        cumulative_returns = np.array(get_cumulative_rewards(-costs, 0.99))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        logits = policy(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = torch.sum(
            log_probs * to_one_hot(actions, env.action_space.n), dim=1)
    
        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*1e-2) 

        # UPDATING THE POLICY NETWORK
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

    returns = sum(rewards)
    sum_of_cost = np.sum(costs)
    return_list.append(returns)
    sum_of_cost_list.append(sum_of_cost)

    mean_rewards.append(np.mean(return_list))
    mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))

    # PLOTTING PERFORMANCE
    if i % 10 == 0:
        # clear_output(True)
        print(f"mean reward:{np.mean(return_list)} loss: {loss_IOC}")

        plt.figure(figsize=[16, 12])
        plt.subplot(2, 2, 1)
        plt.title(f"Mean reward per {EPISODES_TO_PLAY} games")
        plt.plot(mean_rewards)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
        plt.plot(mean_costs)
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
        plt.plot(mean_loss_rew)
        plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve.png')
        plt.close()

    if np.mean(return_list) > 500:
        break
