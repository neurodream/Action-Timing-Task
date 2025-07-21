from BFTUSk.analysis import *
from BFTUSk.sim_task import *

# constructed example: steepest rise of reward likelihood over trial time
p_start = 0.0
reward_chance_endpoints = 6
noise = True
reward_sizes = [1,2,3] # much higher last reward

# always responding in the middle of the trial - not a good strategy, 
# so this should def. be beaten by any correctly implemented optimal agent
sim_task_middle_resp = SimTask(
    policy="middle",
    p_start=p_start, reward_chance_endpoints=reward_chance_endpoints, noise=noise, reward_sizes=reward_sizes)
sim_task_middle_resp.policy_lookup_table[0,:] = 0 # dispense with low rewards right away
sim_task_middle_resp.policy_lookup_table[2,:] = 1 # wait for best likelihood for highest reward
sim_task_middle_resp.run_sim()

sim_task_allegedly_optimal = SimTask(
    policy="optimal", 
    p_start=p_start, reward_chance_endpoints=reward_chance_endpoints, noise=noise, reward_sizes=reward_sizes)
sim_task_allegedly_optimal.run_sim()

print(sim_task_allegedly_optimal.score, sim_task_middle_resp.score)
if sim_task_allegedly_optimal.score < sim_task_middle_resp.score:
    print("test failed")