#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy.optimize import differential_evolution
from functools import lru_cache
import itertools

n_reward_chance_endpoints = 6
reward_sizes = [1,2,3]

strategies = {
    "random":       np.random.rand(len(reward_sizes), n_reward_chance_endpoints),
    "early":        np.ones((len(reward_sizes),n_reward_chance_endpoints))*0.0,
    "middle":       np.ones((len(reward_sizes),n_reward_chance_endpoints))*0.5,
    "late":         np.ones((len(reward_sizes),n_reward_chance_endpoints))*1.0,
    "strategy1":    np.array([
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.60]
                    ]),
}

class QAgent:
    def __init__(self, n_bins=10, α=0.2, γ=0.95, ε=0.1):
        self.n_bins, self.α, self.γ, self.ε = n_bins, α, γ, ε
        self.Q = np.zeros((3, n_bins + 1, 2))     # reward_size-1 , p_bin , action

    def _bin(self, p):                       # 0-…-n_bins
        return min(int(p * self.n_bins), self.n_bins)

    def act(self, r_sz, p):
        s = (r_sz - 1, self._bin(p))
        return np.random.randint(2) if random() < self.ε else np.argmax(self.Q[s])

    def learn(self, r_sz, p, a, r, p_next, done):
        s = (r_sz - 1, self._bin(p))
        s_next = (r_sz - 1, self._bin(p_next))
        target = r if done else r + self.γ * self.Q[s_next].max()
        self.Q[s + (a,)] += self.α * (target - self.Q[s + (a,)])

class OptimalAgent:
    def __init__(self, max_flashes=12, cost_per_frame=0.0):
        self.m  = max_flashes
        self.c  = cost_per_frame          # can be reassigned anytime
        self.V  = self._memo()            # lazily rebuilt when c changes

    def _memo(self):
        from functools import lru_cache
        c = self.c                        # close over current cost
        @lru_cache(None)
        def V(r_sz, a, b, s, f):
            if f <= 0 or s <= 0:                    # must press now
                return r_sz * a / (a + b) - c
            exp = a / (a + b)
            press = r_sz * exp - c                 # reward – cost this frame
            wait  = -c + exp * V(r_sz, a + 1, b,     s - 1, f - 1) \
                       + (1-exp) * V(r_sz, a,     b + 1, s - 1, f - 1)
            return max(press, wait)
        return V

    # call after you change self.c so the cache uses new cost
    def set_cost(self, new_c):
        self.c = new_c
        self.V = self._memo()

    # decision rule
    def act(self, r_sz, a, b, s, f):
        exp = a / (a + b)
        press = r_sz * exp - self.c
        wait  = -self.c + exp * self.V(r_sz, a + 1, b,     s - 1, f - 1) \
                        + (1-exp) * self.V(r_sz, a,     b + 1, s - 1, f - 1)
        return press >= wait


def run_sim(policy="random", exploit_kill=True, verbose=False, float_vals=None):
    """
    policy ∈ {"optimal","qlearn"} ∪ strategies.keys()
        'optimal'  → Bayesian oracle
        'qlearn'   → ε-greedy tabular Q-learner
    """

    # main output var
    rows = []

    # global vars
    fps = 60
    snr_db_levels = [34] # [15, 23, 34, 51, 76]
    task_time_min = 25
    n_samples = 9
    dec_win_base_sec = 0.8
    dec_win_jitter_rng_sec = 0.05
    flash_off_dur = 0
    reward_sizes = [1, 2, 3]
    n_blocks = 4
    blocked_time = 0.0 # consider decrepit, let's not block time - did not help, and bloats design
    break_dur_sec = 60

    if policy == "optimal":
        agent_opt = OptimalAgent(cost_per_frame=0)
    elif policy == "qlearn":
        agent_q   = QAgent()

    # derived vars
    task_time = task_time_min*60*fps
    dec_win_base = dec_win_base_sec*fps
    dec_win_jitter_rng = dec_win_jitter_rng_sec*fps
    break_dur = break_dur_sec*fps
    reward_chance_endpoints = np.linspace(0, 1, n_reward_chance_endpoints)
    block_dur = task_time//n_blocks
    
    # init_counters
    score = 0
    breaks_given = 0
    
    # reset timer
    t = 0
    
    trial_i = 0
    
    for _ in range(n_blocks):

        time_for_break = False

        while not time_for_break:

            trial_i += 1

            if verbose: print("trial #", trial_i)

            ################################
            
            # --- Routine "trial_baseline" ---
            baseline_time = 1.5*fps
            jitter          = np.random.uniform(-0.1*fps, +0.1*fps)
            baseline_time += jitter
            
            routine_t = 0
            while True:
                
                if routine_t > baseline_time:
                    break
                
                # refresh the screen
                t += 1
                routine_t += 1

            ################################

            # --- Routine "reward_size_cue" ---
            
            reward_size = np.random.choice(reward_sizes)
            
            routine_t = 0
            while routine_t < 0.6*fps:
                
                # refresh the screen
                t += 1
                routine_t += 1

            ################################

            # --- Routine "decision_window" ---
            pressed = False
            choice_time = None
            if policy == "optimal":
                a = b = 1                      # Beta(1,1) prior
            if policy == "qlearn":
                prev_p = 0
            
            # ---------- probability path ----------
            p_end          = np.random.choice(reward_chance_endpoints)
            p_lin          = np.linspace(0.5, p_end, n_samples)
            snr_db         = np.random.choice(snr_db_levels)
            noise          = np.random.normal(0,
                                 np.sqrt(np.mean(p_lin**2)) / (10**(snr_db/20)),
                                 p_lin.shape)
            if exploit_kill:
                x       = np.linspace(0, 1, n_samples)          # 0 … last flash
                alpha   = 2.5                                   # >1 → slow start, fast finish
                p_curve = 0.5 + (p_end - 0.5) * x**alpha        # concave ramp
                p_noisy        = np.clip(p_curve + noise, 0, 1)
            else:
                p_noisy        = np.clip(p_lin + noise, 0, 1)
            
            # ---------- flash schedule ----------
            on_dur        = dec_win_base + np.random.uniform(-dec_win_jitter_rng,
                                                             +dec_win_jitter_rng,
                                                             n_samples)      # ON per flash
            soa           = on_dur + flash_off_dur                          # ON + OFF
            flash_onsets  = np.cumsum(np.concatenate(([0], soa[:-1])))
            flash_offsets = flash_onsets + on_dur
            
            # ---------- pointer setup ------------
            
            current_i      = -1                           # “no flash yet”
            
            sweep_dur = flash_onsets[-1] + soa[-1]

            if float_vals is None:
                if policy in strategies:
                    float_vals = strategies[policy] # TODO find better name for float_vals
                else:
                    float_vals = strategies["random"] # TODO find better name for float_vals
            
            # clamp to minimum
            float_vals = np.maximum(float_vals, blocked_time)

            for i, reward_magnitude_it in enumerate(reward_sizes):
                for j, p_end_it in enumerate(reward_chance_endpoints):
                    if reward_size == reward_magnitude_it and p_end == p_end_it:
                        choice_time = float_vals[i,j]*sweep_dur
                        break

            current_p = 0
            routine_t = 0
            while True:
                
                # t = Builder’s routine clock
                
                # ------- start next flash -----------
                if current_i + 1 < n_samples and routine_t >= flash_onsets[current_i + 1]:
                    current_i += 1
                    current_p  = p_noisy[current_i]

                    if policy == "optimal":        # update posterior counts
                        a += current_p
                        b += 1 - current_p

                    if policy == "qlearn":         # keep last visible p for update
                        prev_p = current_p
                
                # ------- end flash ------------------
                if current_i >= 0 and routine_t >= flash_offsets[current_i]:
                    pass
                
                if policy == "optimal":
                    frames_left = task_time - t                      # t = global frame counter
                    steps_left = max(0, n_samples - current_i - 1)
                    press_now  = agent_opt.act(reward_size, a, b, steps_left, frames_left)
                elif policy == "qlearn":
                    act        = agent_q.act(reward_size, current_p)
                    press_now  = bool(act)
                else:                          # fixed look-up strategy
                    press_now  = routine_t > choice_time
                
                if press_now:
                    pressed = True
                    choice_time = routine_t
                    break
                
                # ------- end trial ----------------------------------------    
                if routine_t > sum(soa):
                    current_i += 1
                    break
                
                # refresh the screen
                t += 1
                routine_t += 1

            ################################

            # --- Routine "hold" ---
            routine_t = 0
            while routine_t < 0.3*fps:
                
                # refresh the screen
                t += 1
                routine_t += 1

            ################################

            # --- Routine "reward_reveal" ---
            
            if current_i + 1 > len(p_noisy):
                reward_prob = 0 # waited until end of trial
            else:
                reward_prob = p_noisy[current_i]
            
            rand_float = random()
            if rand_float > reward_prob:
                reward = False
            else:
                reward = True
                score += reward_size

            reward_obtained = reward_size if reward else 0
            if policy == "qlearn":
                agent_q.learn(reward_size, prev_p, act,
                            reward_obtained, 0, True)
            if policy == "optimal":
                new_c = score / max(1, t)          # reward per frame so far
                agent_opt.set_cost(new_c)                     # adaptive opportunity cost
            
            reward_reveal_time = 1*fps
            jitter          = np.random.uniform(-0.1*fps, +0.1*fps)
            reward_reveal_time += jitter
            
            routine_t = 0
            while True:

                # Run 'Each Frame' code from calc_reward_reveal_time
                if routine_t > reward_reveal_time:
                    break
                
                # refresh the screen
                t += 1
                routine_t += 1
            
            rows.append(dict(
                rt_frames=choice_time,
                sample_ind=int(current_i),
                likelihood=p_noisy[current_i] if current_i + 1 <= n_samples else 0,
                likelihoods=p_noisy,
                reward=reward,
                reward_size=reward_size,
                reward_chance_endpoint=p_end,
                snr_db=snr_db,
                score=score
            ))

            
            # check if it is time for a break
            if (t // block_dur > breaks_given) or (t >= task_time):
                time_for_break = True
        
        ################################
            
        # break
        breaks_given += 1
        
        if int(breaks_given) < int(n_blocks):
            if verbose: print("BREAK")
            t += break_dur

    ################################

    return pd.DataFrame(rows), score, float_vals


def optimize_float_vals(n_iter=2000):
    best = -np.inf
    best_vals = None
    for _ in tqdm(range(n_iter)):
        vals = np.random.rand(3,6)          # in-range proposal
        _, score, _ = run_sim(strategy="strategy1", float_vals=vals)    # evaluate
        if score > best:
            best, best_vals = score, vals
    return best_vals, best

def _objective_flat(x):
    # x has shape (18,)
    _, score = run_sim(float_vals=x.reshape(3, 6))
    return -score                        # maximise score → minimise –score

# TODO this function fails
def optimize_float_vals_de(popsize=5, maxiter=100, seed=None):
    res = differential_evolution(
        _objective_flat,
        bounds=[(0, 1)] * 18,
        popsize=popsize,
        maxiter=maxiter,
        polish=False,
        seed=seed                       # no workers -> serial, no pickling issues
    )
    return res.x.reshape(3, 6), -res.fun


def plot_best_vals(best_vals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    mat = best_vals.T
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='gray')
    for i, j in np.ndindex(mat.shape):
        ax.text(j, i, f'{mat[i, j]:.2f}',
                ha='center', va='center',
                color='white' if mat[i, j] > .5 else 'black')
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(0, 1, mat.shape[1])])
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel('reward_magnitude index')
    ax.set_ylabel('p_end index')
    return ax


def plot_simulation_old(strategies, colors,
                    n_runs=10,
                    agent_mode="strategy",
                    plot_strategy=None,
                    best_vals_dict=None,
                    **run_sim_kwargs):

    dfs = {s: [run_sim(strategy=s,
                       exploit_kill=False,
                       agent_mode=agent_mode,
                       **run_sim_kwargs)[0]          # keep original return
               for _ in range(n_runs)]
           for s in strategies}

    patches = [mpatches.Patch(color=c, label=s) for c, s in zip(colors, strategies)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6),
                             sharex='col', sharey='row',
                             gridspec_kw={'width_ratios': [3, 1]})

    if plot_strategy and best_vals_dict and plot_strategy in best_vals_dict:
        plot_best_vals(best_vals_dict[plot_strategy], ax=axes[0, 1])
    else:
        axes[0, 1].axis('off')

    for c, s in zip(colors, strategies):
        for df in dfs[s]:
            axes[0, 0].plot(sorted(df["rt_frames"]), color=c)
            axes[1, 0].plot(sorted(df["score"]),     color=c)

    axes[0, 0].set_ylabel("sorted RT (frames)")
    axes[1, 0].set_ylabel("score (AU)")
    axes[0, 0].set_title("independent var")
    axes[1, 0].set_title("dependent var")
    axes[1, 0].set_xlabel("trial N")
    axes[0, 0].legend(handles=patches, title="response strategy")

    end_scores = {s: [df["score"].iloc[-1] for df in dfs[s]] for s in strategies}
    means = [np.mean(end_scores[s]) for s in strategies]
    sems  = [np.std(end_scores[s], ddof=1)/np.sqrt(n_runs) for s in strategies]
    axes[1, 1].bar(range(len(strategies)), means, yerr=sems,
                   color=colors, capsize=4)
    axes[1, 1].set_xticks(range(len(strategies)))
    axes[1, 1].set_xticklabels(strategies)
    axes[1, 1].set_title("end scores")

    fig.suptitle(f"agent_mode = {agent_mode}")
    plt.tight_layout()
    plt.show()

def plot_simulation(policies,
                    colors=None,
                    n_runs=10,
                    best_vals_dict=None,
                    **run_sim_kwargs):
    """
    policies : list[str]
        Each item is either "optimal", "qlearn", or a fixed-strategy name
        understood by run_sim(policy=…)
    colors   : list[str] | None
        One colour per policy; if None a default MPL cycle is used
    n_runs   : int
        How many independent runs per policy
    **run_sim_kwargs
        Forwarded verbatim to run_sim
    """
    # ------------------ colours ------------------
    if colors is None:
        base_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = list(itertools.islice(itertools.cycle(base_cycle),
                                       len(policies)))
    if len(colors) < len(policies):
        raise ValueError("Need ≥1 colour per policy")

    # ------------------- sims --------------------
    dfs = {p: [run_sim(policy=p, **run_sim_kwargs)[0]
               for _ in range(n_runs)]
           for p in policies}

    # ------------------ figure -------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 6),
                             sharex='col', sharey='row',
                             gridspec_kw={'width_ratios': [3, 1]})

    # left column ─ RT & score traces
    for col, pol in zip(colors, policies):
        for df in dfs[pol]:
            axes[0, 0].plot(sorted(df["rt_frames"]), color=col, alpha=.6)
            axes[1, 0].plot(sorted(df["score"]),     color=col, alpha=.6)

    axes[0, 0].set_ylabel("sorted RT (frames)")
    axes[1, 0].set_ylabel("score (AU)")
    axes[1, 0].set_xlabel("trial N")

    patches = [mpatches.Patch(color=c, label=p)
               for c, p in zip(colors, policies)]
    axes[0, 0].legend(handles=patches, title="policy")

    # right-bottom ─ end-score bar-plot
    end_scores = {p: [df["score"].iloc[-1] for df in dfs[p]]
                  for p in policies}
    means = [np.mean(end_scores[p]) for p in policies]
    sems  = [np.std(end_scores[p], ddof=1)/np.sqrt(n_runs)
             for p in policies]

    axes[1, 1].bar(range(len(policies)), means, yerr=sems,
                   color=colors, capsize=4)
    axes[1, 1].set_xticks(range(len(policies)))
    axes[1, 1].set_xticklabels(policies, rotation=45)
    axes[1, 1].set_title("end scores")

    # spare cell top-right (can be used for best_vals etc.)
    axes[0, 1].axis("off")
    if best_vals_dict:
        for pol, vals in best_vals_dict.items():
            axes[0, 1].plot(vals, label=pol)
        axes[0, 1].legend()

    plt.tight_layout()
    plt.show()





# if running this experiment as a script...
if __name__ == '__main__':
    run_sim()