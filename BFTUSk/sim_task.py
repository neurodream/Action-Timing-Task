#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy.optimize import differential_evolution


n_reward_chance_endpoints = 6
reward_sizes = [1,2,3]

# ratios as 
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

def run_sim(strategy="random", exploit_kill=True, verbose=False, float_vals=None):

    # main output var
    rows = []

    # global vars
    fps = 60
    snr_db_levels = [34] # [15, 23, 34, 51, 76]
    task_time_min = 25
    n_samples = 9
    dec_win_base_sec = 0.8
    dec_win_jitter_rng_sec = 0.5
    flash_off_dur = 0
    reward_sizes = [1, 2, 3]
    n_blocks = 4
    blocked_time = 0.3
    break_dur_sec = 60

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
                float_vals = strategies[strategy] # TODO find better name for float_vals
            
            # clamp to minimum
            float_vals = np.maximum(float_vals, blocked_time)

            for i, reward_magnitude_it in enumerate(reward_sizes):
                for j, p_end_it in enumerate(reward_chance_endpoints):
                    if reward_size == reward_magnitude_it and p_end == p_end_it:
                        choice_time = float_vals[i,j]*sweep_dur
                        break

            routine_t = 0
            while True:
                
                # t = Builder’s routine clock
                
                # ------- start next flash -----------
                if current_i + 1 < n_samples and routine_t >= flash_onsets[current_i + 1]:
                    
                    current_i += 1
                
                # ------- end flash ------------------
                if current_i >= 0 and routine_t >= flash_offsets[current_i]:
                    pass
                
                if routine_t > choice_time:
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


def plot_simulation(strategies, colors, n_runs=10, plot_strategy=None, best_vals_dict=None):
    dfs = {s: [run_sim(strategy=s, exploit_kill=False)[0] for _ in range(n_runs)]
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
            axes[1, 0].plot(sorted(df["score"]), color=c)

    axes[0, 0].set_ylabel("sorted RT (frames)")
    axes[1, 0].set_ylabel("score (AU)")
    axes[0, 0].set_title("independent var")
    axes[1, 0].set_title("dependent var")
    axes[1, 0].set_xlabel("trial N")
    axes[0, 0].legend(handles=patches, title="response strategy")

    end_scores = {s: [df["score"].iloc[-1] for df in dfs[s]] for s in strategies}
    means = [np.mean(end_scores[s]) for s in strategies]
    sems = [np.std(end_scores[s], ddof=1)/np.sqrt(n_runs) for s in strategies]
    axes[1, 1].bar(range(len(strategies)), means, yerr=sems,
                   color=colors, capsize=4)
    axes[1, 1].set_xticks(range(len(strategies)))
    axes[1, 1].set_xticklabels(strategies)
    axes[1, 1].set_title("end scores")

    plt.tight_layout()
    plt.show()



# if running this experiment as a script...
if __name__ == '__main__':
    run_sim()