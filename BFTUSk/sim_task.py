#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import itertools
from matplotlib.colors import LinearSegmentedColormap


class OptimalAgent:
    """
    Bayesian one‑step‑look‑ahead agent
    """
    def __init__(self, task, sigma=0.05):

        self.task         = task
        self.prior_mean   = 0.5
        self.p_end_mean   = self.prior_mean      # running posterior mean

        # time constants
        self.step_sec = task.dec_win_base_sec    # 0.9
        self.reward_rate = self._compute_reward_rate()

        self.reset_trial()

    # -----------------------------------------------------------
    def _compute_reward_rate(self):
        mean_reward = np.mean(self.task.reward_sizes)
        mean_p      = 0.5                         # average p in a trial
        exp_trial_s = 3 + self.task.n_samples/2 * self.step_sec
        return mean_p * mean_reward / exp_trial_s

    # -----------------------------------------------------------
    def act(self, reward_size, seen_ps, _task_minutes_left):
        k = len(seen_ps) - 1
        if k < 0:
            return False

        self._update_sufficient(k, seen_ps[-1])

        cur_p  = seen_ps[-1]
        ev_now = cur_p * reward_size

        if k + 1 < self.task.n_samples:
            frac_next = (k + 1) / (self.task.n_samples - 1)
            p_next    = self.task.p_start + frac_next * (self.p_end_mean - self.task.p_start)
            ev_next   = p_next * reward_size
        else:
            return True                                # must press

        opp_cost  = self.step_sec * self.reward_rate
        gain_wait = ev_next - ev_now
        return gain_wait <= opp_cost

    # -----------------------------------------------------------
    def reset_trial(self):
        self.S2 = 0.0
        self.Sy = 0.0
        self.p_end_mean = self.prior_mean

    # -----------------------------------------------------------
    def _update_sufficient(self, k, p_obs):
        frac = k / (self.task.n_samples - 1)
        if frac == 0:          # first flash carries no slope info
            return

        y = p_obs - self.task.p_start
        self.S2 += frac**2
        self.Sy += frac * y

        if self.S2 > 1 * 10**(-9):
            slope         = self.Sy / self.S2
            self.p_end_mean = np.clip(self.task.p_start + slope, 0, 1)


class SimTask:
    def __init__(self, 
                 reward_chance_endpoints = 6,
                 reward_sizes = [1,2,3],
                 concave_trial_ramp = False,
                 policy_LUT = None,
                 p_start = 0.5,
                 n_samples = 9,
                 policy = "random",
                 fps = 60,
                 snr_db_levels = [34], # [15, 23, 34, 51, 76]
                 task_time_min = 25,
                 dec_win_base_sec = 0.8,
                 dec_win_jitter_rng_sec = 0.05,
                 flash_off_dur_sec = 0.1,
                 n_blocks = 4,
                 break_dur_sec = 60,
                 noise=False
                 ):
        
        self.reward_sizes = reward_sizes
        self.concave_trial_ramp = concave_trial_ramp
        self.policy_LUT = policy_LUT
        self.p_start = p_start
        self.n_samples = n_samples
        self.policy = policy
        self.fps = fps
        self.snr_db_levels = snr_db_levels
        self.task_time_min = task_time_min
        self.dec_win_base_sec = dec_win_base_sec
        self.dec_win_jitter_rng_sec = dec_win_jitter_rng_sec
        self.flash_off_dur_sec = flash_off_dur_sec
        self.n_blocks = n_blocks
        self.break_dur_sec = break_dur_sec
        self.noise = noise

        # derived vars
        self.task_time = self.task_time_min*60*self.fps
        self.dec_win_base = self.dec_win_base_sec*self.fps
        self.flash_off_dur = self.flash_off_dur_sec*self.fps
        self.dec_win_jitter_rng = self.dec_win_jitter_rng_sec*self.fps
        self.break_dur = self.break_dur_sec*self.fps
        if type(reward_chance_endpoints) == list:
            # directly set the possible endpoints
            self.reward_chance_endpoints = reward_chance_endpoints
        else:
            self.reward_chance_endpoints = [np.round(e, 3) for e in np.linspace(0, 1, reward_chance_endpoints)]
        self.block_dur = self.task_time//self.n_blocks

        self.set_fixed_strategies()

    def set_fixed_strategies(self):
        self.strategies = {
            "random":       np.random.rand( len(self.reward_sizes), len(self.reward_chance_endpoints)),
            "early":        np.ones((       len(self.reward_sizes), len(self.reward_chance_endpoints)))*0.0,
            "middle":       np.ones((       len(self.reward_sizes), len(self.reward_chance_endpoints)))*0.5,
            "late":         np.ones((       len(self.reward_sizes), len(self.reward_chance_endpoints)))*1.0,
            "strategy1":    np.array([ # TODO remove hardcoding - or strategy1 altogether
                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.60]
                            ]),
        }

        if self.policy_LUT is None:
            if self.policy in self.strategies:
                self.policy_LUT = self.strategies[self.policy]
            else:
                self.policy_LUT = self.strategies["random"]
                # ragged array to be filled, averaged at end of sim run, to have post hoc best values
                self._policy_LUT = [[[] for _ in self.reward_chance_endpoints] for _ in self.reward_sizes]

    def run_trial_main_part(self, reward_size):

        # --- Routine "decision_window" ---
        choice_time = float('inf')
        
        # ---------- probability path ----------
        p_end          = np.random.choice(self.reward_chance_endpoints)
        p_lin          = np.linspace(self.p_start, p_end, self.n_samples)
        snr_db         = np.random.choice(self.snr_db_levels)
        noise          = np.random.normal(0,
                            np.sqrt(np.mean(p_lin**2)) / (10**(snr_db/20)),
                            p_lin.shape)
        if not self.noise:
            noise = np.linspace(0, 0, self.n_samples)
        
        # TODO not tested whether p_start correctly implemented in concave ramp part
        if self.concave_trial_ramp:
            x       = np.linspace(0, 1, self.n_samples)          # 0 … last flash
            alpha   = 2.5                                   # >1 → slow start, fast finish
            p_curve = self.p_start + (p_end - self.p_start) * x**alpha        # concave ramp
            p_noisy        = np.clip(p_curve + noise, 0, 1)
        else:
            p_noisy        = np.clip(p_lin + noise, 0, 1)
        
        # ---------- flash schedule ----------
        on_dur        = self.dec_win_base + np.random.uniform(-self.dec_win_jitter_rng,
                                                        +self.dec_win_jitter_rng,
                                                        self.n_samples)      # ON per flash
        soa           = on_dur + self.flash_off_dur                          # ON + OFF
        flash_onsets  = np.cumsum(np.concatenate(([0], soa[:-1])))
        flash_offsets = flash_onsets + on_dur
        
        # ---------- pointer setup ------------
        
        current_i      = -1                           # “no flash yet”
        
        sweep_dur = flash_onsets[-1] + soa[-1]

        if self.policy != "optimal":
            for i, reward_magnitude_it in enumerate(self.reward_sizes):
                for j, p_end_it in enumerate(self.reward_chance_endpoints):
                    if reward_size == reward_magnitude_it and np.isclose(p_end, p_end_it):
                        choice_time = self.policy_LUT[i,j]*sweep_dur
                        break

        self.current_p = 0
        self.routine_t = 0
        seen_ps = []

        while True:
            # ------- start next flash -----------
            if current_i + 1 < self.n_samples and self.routine_t >= flash_onsets[current_i + 1]:
                current_i += 1
                current_p  = p_noisy[current_i]
                seen_ps.append(current_p)
            
            # ------- end flash ------------------
            if current_i >= 0 and self.routine_t >= flash_offsets[current_i]:
                pass
            
            if self.policy == "optimal":
                task_minutes_left = (self.task_time - self.t)/(60*self.fps)
                press_now  = self.agent_opt.act(reward_size, seen_ps, task_minutes_left)
            else:                          # fixed look-up strategy
                press_now  = self.routine_t >= choice_time
            
            if press_now:
                choice_time = self.routine_t
                return choice_time, sweep_dur, p_noisy, p_end, snr_db, current_i
            
            # ------- end trial ----------------------------------------    
            if self.routine_t > sum(soa):
                current_i += 1
                choice_time = self.routine_t
                return choice_time, sweep_dur, p_noisy, p_end, snr_db, current_i
            
            # refresh the screen
            self.t += 1
            self.routine_t += 1

    def run_trial(self):

        ################################
            
        # --- Routine "trial_baseline" ---
        baseline_time = 1.5*self.fps
        jitter          = np.random.uniform(-0.1*self.fps, +0.1*self.fps)
        baseline_time += jitter
        
        self.routine_t = 0
        while True:
            
            if self.routine_t > baseline_time:
                break
            
            # refresh the screen
            self.t += 1
            self.routine_t += 1

        ################################

        # --- Routine "reward_size_cue" ---
        
        reward_size = np.random.choice(self.reward_sizes)
        
        self.routine_t = 0
        while self.routine_t < 0.6*self.fps:
            
            # refresh the screen
            self.t += 1
            self.routine_t += 1

        ################################

        if self.policy == "optimal":
            self.agent_opt.reset_trial()

        choice_time, sweep_dur, p_noisy, p_end, snr_db, last_i = self.run_trial_main_part(reward_size)

        if self.policy == "optimal":
            self._policy_LUT[list(self.reward_sizes).index(reward_size)][list(self.reward_chance_endpoints).index(p_end)].append(choice_time/sweep_dur)

        ################################

        # --- Routine "hold" ---
        self.routine_t = 0
        while self.routine_t < 0.3*self.fps:
            
            # refresh the screen
            self.t += 1
            self.routine_t += 1

        ################################

        # --- Routine "reward_reveal" ---
        
        if last_i + 1 > len(p_noisy):
            reward_prob = 0 # waited until end of trial
        else:
            reward_prob = p_noisy[last_i]
        
        rand_float = random()
        if rand_float > reward_prob:
            reward = False
        else:
            reward = True
            self.score += reward_size
        
        reward_reveal_time = 1*self.fps
        jitter          = np.random.uniform(-0.1*self.fps, +0.1*self.fps)
        reward_reveal_time += jitter
        
        self.routine_t = 0
        while True:

            # Run 'Each Frame' code from calc_reward_reveal_time
            if self.routine_t > reward_reveal_time:
                break
            
            # refresh the screen
            self.t += 1
            self.routine_t += 1
        
        self.rows.append(dict(
            rt_frames=choice_time,
            trial_max=sweep_dur,
            sample_ind=int(last_i),
            likelihood=p_noisy[last_i] if last_i + 1 <= self.n_samples else 0,
            likelihoods=p_noisy,
            reward=reward,
            reward_size=reward_size,
            reward_chance_endpoint=p_end,
            snr_db=snr_db,
            score=self.score
        ))

    def run_block(self, verbose=False):

        time_for_break = False

        while not time_for_break:

            self.trial_i += 1

            if verbose: print("trial #", self.trial_i)

            self.run_trial()
            
            # check if it is time for a break
            if (self.t // self.block_dur > self.breaks_given) or (self.t >= self.task_time):
                time_for_break = True
        
        ################################
            
        # break
        self.breaks_given += 1
        
        if int(self.breaks_given) < int(self.n_blocks):
            if verbose: print("BREAK")
            self.t += self.break_dur

    def run_sim(self, verbose=False):
        """
        policy ∈ "optimal" ∪ strategies.keys()
            'optimal'  → Bayesian oracle
        """

        # main output var
        self.rows = []

        if self.policy == "optimal":
            self.agent_opt = OptimalAgent(self)
        
        # init_counters
        self.score = 0
        self.breaks_given = 0
        
        # reset timer
        self.t = 0
        
        self.trial_i = 0
        
        for _ in range(self.n_blocks):

            self.run_block(verbose=verbose)
        
        ################################

        # store post hoc "look up" (for plotting and diagnostics)
        if self.policy == "optimal":
            self.policy_LUT = np.array([[np.mean(cell) if cell else np.nan for cell in row] for row in self._policy_LUT])

        self.df = pd.DataFrame(self.rows)

    def get_blind_search_optimized_policy_LUT(self, n_iter=2000):
        best = -np.inf
        best_vals = None
        tmp_policy = self.policy
        self.policy = "random"
        for _ in tqdm(range(n_iter)):
            vals = np.random.rand(len(self.reward_sizes), self.n_reward_chance_endpoints)          # in-range proposal
            _, score, _ = self.run_sim()    # evaluate
            if score > best:
                best, best_vals = score, vals
        self.policy = tmp_policy
        return best_vals, best


def plot_policy_LUT(policy_LUT, sim_task=None, color="black", policy_label="", ax=None):

    custom_cmap = LinearSegmentedColormap.from_list("ow", ["white", color])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    mat = policy_LUT.T
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=custom_cmap)
    for i, j in np.ndindex(mat.shape):
        ax.text(j, i, f'{mat[i, j]:.2f}',
                ha='center', va='center', fontsize=8,
                color='white' if mat[i, j] > .5 else 'black')
    if sim_task:
        ax.set_xticks(range(mat.shape[1]))
        # ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(0, 1, mat.shape[1])])
        ax.set_xticklabels(sim_task.reward_sizes)
        ax.set_xlabel('reward magnitude')
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(sim_task.reward_chance_endpoints)
        ax.set_ylabel(r'$p_{end}$')
        # Force ticks and labels to be visible
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.tick_params(labelbottom=True, labelleft=True)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([])
    cbar.set_label('fraction of trial time', rotation=270, labelpad=10)

    if len(policy_label) > 0:
        ax.set_title(f"{policy_label} reaction policy")

    return ax


def plot_compare_policies(
                    policies,
                    colors=None,
                    n_runs=10,
                    policy_LUT_ind=False,
                    n_samples=9,
                    percent=True,
                    sort_RTs=True,
                    **run_sim_kwargs):
    """
    policies : list[str]
        Each item is either "optimal", or a fixed-strategy name
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
    sim_tasks = [SimTask(policy=p, **run_sim_kwargs) for p in policies]
    
    dfs = {}
    for i, p in enumerate(policies):
        dfs[p] = []
        for _ in range(n_runs):
            sim_tasks[i].run_sim()
            dfs[p].append(sim_tasks[i].df)
        

    fig, axes = plt.subplots(2, 2, figsize=(8, 6),
                            gridspec_kw={'width_ratios': [3, 1]})
    
    axes[1][0].sharex(axes[0][0])
    axes[1][1].sharey(axes[1][0])

    for col, pol in zip(colors, policies):
        for df in dfs[pol]:
            if percent:
                rt = (df["rt_frames"] / df["trial_max"])#*(10/9)
                ylab = "sorted RT (samples)"
            else:
                rt = df["rt_frames"]
                ylab = "sorted RT (frames)"
            if sort_RTs:
                axes[0, 0].plot(sorted(rt), color=col, alpha=.6)
                axes[1, 0].plot(sorted(df["score"]), color=col, alpha=.6)
            else:
                axes[0, 0].plot(rt, color=col, alpha=.6)
                axes[1, 0].plot(df["score"], color=col, alpha=.6)
                ylab = ylab.replace("sorted ", "")

    axes[0, 0].set_ylabel(ylab)
    axes[1, 0].set_ylabel("score (AU)")
    axes[1, 0].set_xlabel("trial N")

    if percent:
        tick_pos = np.arange(n_samples) / n_samples      # 0, 1/9, …, 8/9
        axes[0,0].set_yticks(tick_pos)
        axes[0,0].hlines(tick_pos, 0,
                        axes[0,0].get_xlim()[1],
                        color="k", ls="--", lw=.5)
        axes[0,0].set_yticklabels(range(1, n_samples+1))

        # axes[0, 0].hlines(np.linspace(0, 1, n_samples), 0, axes[0, 0].get_xlim()[1], color="k", linestyle="--", linewidth=0.5)

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

    # spare cell top-right (can be used for showing policy_LUT values)
    axes[0, 1].axis("off")
    if type(policy_LUT_ind) == int:
        base_color = colors[policy_LUT_ind]
        policy = policies[policy_LUT_ind]
        axes[0, 1].set_axis_on() 
        plot_policy_LUT(
            sim_tasks[policy_LUT_ind].policy_LUT, 
            sim_task=sim_tasks[policy_LUT_ind],
            color=base_color, policy_label=policy, ax=axes[0, 1]
            )

    plt.tight_layout()
    plt.show()