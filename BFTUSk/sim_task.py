#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import random
import numpy as np
import pandas as pd



def run_sim(verbose=False):

    rows = []

    # global vars
    fps = 60
    n_reward_chance_endpoints = 6
    snr_db_levels = [34] # [15, 23, 34, 51, 76]
    task_time = 25*60*fps
    n_samples = 9
    dec_win_base = 0.8*fps
    dec_win_jitter_rng = 0.5*fps
    flash_off_dur = 0
    reward_sizes = [1, 2, 3]
    n_blocks = 4

    # derived vars
    reward_chance_endpoints = np.linspace(0, 1, n_reward_chance_endpoints)
    block_dur = task_time//n_blocks
    sweep_span = 360
    # Set experiment start values for variable component break_dur
    break_dur = 60
    # init_counters
    score = 0
    score_computer = 0
    breaks_given = 0
    
    
    
    # reset timer
    t = 0
    

    
    trial_i = 0
    
    for _ in range(n_blocks):

        time_for_break = False

        while not time_for_break:

            trial_i += 1

            if verbose: print("trial #", trial_i)
            
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
            



            # --- Routine "reward_size_cue" ---
            
            reward_size = np.random.choice(reward_sizes)
            
            routine_t = 0
            while routine_t < 0.6*fps:
                
                # refresh the screen
                t += 1
                routine_t += 1




            # --- Routine "decision_window" ---
            
            # ---------- probability path ----------
            p_end          = np.random.choice(reward_chance_endpoints)
            p_lin          = np.linspace(0.5, p_end, n_samples)
            snr_db         = np.random.choice(snr_db_levels)
            noise          = np.random.normal(0,
                                 np.sqrt(np.mean(p_lin**2)) / (10**(snr_db/20)),
                                 p_lin.shape)
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
            
            # random answer
            choice_time = np.random.uniform(0, sweep_dur)

            routine_t = 0
            while True:
                
                # t = Builder’s routine clock
                
                # ------- start next flash -----------
                if current_i + 1 < n_samples and routine_t >= flash_onsets[current_i + 1]:
                    
                    current_i += 1
                    
                    # TODO set auto continue
                
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
            
            




            # --- Routine "hold" ---
            routine_t = 0
            while routine_t < 0.3*fps:
                
                # refresh the screen
                t += 1
                routine_t += 1
            




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
                snr_db=snr_db
            ))

            
            # check if it is time for a break
            if (t // block_dur > breaks_given) or (t >= task_time):
                time_for_break = True
            
        
        
        
        
            
        # break
        breaks_given += 1
        
        if int(breaks_given) < int(n_blocks):
            if verbose: print("BREAK")
            t += break_dur*fps




    return pd.DataFrame(rows)







# if running this experiment as a script...
if __name__ == '__main__':
    run_sim()
