#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on May 27, 2025, at 07:11
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from imports
import random
import numpy as np
from psychopy import logging
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'BFTUS_task'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\nicade\\Documents\\projects\\BFTUS\\task\\Action-Timing-Task\\tests\\auto-run_tmp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-0.25,-0.25,-0.25], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.25,-0.25,-0.25]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "init_vars" ---
    # Set experiment start values for variable component n_reward_chance_endpoints
    n_reward_chance_endpoints = 6
    n_reward_chance_endpointsContainer = []
    # Set experiment start values for variable component snr_db_levels
    snr_db_levels = [34] # [15, 23, 34, 51, 76]
    snr_db_levelsContainer = []
    # Set experiment start values for variable component task_time_minutes
    task_time_minutes = 25
    task_time_minutesContainer = []
    # Set experiment start values for variable component n_samples
    n_samples = 9
    n_samplesContainer = []
    # Set experiment start values for variable component exp_scaling_factor
    exp_scaling_factor = 2.0
    exp_scaling_factorContainer = []
    # Set experiment start values for variable component pointer_start_angle
    pointer_start_angle = 0
    pointer_start_angleContainer = []
    # Set experiment start values for variable component letter_height
    letter_height = 0.5
    letter_heightContainer = []
    # Set experiment start values for variable component dec_win_base
    dec_win_base = 0.8
    dec_win_baseContainer = []
    # Set experiment start values for variable component dec_win_jitter_rng
    dec_win_jitter_rng = 0.05
    dec_win_jitter_rngContainer = []
    # Set experiment start values for variable component flash_off_dur
    flash_off_dur = 0.1
    flash_off_durContainer = []
    # Set experiment start values for variable component disk_r
    disk_r = 1.0
    disk_rContainer = []
    # Set experiment start values for variable component pointer_len
    pointer_len = 0.75
    pointer_lenContainer = []
    # Set experiment start values for variable component inner_min_frac
    inner_min_frac = 0.1
    inner_min_fracContainer = []
    # Set experiment start values for variable component inner_max_frac
    inner_max_frac = 0.95
    inner_max_fracContainer = []
    # Set experiment start values for variable component line_thickness
    line_thickness = 2
    line_thicknessContainer = []
    # Set experiment start values for variable component reward_sizes_euros
    reward_sizes_euros = [0.02, 0.05, 0.10]
    reward_sizes_eurosContainer = []
    # Set experiment start values for variable component money_amount
    money_amount = 0
    money_amountContainer = []
    # Set experiment start values for variable component resting_dur
    resting_dur = 1
    resting_durContainer = []
    # Set experiment start values for variable component break_pending
    break_pending = True
    break_pendingContainer = []
    # Set experiment start values for variable component n_blocks
    n_blocks = 4
    n_blocksContainer = []
    # Set experiment start values for variable component breaks_given
    breaks_given = 0
    breaks_givenContainer = []
    # Run 'Begin Experiment' code from derive_vars
    reward_chance_endpoints = np.linspace(0, 1, n_reward_chance_endpoints)
    
    block_dur = (task_time_minutes * 60)//n_blocks
    
    # apply global visual scaling factor
    
    line_thickness *= exp_scaling_factor
    pointer_len    *= exp_scaling_factor
    disk_r         *= exp_scaling_factor
    
    px_per_deg = win.size[0] / win.monitor.getWidth()   # ≈ 21 px/deg for your setup
    w_deg = (line_thickness/2) / px_per_deg                 # 3 px  → 0.14 deg
    
    # utility function
    def hex_to_psychopy_rgb(hexcode):
        hexcode = hexcode.lstrip('#')
        if len(hexcode) == 3:                       # allow shorthand
            hexcode = ''.join(ch*2 for ch in hexcode)
        if len(hexcode) != 6:
            raise ValueError('hex must be 3 or 6 digits')
        return [int(hexcode[i:i+2],16)/127.5-1 for i in (0,2,4)]
    # Set experiment start values for variable component element_color
    element_color = 0.75, 0.75, 0.75
    element_colorContainer = []
    # Set experiment start values for variable component sweep_span
    sweep_span = 360
    sweep_spanContainer = []
    # Set experiment start values for variable component progress_perc
    progress_perc = 0
    progress_percContainer = []
    # Run 'Begin Experiment' code from init_EEG
    class CLIMarker:
        
        def __init__(self):
            pass
            
        def sendMarker(self, val=20):
            print("marker received: ", val)
    
    from rusocsci import buttonbox
    
    try:
        bb = buttonbox.Buttonbox(port="COM1")
        bb.sendMarker(val=1)
    except:
        print("WARNING: no EEG device connected/detected - falling back to command line printing instead")
        bb = CLIMarker()
    # Set experiment start values for variable component break_dur
    break_dur = 30
    break_durContainer = []
    # Run 'Begin Experiment' code from surpress_logging
    from psychopy import logging
    logging.console.setLevel(logging.CRITICAL)
    
    # --- Initialize components for Routine "resting_state" ---
    fix_cross_resting_1 = visual.ShapeStim(
        win=win, name='fix_cross_resting_1', vertices='cross',
        size=(pointer_len/4, pointer_len/4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "TUS" ---
    fix_cross_TUS = visual.ShapeStim(
        win=win, name='fix_cross_TUS', vertices='cross',
        size=(pointer_len/4, pointer_len/4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "resting_state" ---
    fix_cross_resting_1 = visual.ShapeStim(
        win=win, name='fix_cross_resting_1', vertices='cross',
        size=(pointer_len/4, pointer_len/4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_baseline" ---
    fix_cross_baseline = visual.ShapeStim(
        win=win, name='fix_cross_baseline', vertices='cross',
        size=(disk_r/2, disk_r/2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=element_color,
        opacity=0.0, depth=-3.0, interpolate=True)
    # Run 'Begin Experiment' code from setup_pointer
    # Begin Experiment
    pointer = visual.Rect(win, width=w_deg, height=pointer_len,
                          anchor='bottom-center', fillColor=element_color, ori=pointer_start_angle)
                          
    pointer_blocker = visual.Rect(win, width=w_deg*3, height=pointer_len*0.9,
                          anchor='bottom-center', fillColor=(-0.25,-0.25,-0.25), ori=pointer_start_angle)
                          
    pointer.depth = -1
    pointer_blocker.depth = -2
    # Run 'Begin Experiment' code from setup_outline
    outline = visual.ShapeStim(
            win=win, name='outline',
            size=(pointer_len*2,pointer_len*2), 
            # size=(disk_r*inner_max_frac,disk_r*inner_max_frac), 
            vertices='circle',
            ori=0.0, pos=(0, 0), anchor='center',
            lineWidth=line_thickness/2,     colorSpace='rgb',  lineColor=element_color, fillColor=None,
            opacity=None, depth=0.0, interpolate=True)
            
    outline_bg = visual.ShapeStim(
            win=win, name='outline',        
            size=(pointer_len*2,pointer_len*2), 
            # size=(disk_r*inner_max_frac,disk_r*inner_max_frac), 
            vertices='circle',
            ori=0.0, pos=(0, 0), anchor='center',
            lineWidth=line_thickness/2,     colorSpace='rgb',  lineColor=[e*0.5 for e in element_color], fillColor=None,
            opacity=None, depth=0.0, interpolate=True)
    
    outline.closeShape = False
    
    # TODO maybe have to set depth to specific value, e.g.:
    outline.depth = -7
    outline_bg.depth = - 6
    # Run 'Begin Experiment' code from setup_rewardbar
    # Begin Experiment
    rewardbar = visual.Rect(win, width=w_deg*3, height=pointer_len*0.5,
                          anchor='bottom-center', fillColor=element_color, ori=pointer_start_angle)
                          
    rewardbar.depth = -3
    # Run 'Begin Experiment' code from setup_crosshair
    cross = visual.ShapeStim(
        win=win, name='fix_cross_baseline', vertices='cross',
        size=(pointer_len/4, pointer_len/4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=0,     colorSpace='rgb',  lineColor=None, fillColor=element_color,
        depth=10.0, interpolate=True)
                          
    cross.depth = -4
    
    # --- Initialize components for Routine "reward_size_cue" ---
    blank_text = visual.TextStim(win=win, name='blank_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from setup_inner
    # Begin Experiment
    inner = visual.ShapeStim(
            win=win, name='inner',
            size=(0,0), vertices='circle',
            ori=0.0, pos=(0, 0), anchor='center',
            lineWidth=line_thickness,     colorSpace='rgb',  lineColor=None, fillColor=element_color,
            opacity=None, depth=0.0, interpolate=True)
                          
    inner.depth = -3
    
    px_per_deg     = win.size[0] / win.monitor.getWidth()
    usable_r       = disk_r - (line_thickness/2) / px_per_deg
    inner_r_min    = inner_min_frac * usable_r
    inner_r_max    = inner_max_frac * usable_r
    # Run 'Begin Experiment' code from set_reward_size
    cent_colors = {
        10: [ 1.0,  0.68, -1.0 ],
        5:  [ 0.5,  0.5,  0.5  ],
        2: [ 0.6,  0.0, -0.6  ]
        }
    
    
    """
    cent_colors = {
        10: hex_to_psychopy_rgb("#D4B03E"),
        5: hex_to_psychopy_rgb("#B0B0B4"),
        2: hex_to_psychopy_rgb("#D1A679")
        
        }
    """
    
    # --- Initialize components for Routine "decision_window" ---
    DEBUG = visual.TextStim(win=win, name='DEBUG',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    give_response = keyboard.Keyboard()
    
    # --- Initialize components for Routine "hold" ---
    blank_post_response = visual.TextStim(win=win, name='blank_post_response',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reward_reveal" ---
    current_score_text = visual.TextStim(win=win, name='current_score_text',
        text=None,
        font='Open Sans',
        pos=(0, -pointer_len*1.75), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    current_win_text = visual.TextStim(win=win, name='current_win_text',
        text=None,
        font='Open Sans',
        pos=(0, -pointer_len*1.35), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "break_rest" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='Short break!\n\nPlease take a moment to actively relax, before we continue in...\n\n',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    break_counter = visual.TextStim(win=win, name='break_counter',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "goodbye_screen" ---
    money_reveal_text = visual.TextStim(win=win, name='money_reveal_text',
        text='The task is over.\nIn this session, you have made a total of ...€. Congratulations!\n\nThe experimenter will approach you shortly.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    quit_exp = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "init_vars" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('init_vars.started', globalClock.getTime())
    # keep track of which components have finished
    init_varsComponents = []
    for thisComponent in init_varsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "init_vars" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in init_varsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "init_vars" ---
    for thisComponent in init_varsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('init_vars.stopped', globalClock.getTime())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Run 'End Routine' code from start_exp_counter
    global_clock = core.Clock()
    task_start_time = global_clock.getTime()
    task_time = task_time_minutes * 60  # convert to seconds
    
    
    
    
    # the Routine "init_vars" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "resting_state" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state.started', globalClock.getTime())
    # keep track of which components have finished
    resting_stateComponents = [fix_cross_resting_1]
    for thisComponent in resting_stateComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fix_cross_resting_1* updates
        
        # if fix_cross_resting_1 is starting this frame...
        if fix_cross_resting_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fix_cross_resting_1.frameNStart = frameN  # exact frame index
            fix_cross_resting_1.tStart = t  # local t and not account for scr refresh
            fix_cross_resting_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross_resting_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            fix_cross_resting_1.status = STARTED
            fix_cross_resting_1.setAutoDraw(True)
        
        # if fix_cross_resting_1 is active this frame...
        if fix_cross_resting_1.status == STARTED:
            # update params
            pass
        
        # if fix_cross_resting_1 is stopping this frame...
        if fix_cross_resting_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross_resting_1.tStartRefresh + resting_dur-frameTolerance:
                # keep track of stop time/frame for later
                fix_cross_resting_1.tStop = t  # not accounting for scr refresh
                fix_cross_resting_1.frameNStop = frameN  # exact frame index
                # update status
                fix_cross_resting_1.status = FINISHED
                fix_cross_resting_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_stateComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state" ---
    for thisComponent in resting_stateComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('resting_state.stopped', globalClock.getTime())
    # the Routine "resting_state" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "TUS" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('TUS.started', globalClock.getTime())
    # keep track of which components have finished
    TUSComponents = [fix_cross_TUS]
    for thisComponent in TUSComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "TUS" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fix_cross_TUS* updates
        
        # if fix_cross_TUS is starting this frame...
        if fix_cross_TUS.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fix_cross_TUS.frameNStart = frameN  # exact frame index
            fix_cross_TUS.tStart = t  # local t and not account for scr refresh
            fix_cross_TUS.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross_TUS, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fix_cross_TUS.started')
            # update status
            fix_cross_TUS.status = STARTED
            fix_cross_TUS.setAutoDraw(True)
        
        # if fix_cross_TUS is active this frame...
        if fix_cross_TUS.status == STARTED:
            # update params
            pass
        
        # if fix_cross_TUS is stopping this frame...
        if fix_cross_TUS.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross_TUS.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                fix_cross_TUS.tStop = t  # not accounting for scr refresh
                fix_cross_TUS.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross_TUS.stopped')
                # update status
                fix_cross_TUS.status = FINISHED
                fix_cross_TUS.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TUSComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "TUS" ---
    for thisComponent in TUSComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('TUS.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # --- Prepare to start Routine "resting_state" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state.started', globalClock.getTime())
    # keep track of which components have finished
    resting_stateComponents = [fix_cross_resting_1]
    for thisComponent in resting_stateComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fix_cross_resting_1* updates
        
        # if fix_cross_resting_1 is starting this frame...
        if fix_cross_resting_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fix_cross_resting_1.frameNStart = frameN  # exact frame index
            fix_cross_resting_1.tStart = t  # local t and not account for scr refresh
            fix_cross_resting_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_cross_resting_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            fix_cross_resting_1.status = STARTED
            fix_cross_resting_1.setAutoDraw(True)
        
        # if fix_cross_resting_1 is active this frame...
        if fix_cross_resting_1.status == STARTED:
            # update params
            pass
        
        # if fix_cross_resting_1 is stopping this frame...
        if fix_cross_resting_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_cross_resting_1.tStartRefresh + resting_dur-frameTolerance:
                # keep track of stop time/frame for later
                fix_cross_resting_1.tStop = t  # not accounting for scr refresh
                fix_cross_resting_1.frameNStop = frameN  # exact frame index
                # update status
                fix_cross_resting_1.status = FINISHED
                fix_cross_resting_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_stateComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state" ---
    for thisComponent in resting_stateComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('resting_state.stopped', globalClock.getTime())
    # the Routine "resting_state" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    breaks = data.TrialHandler(nReps=4.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='breaks')
    thisExp.addLoop(breaks)  # add the loop to the experiment
    thisBreak = breaks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBreak.rgb)
    if thisBreak != None:
        for paramName in thisBreak:
            globals()[paramName] = thisBreak[paramName]
    
    for thisBreak in breaks:
        currentLoop = breaks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBreak.rgb)
        if thisBreak != None:
            for paramName in thisBreak:
                globals()[paramName] = thisBreak[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=10000.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial_baseline" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial_baseline.started', globalClock.getTime())
            # Run 'Begin Routine' code from calc_progress_perc
            progress_perc = global_clock.getTime()/task_time
            # Run 'Begin Routine' code from calc_baseline_time
            baseline_time = 1.5
            jitter          = np.random.uniform(-0.1, +0.1)
            baseline_time += jitter
            # Run 'Begin Routine' code from show_disk
            # disk.setAutoDraw(True)
            # Run 'Begin Routine' code from setup_pointer
            pointer.setOri(pointer_start_angle)
            pointer.setAutoDraw(True)
            pointer.opacity = 0
            
            pointer_blocker.setOri(pointer_start_angle)
            pointer_blocker.setAutoDraw(True)
            pointer_blocker.opacity = 0
            # Run 'Begin Routine' code from setup_outline
            outline.setOri(pointer_start_angle)
            outline.setAutoDraw(True)
            
            outline_bg.setOri(pointer_start_angle)
            outline_bg.setAutoDraw(True)
            # Run 'Begin Routine' code from setup_rewardbar
            rewardbar.setOri(pointer_start_angle)
            rewardbar.setAutoDraw(True)
            
            rewardbar.opacity = 0
            # Run 'Begin Routine' code from setup_crosshair
            cross.setAutoDraw(True)
            cross.ori = pointer_start_angle
            # Run 'Begin Routine' code from trigger_trial_baseline_start
            bb.sendMarker(val=10)
            # keep track of which components have finished
            trial_baselineComponents = [fix_cross_baseline]
            for thisComponent in trial_baselineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_baseline" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from calc_baseline_time
                if t > baseline_time:
                    continueRoutine = False
                
                # *fix_cross_baseline* updates
                
                # if fix_cross_baseline is starting this frame...
                if fix_cross_baseline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_cross_baseline.frameNStart = frameN  # exact frame index
                    fix_cross_baseline.tStart = t  # local t and not account for scr refresh
                    fix_cross_baseline.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_cross_baseline, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross_baseline.started')
                    # update status
                    fix_cross_baseline.status = STARTED
                    fix_cross_baseline.setAutoDraw(True)
                
                # if fix_cross_baseline is active this frame...
                if fix_cross_baseline.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from setup_outline
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * (1 - progress_perc)
                
                theta = np.linspace(start_angle, end_angle, 100)
                radius = pointer_len 
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                outline.size = (1,1)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_baselineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_baseline" ---
            for thisComponent in trial_baselineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial_baseline.stopped', globalClock.getTime())
            # Run 'End Routine' code from setup_rewardbar
            rewardbar.ori = pointer_start_angle
            rewardbar.height = pointer_len*0.5
            rewardbar.opacity = 1
            # the Routine "trial_baseline" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "reward_size_cue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('reward_size_cue.started', globalClock.getTime())
            # Run 'Begin Routine' code from setup_inner
            inner.setAutoDraw(False)
            inner_radius_start = inner_r_min + (inner_r_max-inner_r_min) * 0.5
            inner.size  = (inner_radius_start, inner_radius_start)
            inner.opacity = 1
            # Run 'Begin Routine' code from set_reward_size
            reward_size_euro = random.choice(reward_sizes_euros)
            reward_size_cent = int(reward_size_euro*100)
            
            bb.sendMarker(val=100+reward_size_cent)
            
            inner.color = cent_colors[reward_size_cent]
            rewardbar.color = cent_colors[reward_size_cent]
            
            pointer.opacity = 1
            pointer_blocker.opacity = 1
            # keep track of which components have finished
            reward_size_cueComponents = [blank_text]
            for thisComponent in reward_size_cueComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "reward_size_cue" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.6:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_text* updates
                
                # if blank_text is starting this frame...
                if blank_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_text.frameNStart = frameN  # exact frame index
                    blank_text.tStart = t  # local t and not account for scr refresh
                    blank_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_text.started')
                    # update status
                    blank_text.status = STARTED
                    blank_text.setAutoDraw(True)
                
                # if blank_text is active this frame...
                if blank_text.status == STARTED:
                    # update params
                    pass
                
                # if blank_text is stopping this frame...
                if blank_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_text.tStartRefresh + 0.6-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_text.tStop = t  # not accounting for scr refresh
                        blank_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_text.stopped')
                        # update status
                        blank_text.status = FINISHED
                        blank_text.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * (1 - progress_perc)
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in reward_size_cueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "reward_size_cue" ---
            for thisComponent in reward_size_cueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('reward_size_cue.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.600000)
            
            # --- Prepare to start Routine "decision_window" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('decision_window.started', globalClock.getTime())
            # Run 'Begin Routine' code from update_searchloc
            inner.depth = -3
            
            # ---------- probability path ----------
            p_end          = random.choice(reward_chance_endpoints)
            p_lin          = np.linspace(0.5, p_end, n_samples)
            snr_db         = random.choice(snr_db_levels)
            noise          = np.random.normal(0,
                                 np.sqrt(np.mean(p_lin**2)) / (10**(snr_db/20)),
                                 p_lin.shape)
            p_noisy        = np.clip(p_lin + noise, 0, 1)
            
            # ---------- inner-dot radii ----------
            
            # other version:
            
            # def p_to_diameter(p, d_min, d_max):
            #     """
            #     p 0-1 → diameter with equal annulus-area steps
            #     d_min, d_max in same units as ShapeStim.size
            #     """
            #     p      = np.clip(p, 0, 1)
            #     r_min  = d_min / 2
            #     r_max  = d_max / 2
            #     r      = np.sqrt(r_min**2 + p * (r_max**2 - r_min**2))
            #     return 2 * r
            # 
            # inner_radii = p_to_diameter(p_noisy, inner_r_min, inner_r_max) # careful: misnomer: actually, "r" should be "d" (diameter)
            
            # old version:
            
            # inner_radii    = inner_r_min + (inner_r_max-inner_r_min) * p_noisy
            
            inner_radii = pointer_len*p_noisy
            
            # ---------- flash schedule ----------
            on_dur        = dec_win_base + np.random.uniform(-dec_win_jitter_rng,
                                                             +dec_win_jitter_rng,
                                                             n_samples)      # ON per flash
            soa           = on_dur + flash_off_dur                          # ON + OFF
            flash_onsets  = np.cumsum(np.concatenate(([0], soa[:-1])))
            flash_offsets = flash_onsets + on_dur
            
            rand_t = random.uniform(0.1, flash_offsets[-1] - 0.1)
            
            # ---------- pointer setup ------------
            angle_step     = 360 / (n_samples-1)          # CCW 9 → 3 o’clock
            pointer.pos    = (0, 0)                       # anchored at centre
            
            current_i      = -1                           # “no flash yet”
            
            sweep_dur = flash_onsets[-1] + soa[-1]
            deg_per_sec       = sweep_span / sweep_dur  # constant angular speed
            give_response.keys = []
            give_response.rt = []
            _give_response_allKeys = []
            # keep track of which components have finished
            decision_windowComponents = [DEBUG, give_response]
            for thisComponent in decision_windowComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "decision_window" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from track_exp_time
                if global_clock.getTime() >= task_time:
                    trials.finished = True
                    continueRoutine = False
                # Run 'Each Frame' code from update_searchloc
                pointer.ori   = pointer_start_angle + deg_per_sec * t          # continuous CCW sweep
                pointer_blocker.ori   = pointer_start_angle + deg_per_sec * t          
                rewardbar.ori = pointer_start_angle + deg_per_sec * t
                # cross.ori = pointer_start_angle + deg_per_sec * t
                
                pointer.pos   = (0, 0)
                rewardbar.pos = (0, 0)
                
                # t = Builder’s routine clock
                
                # ------- start next flash -----------
                if current_i + 1 < n_samples and t >= flash_onsets[current_i + 1]:
                    
                    bb.sendMarker(val=110 + int(p_end*10))
                    
                    current_i += 1
                    
                    # inner rewardbar appears
                    rewardbar.height = inner_radii[current_i]
                    rewardbar.opacity = 1
                
                    # inner-dot appears
                    inner.size  = (inner_radii[current_i], inner_radii[current_i])
                    inner.opacity = 1
                    
                    # set auto continue
                    if t > rand_t:
                        continueRoutine = False
                
                # ------- end flash ------------------
                if current_i >= 0 and t >= flash_offsets[current_i]:
                    inner.opacity = 0      # hide inner dot
                    rewardbar.opacity = 0
                
                # ------- end trial ----------------------------------------    
                if t > sum(soa):
                    current_i += 1
                    continueRoutine = False
                
                # *DEBUG* updates
                
                # if DEBUG is starting this frame...
                if DEBUG.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    DEBUG.frameNStart = frameN  # exact frame index
                    DEBUG.tStart = t  # local t and not account for scr refresh
                    DEBUG.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(DEBUG, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    DEBUG.status = STARTED
                    DEBUG.setAutoDraw(True)
                
                # if DEBUG is active this frame...
                if DEBUG.status == STARTED:
                    # update params
                    pass
                
                # *give_response* updates
                waitOnFlip = False
                
                # if give_response is starting this frame...
                if give_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    give_response.frameNStart = frameN  # exact frame index
                    give_response.tStart = t  # local t and not account for scr refresh
                    give_response.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(give_response, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'give_response.started')
                    # update status
                    give_response.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(give_response.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(give_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if give_response.status == STARTED and not waitOnFlip:
                    theseKeys = give_response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _give_response_allKeys.extend(theseKeys)
                    if len(_give_response_allKeys):
                        give_response.keys = _give_response_allKeys[-1].name  # just the last key pressed
                        give_response.rt = _give_response_allKeys[-1].rt
                        give_response.duration = _give_response_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from update_outline_cont1
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * (1 - progress_perc)
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in decision_windowComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "decision_window" ---
            for thisComponent in decision_windowComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('decision_window.stopped', globalClock.getTime())
            # Run 'End Routine' code from store_DVs
            thisExp.addData('rt_ms', t*1000)
            thisExp.addData('sample_ind', int(current_i))
            if current_i + 1 <= n_samples:
                thisExp.addData('likelihood', p_noisy[current_i])
            else:
                thisExp.addData('likelihood', 0)
            thisExp.addData('likelihoods', list(p_noisy))
            # Run 'End Routine' code from trigger_action_decision
            bb.sendMarker(val=220)
            # the Routine "decision_window" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "hold" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('hold.started', globalClock.getTime())
            # keep track of which components have finished
            holdComponents = [blank_post_response]
            for thisComponent in holdComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "hold" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_post_response* updates
                
                # if blank_post_response is starting this frame...
                if blank_post_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_post_response.frameNStart = frameN  # exact frame index
                    blank_post_response.tStart = t  # local t and not account for scr refresh
                    blank_post_response.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_post_response, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_post_response.started')
                    # update status
                    blank_post_response.status = STARTED
                    blank_post_response.setAutoDraw(True)
                
                # if blank_post_response is active this frame...
                if blank_post_response.status == STARTED:
                    # update params
                    pass
                
                # if blank_post_response is stopping this frame...
                if blank_post_response.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_post_response.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_post_response.tStop = t  # not accounting for scr refresh
                        blank_post_response.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_post_response.stopped')
                        # update status
                        blank_post_response.status = FINISHED
                        blank_post_response.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline_cont2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * (1 - progress_perc)
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in holdComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "hold" ---
            for thisComponent in holdComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('hold.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "reward_reveal" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('reward_reveal.started', globalClock.getTime())
            # Run 'Begin Routine' code from calculate_reward
            # TODO: yes or no based on bubble size
            
            if current_i + 1 > len(p_noisy):
                reward_prob = 0 # waited until end of trial
            else:
                reward_prob = p_noisy[current_i]
            
            rand_float = random.random()
            if rand_float > reward_prob:
                reward = False
                current_win_text.text = "NO GAIN" # ""
                bb.sendMarker(val=200)
            else:
                reward = True
                # reward_sound.play()
                current_win_text.text = f"GAIN + {reward_size_euro:.2f} €"
                money_amount += reward_size_euro
                bb.sendMarker(val=210)
                
            current_score_text.text = f"total: {money_amount:.2f} €"
            # Run 'Begin Routine' code from calc_reward_reveal_time
            reward_reveal_time = 1.0
            jitter          = np.random.uniform(-0.1, +0.1)
            reward_reveal_time += jitter
            # keep track of which components have finished
            reward_revealComponents = [current_score_text, current_win_text]
            for thisComponent in reward_revealComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "reward_reveal" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *current_score_text* updates
                
                # if current_score_text is starting this frame...
                if current_score_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    current_score_text.frameNStart = frameN  # exact frame index
                    current_score_text.tStart = t  # local t and not account for scr refresh
                    current_score_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(current_score_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'current_score_text.started')
                    # update status
                    current_score_text.status = STARTED
                    current_score_text.setAutoDraw(True)
                
                # if current_score_text is active this frame...
                if current_score_text.status == STARTED:
                    # update params
                    pass
                
                # *current_win_text* updates
                
                # if current_win_text is starting this frame...
                if current_win_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    current_win_text.frameNStart = frameN  # exact frame index
                    current_win_text.tStart = t  # local t and not account for scr refresh
                    current_win_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(current_win_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'current_win_text.started')
                    # update status
                    current_win_text.status = STARTED
                    current_win_text.setAutoDraw(True)
                
                # if current_win_text is active this frame...
                if current_win_text.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from calc_reward_reveal_time
                if t > reward_reveal_time:
                    continueRoutine = False
                # Run 'Each Frame' code from update_outline_cont3
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * (1 - progress_perc)
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in reward_revealComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "reward_reveal" ---
            for thisComponent in reward_revealComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('reward_reveal.stopped', globalClock.getTime())
            # Run 'End Routine' code from calculate_reward
            # check if it is time for a break
            if (global_clock.getTime() - task_start_time) // block_dur > breaks_given:
                trials.finished = True
                break_pending = True
                breaks_given += 1
            # Run 'End Routine' code from store_reward_info
            thisExp.addData('reward', reward)
            thisExp.addData('reward_size', reward_size_euro)
            thisExp.addData('reward_chance_endpoint', p_end)
            thisExp.addData('snr_db', snr_db)
            
            thisExp.nextEntry()
            # the Routine "reward_reveal" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 10000.0 repeats of 'trials'
        
        
        # --- Prepare to start Routine "break_rest" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_rest.started', globalClock.getTime())
        # Run 'Begin Routine' code from update_break_time_remain
        if not break_pending:
            continueRoutine = False  # skip break if not flagged
        else:
            break_pending = False  # reset break flag
            total_time = 0  # reset total_time for next block
        # Run 'Begin Routine' code from hide_pointer_outline
        pointer.opacity = 0
        pointer_blocker.opacity = 0
        outline.opacity = 0
        outline_bg.opacity = 0
        rewardbar.opacity = 0
        cross.opacity = 0
        # keep track of which components have finished
        break_restComponents = [break_text, break_counter]
        for thisComponent in break_restComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_rest" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + resting_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *break_counter* updates
            
            # if break_counter is starting this frame...
            if break_counter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_counter.frameNStart = frameN  # exact frame index
                break_counter.tStart = t  # local t and not account for scr refresh
                break_counter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_counter, 'tStartRefresh')  # time at next scr refresh
                # update status
                break_counter.status = STARTED
                break_counter.setAutoDraw(True)
            
            # if break_counter is active this frame...
            if break_counter.status == STARTED:
                # update params
                pass
            
            # if break_counter is stopping this frame...
            if break_counter.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_counter.tStartRefresh + break_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    break_counter.tStop = t  # not accounting for scr refresh
                    break_counter.frameNStop = frameN  # exact frame index
                    # update status
                    break_counter.status = FINISHED
                    break_counter.setAutoDraw(False)
            # Run 'Each Frame' code from update_break_time_remain
            break_counter.text = f"\n\n\n\n{int(resting_dur + 1 - t)}"
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_restComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_rest" ---
        for thisComponent in break_restComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_rest.stopped', globalClock.getTime())
        # Run 'End Routine' code from hide_pointer_outline
        pointer.opacity = 1
        pointer_blocker.opacity = 1
        outline.opacity = 1
        outline_bg.opacity = 1
        cross.opacity = 1
        # the Routine "break_rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 4.0 repeats of 'breaks'
    
    
    # --- Prepare to start Routine "goodbye_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('goodbye_screen.started', globalClock.getTime())
    quit_exp.keys = []
    quit_exp.rt = []
    _quit_exp_allKeys = []
    # Run 'Begin Routine' code from adjust_money_amount_display
    pointer.opacity = 0
    pointer_blocker.opacity = 0
    outline.opacity = 0
    outline_bg.opacity = 0
    rewardbar.opacity = 0
    cross.opacity = 0
    # keep track of which components have finished
    goodbye_screenComponents = [money_reveal_text, quit_exp]
    for thisComponent in goodbye_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "goodbye_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *money_reveal_text* updates
        
        # if money_reveal_text is starting this frame...
        if money_reveal_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            money_reveal_text.frameNStart = frameN  # exact frame index
            money_reveal_text.tStart = t  # local t and not account for scr refresh
            money_reveal_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(money_reveal_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            money_reveal_text.status = STARTED
            money_reveal_text.setAutoDraw(True)
        
        # if money_reveal_text is active this frame...
        if money_reveal_text.status == STARTED:
            # update params
            pass
        
        # *quit_exp* updates
        waitOnFlip = False
        
        # if quit_exp is starting this frame...
        if quit_exp.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            quit_exp.frameNStart = frameN  # exact frame index
            quit_exp.tStart = t  # local t and not account for scr refresh
            quit_exp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(quit_exp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'quit_exp.started')
            # update status
            quit_exp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(quit_exp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(quit_exp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if quit_exp.status == STARTED and not waitOnFlip:
            theseKeys = quit_exp.getKeys(keyList=['e'], ignoreKeys=["escape"], waitRelease=False)
            _quit_exp_allKeys.extend(theseKeys)
            if len(_quit_exp_allKeys):
                quit_exp.keys = _quit_exp_allKeys[-1].name  # just the last key pressed
                quit_exp.rt = _quit_exp_allKeys[-1].rt
                quit_exp.duration = _quit_exp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # Run 'Each Frame' code from adjust_money_amount_display
        money_reveal_text.text = f"The task is over.\nIn this session, you have made a total of {money_amount:.2f}€. Congratulations!\n\nThe experimenter will approach you shortly."
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in goodbye_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbye_screen" ---
    for thisComponent in goodbye_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('goodbye_screen.stopped', globalClock.getTime())
    # the Routine "goodbye_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
