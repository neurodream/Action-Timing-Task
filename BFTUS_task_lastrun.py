#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Mon Jun 30 14:25:33 2025
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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
# Run 'Before Experiment' code from code
#see begin routine
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'BFTUS_task_with_instructions_v3'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
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
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
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
        originPath='/Users/lutztebbe/Documents/Master_Internship/Task/Action-Timing-Task/BFTUS_task_lastrun.py',
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
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-0.25,-0.25,-0.25], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.25,-0.25,-0.25]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
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
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('start_demonstration') is None:
        # initialise start_demonstration
        start_demonstration = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_demonstration',
        )
    if deviceManager.getDevice('give_response_2') is None:
        # initialise give_response_2
        give_response_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='give_response_2',
        )
    if deviceManager.getDevice('go_to_task') is None:
        # initialise go_to_task
        go_to_task = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='go_to_task',
        )
    if deviceManager.getDevice('give_response') is None:
        # initialise give_response
        give_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='give_response',
        )
    if deviceManager.getDevice('quit_exp') is None:
        # initialise quit_exp
        quit_exp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='quit_exp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
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
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
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
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
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
    task_time_minutes = 2
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
    # Set experiment start values for variable component reward_sizes
    reward_sizes = [1, 2, 3]
    reward_sizesContainer = []
    # Set experiment start values for variable component resting_dur
    resting_dur = 1
    resting_durContainer = []
    # Set experiment start values for variable component n_blocks
    n_blocks = 1
    n_blocksContainer = []
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
    break_dur = 3
    break_durContainer = []
    # Run 'Begin Experiment' code from surpress_logging
    from psychopy import logging
    logging.console.setLevel(logging.CRITICAL)
    # Run 'Begin Experiment' code from init_counters
    score = 0
    score_computer = 0
    progress_perc = 0
    breaks_given = 0
    
    # decrepit (should be safe to delete)
    money_amount = 0
    
    # --- Initialize components for Routine "general_instruction" ---
    general_instruction_text = visual.TextStim(win=win, name='general_instruction_text',
        text='',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from instruction_master_code
    # This code REPLACES the old instruction_script in the 'Begin Experiment' tab.
    
    # This master script holds all our text and triggers for the tutorial.
    # Using parentheses makes the text easy to edit in the code editor.
    instruction_script = [
        # Page 1
        (
            "For the next 25 minutes you will sit infront of a clock that repeatedly "
            "offers you a chance to win bronze, silver or golden rewards.\n\n"
            "For any reward you manage to obtain, you will receive points which at the "
            "end of the experiment will be payed out as real money (€).\n\n"
            "--> bronze: worth 1 Point\n"
            "--> silver: worth 2 Points\n"
            "--> gold: worth 3 Points\n\n"
            "Every point is worth 0.XX€ at the end of the experiment.\n\n"
            "Press E to learn how the game works."
        ),
        
        # Page 2
        (
            "Every round, the clock starts running from 12 o'clock. You may choose at any point, "
            "until it completes a full circle, to go for the reward by pressing SPACE.\n\n"
            "Be careful, your chances of success will vary over time. As the clock ticks, "
            "the indicator will grow or shrink in size making it more or less likely for you to win.\n\n"
            "When you press SPACE, the clock determines whether or not you win the current reward. "
            "Then, the next round will start immediately.\n\n"
            "Press E to try this in a tutorial."
        ),
        
        # This special entry will trigger the 'positive' scripted trial
        {'TUTORIAL': 'positive', 'HINT': "Your chances of success vary between trials. \n\nIn this case, your odds increase over time. You might want to wait a moment for a better chance of winning. \n\n During the trial, press SPACE to go for the reward. Press E to start the demo."},
        
        # This will trigger the 'negative' scripted trial
        {'TUTORIAL': 'negative', 'HINT': "Your chances of success vary between trials. \n\nCareful, in this case, your odds are getting worse over time. \n\n During the trial, press SPACE to go for the reward. Press E to start the demo."},
    
        # This will trigger the 'high_noise' scripted trial
        {'TUTORIAL': 'high_noise', 'HINT': "Your chances of success vary between trials. \n\nIt may not always be so easy to tell if your odds are getting better or worse ... \n\n During the trial, press SPACE to go for the reward. Press E to start the demo."},
        
        # Page 3
        (
            "While the clock is running, please always focus on the fixation cross!\n\n"
            "The size of the clock has been designed so everything will be visible "
            "to you without having to look away from the crosshair.\n\n"
            "On the outer boundary of the clock you can see your overall time of 25 minutes slowly expire. "
            "It is fully in your hands how many rounds you will play within the 25 min.\n\n"
            "There will be three breaks for you to relax.\n\n"
            "Press E to continue"
        ),
        
        # This special entry will trigger your break animation
        "BREAK_ANIMATION",
        
        # Page 4 (Final Choice)
        (
            "That concludes our little tutorial. Good luck with collecting your riches!\n"
            "Remember, it's in your hands how many rounds you will play during the 25min.\n\n"
            "If you want to repeat the tutorial, please press R.\n"
            "If you have any questions you can always ask the experimenter.\n\n"
            "If you have no more questions, you can start the experiment by pressing SPACE."
        )
    ]
    
    # --- Variables to control the flow ---
    # (This part of your code remains the same)
    current_paragraph_index = 0
    run_instruction_trial = False
    run_break_animation = False
    current_tutorial_type = '' 
    tutorial_hint_text = ''
    
    # --- Initialize components for Routine "instruction_trial_baseline" ---
    fix_cross_baseline_3 = visual.ShapeStim(
        win=win, name='fix_cross_baseline_3', vertices='cross',
        size=(disk_r/2, disk_r/2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=element_color,
        opacity=0.0, depth=-1.0, interpolate=True)
    # Run 'Begin Experiment' code from setup_pointer_2
    # Begin Experiment
    pointer = visual.Rect(win, width=w_deg, height=pointer_len,
                          anchor='bottom-center', fillColor=element_color, ori=pointer_start_angle)
                          
    pointer_blocker = visual.Rect(win, width=w_deg*3, height=pointer_len*0.9,
                          anchor='bottom-center', fillColor=(-0.25,-0.25,-0.25), ori=pointer_start_angle)
                          
    pointer.depth = -1
    pointer_blocker.depth = -2
    # Run 'Begin Experiment' code from setup_outline_2
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
    # Run 'Begin Experiment' code from setup_rewardbar_2
    # Begin Experiment
    rewardbar = visual.Rect(win, width=w_deg*3, height=pointer_len*0.5,
                          anchor='bottom-center', fillColor=element_color, ori=pointer_start_angle)
                          
    rewardbar.depth = -3
    # Run 'Begin Experiment' code from setup_crosshair_2
    cross = visual.ShapeStim(
        win=win, name='fix_cross_baseline', vertices='cross',
        size=(pointer_len/4, pointer_len/4),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=0,     colorSpace='rgb',  lineColor=None, fillColor=element_color,
        depth=10.0, interpolate=True)
                          
    cross.depth = -4
    hint_text = visual.TextStim(win=win, name='hint_text',
        text='',
        font='Arial',
        pos=(0,0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    start_demonstration = keyboard.Keyboard(deviceName='start_demonstration')
    
    # --- Initialize components for Routine "instruction_reward_size_cue" ---
    blank_text_2 = visual.TextStim(win=win, name='blank_text_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from setup_inner_2
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
    # Run 'Begin Experiment' code from set_reward_size_2
    reward_size_colors = {
        3: [ 1.0,  0.68, -1.0 ],
        2:  [ 0.5,  0.5,  0.5  ],
        1: [ 0.6,  0.0, -0.6  ]
        }
    
    
    """
    cent_colors = {
        10: hex_to_psychopy_rgb("#D4B03E"),
        5: hex_to_psychopy_rgb("#B0B0B4"),
        2: hex_to_psychopy_rgb("#D1A679")
        
        }
    """
    
    # --- Initialize components for Routine "instruction_decision_window" ---
    give_response_2 = keyboard.Keyboard(deviceName='give_response_2')
    reminder_text = visual.TextStim(win=win, name='reminder_text',
        text='Press SPACE to go for the reward.',
        font='Arial',
        pos=(0, -6), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "instruction_hold" ---
    blank_post_response_2 = visual.TextStim(win=win, name='blank_post_response_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instruction_reward_reveal" ---
    current_win_text_2 = visual.TextStim(win=win, name='current_win_text_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Arial',
        pos=(0, -3), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "instruction_visualize_breaks" ---
    
    # --- Initialize components for Routine "cleanup_instruction" ---
    # Run 'Begin Experiment' code from code
    #see begin routine
    
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
    
    # --- Initialize components for Routine "task_instruction" ---
    task_instruction_text = visual.TextStim(win=win, name='task_instruction_text',
        text='-- ACTION TIMINING TASK --',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    continue_to_task = visual.TextStim(win=win, name='continue_to_task',
        text='\n\n\n\n\n\n\n\npress SPACE to start',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    go_to_task = keyboard.Keyboard(deviceName='go_to_task')
    
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
    reward_size_colors = {
        3: [ 1.0,  0.68, -1.0 ],
        2:  [ 0.5,  0.5,  0.5  ],
        1: [ 0.6,  0.0, -0.6  ]
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
        depth=-1.0);
    give_response = keyboard.Keyboard(deviceName='give_response')
    
    # --- Initialize components for Routine "hold" ---
    blank_post_response = visual.TextStim(win=win, name='blank_post_response',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reward_reveal" ---
    current_win_text = visual.TextStim(win=win, name='current_win_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "break_relax" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    score_text = visual.TextStim(win=win, name='score_text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "goodbye_screen" ---
    money_reveal_text = visual.TextStim(win=win, name='money_reveal_text',
        text='The task is over.\nIn this session, you have made a total of ...€. Congratulations!\n\nThe experimenter will approach you shortly.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='element_color', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    quit_exp = keyboard.Keyboard(deviceName='quit_exp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "init_vars" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('init_vars.started', globalClock.getTime(format='float'))
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
            endExperiment(thisExp, win=win)
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
    thisExp.addData('init_vars.stopped', globalClock.getTime(format='float'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Run 'End Routine' code from init_counters
    global_clock = core.Clock()
    task_start_time = global_clock.getTime()
    task_time = task_time_minutes * 60  # convert to seconds
    thisExp.nextEntry()
    # the Routine "init_vars" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    tutorial_handler = data.TrialHandler(nReps=99.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='tutorial_handler')
    thisExp.addLoop(tutorial_handler)  # add the loop to the experiment
    thisTutorial_handler = tutorial_handler.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTutorial_handler.rgb)
    if thisTutorial_handler != None:
        for paramName in thisTutorial_handler:
            globals()[paramName] = thisTutorial_handler[paramName]
    
    for thisTutorial_handler in tutorial_handler:
        currentLoop = tutorial_handler
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTutorial_handler.rgb)
        if thisTutorial_handler != None:
            for paramName in thisTutorial_handler:
                globals()[paramName] = thisTutorial_handler[paramName]
        
        # --- Prepare to start Routine "general_instruction" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('general_instruction.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from instruction_master_code
        # Reset flags
        run_instruction_trial = False
        run_break_animation = False
        
        # keep track of which components have finished
        general_instructionComponents = [general_instruction_text]
        for thisComponent in general_instructionComponents:
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
        
        # --- Run Routine "general_instruction" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *general_instruction_text* updates
            
            # if general_instruction_text is starting this frame...
            if general_instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                general_instruction_text.frameNStart = frameN  # exact frame index
                general_instruction_text.tStart = t  # local t and not account for scr refresh
                general_instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(general_instruction_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                general_instruction_text.status = STARTED
                general_instruction_text.setAutoDraw(True)
            
            # if general_instruction_text is active this frame...
            if general_instruction_text.status == STARTED:
                # update params
                general_instruction_text.setText('This text does not matter as it is controlled by instruction_master_code. Note: make sure the text is set every frame', log=False)
            # Run 'Each Frame' code from instruction_master_code
            # Get the current item from our script
            currentItem = instruction_script[current_paragraph_index]
            
            # Is the current item a special trigger?
            if isinstance(currentItem, dict):
                if 'TUTORIAL' in currentItem:
                    # Set the flags for the scripted trial
                    run_instruction_trial = True
                    current_tutorial_type = currentItem['TUTORIAL']
                    tutorial_hint_text = currentItem['HINT']
                    current_paragraph_index += 1
                    continueRoutine = False # End this routine to run the trial
            
            # Is it the break animation trigger?
            elif currentItem == "BREAK_ANIMATION":
                run_break_animation = True
                current_paragraph_index += 1
                continueRoutine = False # End this routine to run the animation
            
            # Otherwise, it's a normal text page
            else:
                # Display the text
                general_instruction_text.setText(currentItem)
            
                # Listen for key presses
                keys = event.getKeys(keyList=['e', 'r', 'space'])
            
                # Are we on the final page?
                if current_paragraph_index == len(instruction_script) - 1:
                    if 'r' in keys: # Repeat tutorial
                        current_paragraph_index = 0 # Reset to the beginning
                    elif 'space' in keys: # Start main experiment
                        tutorial_handler.finished = True # End the master loop
                        continueRoutine = False
            
                # Otherwise, on any other text page...
                elif 'e' in keys:
                    current_paragraph_index += 1 # Advance to next page
            
            # Check if we are done with all instruction pages
            if current_paragraph_index >= len(instruction_script):
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in general_instructionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "general_instruction" ---
        for thisComponent in general_instructionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('general_instruction.stopped', globalClock.getTime(format='float'))
        # the Routine "general_instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        instruction_scripted_trials = data.TrialHandler(nReps=run_instruction_trial, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='instruction_scripted_trials')
        thisExp.addLoop(instruction_scripted_trials)  # add the loop to the experiment
        thisInstruction_scripted_trial = instruction_scripted_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInstruction_scripted_trial.rgb)
        if thisInstruction_scripted_trial != None:
            for paramName in thisInstruction_scripted_trial:
                globals()[paramName] = thisInstruction_scripted_trial[paramName]
        
        for thisInstruction_scripted_trial in instruction_scripted_trials:
            currentLoop = instruction_scripted_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisInstruction_scripted_trial.rgb)
            if thisInstruction_scripted_trial != None:
                for paramName in thisInstruction_scripted_trial:
                    globals()[paramName] = thisInstruction_scripted_trial[paramName]
            
            # --- Prepare to start Routine "instruction_trial_baseline" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instruction_trial_baseline.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from calc_progress_perc_2
            progress_perc = global_clock.getTime()/task_time
            # Run 'Begin Routine' code from setup_pointer_2
            pointer.setOri(pointer_start_angle)
            pointer.setAutoDraw(True)
            pointer.opacity = 0  # Make visible
            
            pointer_blocker.setOri(pointer_start_angle)
            pointer_blocker.setAutoDraw(True)
            pointer_blocker.opacity = 0  # Make visible
            # Run 'Begin Routine' code from setup_outline_2
            outline.setOri(pointer_start_angle)
            outline.setAutoDraw(True)
            outline.opacity = 0  
            
            outline_bg.setOri(pointer_start_angle)
            outline_bg.setAutoDraw(True)
            outline_bg.opacity = 0 
            # Run 'Begin Routine' code from setup_rewardbar_2
            rewardbar.setOri(pointer_start_angle)
            rewardbar.setAutoDraw(True)
            rewardbar.opacity = 0 
            # Run 'Begin Routine' code from setup_crosshair_2
            cross.setAutoDraw(True)
            cross.ori = pointer_start_angle
            cross.opacity = 0 
            
            # create starting attributes for start_demonstration
            start_demonstration.keys = []
            start_demonstration.rt = []
            _start_demonstration_allKeys = []
            # keep track of which components have finished
            instruction_trial_baselineComponents = [fix_cross_baseline_3, hint_text, start_demonstration]
            for thisComponent in instruction_trial_baselineComponents:
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
            
            # --- Run Routine "instruction_trial_baseline" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fix_cross_baseline_3* updates
                
                # if fix_cross_baseline_3 is starting this frame...
                if fix_cross_baseline_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_cross_baseline_3.frameNStart = frameN  # exact frame index
                    fix_cross_baseline_3.tStart = t  # local t and not account for scr refresh
                    fix_cross_baseline_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_cross_baseline_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross_baseline_3.started')
                    # update status
                    fix_cross_baseline_3.status = STARTED
                    fix_cross_baseline_3.setAutoDraw(True)
                
                # if fix_cross_baseline_3 is active this frame...
                if fix_cross_baseline_3.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from setup_outline_2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                radius = pointer_len 
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                outline.size = (1,1)
                
                # *hint_text* updates
                
                # if hint_text is starting this frame...
                if hint_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    hint_text.frameNStart = frameN  # exact frame index
                    hint_text.tStart = t  # local t and not account for scr refresh
                    hint_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(hint_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'hint_text.started')
                    # update status
                    hint_text.status = STARTED
                    hint_text.setAutoDraw(True)
                
                # if hint_text is active this frame...
                if hint_text.status == STARTED:
                    # update params
                    hint_text.setText(tutorial_hint_text, log=False)
                
                # *start_demonstration* updates
                waitOnFlip = False
                
                # if start_demonstration is starting this frame...
                if start_demonstration.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    start_demonstration.frameNStart = frameN  # exact frame index
                    start_demonstration.tStart = t  # local t and not account for scr refresh
                    start_demonstration.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(start_demonstration, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'start_demonstration.started')
                    # update status
                    start_demonstration.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(start_demonstration.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(start_demonstration.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if start_demonstration.status == STARTED and not waitOnFlip:
                    theseKeys = start_demonstration.getKeys(keyList=['e'], ignoreKeys=["escape"], waitRelease=False)
                    _start_demonstration_allKeys.extend(theseKeys)
                    if len(_start_demonstration_allKeys):
                        start_demonstration.keys = _start_demonstration_allKeys[-1].name  # just the last key pressed
                        start_demonstration.rt = _start_demonstration_allKeys[-1].rt
                        start_demonstration.duration = _start_demonstration_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruction_trial_baselineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruction_trial_baseline" ---
            for thisComponent in instruction_trial_baselineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instruction_trial_baseline.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from setup_rewardbar_2
            rewardbar.ori = pointer_start_angle
            rewardbar.height = pointer_len*0.5
            rewardbar.opacity = 1
            # check responses
            if start_demonstration.keys in ['', [], None]:  # No response was made
                start_demonstration.keys = None
            instruction_scripted_trials.addData('start_demonstration.keys',start_demonstration.keys)
            if start_demonstration.keys != None:  # we had a response
                instruction_scripted_trials.addData('start_demonstration.rt', start_demonstration.rt)
                instruction_scripted_trials.addData('start_demonstration.duration', start_demonstration.duration)
            # the Routine "instruction_trial_baseline" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instruction_reward_size_cue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instruction_reward_size_cue.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from setup_inner_2
            inner.setAutoDraw(False)
            inner_radius_start = inner_r_min + (inner_r_max-inner_r_min) * 0.5
            inner.size  = (inner_radius_start, inner_radius_start)
            inner.opacity = 1
            # Run 'Begin Routine' code from set_reward_size_2
            # This code goes in the Begin Routine tab of set_reward_size_2.
            
            # 1. Set reward to Bronze for the tutorial.
            reward_size = 1
            
            # 2. Set the color for the reward indicator.
            inner.color = reward_size_colors[reward_size]
            rewardbar.color = reward_size_colors[reward_size]
            
            # 3. Redundantly set visuals to opaque to maintain structural consistency.
            #    This is the "unnecessary step" but it's harmless and keeps the
            #    code comparable to the main trial logic.
            pointer.opacity = 1
            pointer_blocker.opacity = 1
            outline.opacity = 1
            outline_bg.opacity = 1
            cross.opacity = 1
            rewardbar.opacity = 1
            # keep track of which components have finished
            instruction_reward_size_cueComponents = [blank_text_2]
            for thisComponent in instruction_reward_size_cueComponents:
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
            
            # --- Run Routine "instruction_reward_size_cue" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.6:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_text_2* updates
                
                # if blank_text_2 is starting this frame...
                if blank_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_text_2.frameNStart = frameN  # exact frame index
                    blank_text_2.tStart = t  # local t and not account for scr refresh
                    blank_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_text_2.started')
                    # update status
                    blank_text_2.status = STARTED
                    blank_text_2.setAutoDraw(True)
                
                # if blank_text_2 is active this frame...
                if blank_text_2.status == STARTED:
                    # update params
                    pass
                
                # if blank_text_2 is stopping this frame...
                if blank_text_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_text_2.tStartRefresh + 0.6-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_text_2.tStop = t  # not accounting for scr refresh
                        blank_text_2.tStopRefresh = tThisFlipGlobal  # on global time
                        blank_text_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_text_2.stopped')
                        # update status
                        blank_text_2.status = FINISHED
                        blank_text_2.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline_2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruction_reward_size_cueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruction_reward_size_cue" ---
            for thisComponent in instruction_reward_size_cueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instruction_reward_size_cue.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.600000)
            
            # --- Prepare to start Routine "instruction_decision_window" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instruction_decision_window.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from update_searchloc_2
            inner.depth = -3
            
            # ---------- probability path ----------
            
            # Set trial dynamics based on the tutorial type
            if current_tutorial_type == 'positive':
                p_end = 1.0   # Odds always go up to maximum
                snr_db = 51   # Very little noise
            
            elif current_tutorial_type == 'negative':
                p_end = 0.0   # Odds always go down to minimum
                snr_db = 51   # Very little noise
            
            elif current_tutorial_type == 'high_noise':
                p_end = 0.5   # Odds go to a medium value
                snr_db = 15   # A lot of noise
            
            # Now, recalculate the path using these scripted values
            p_lin = np.linspace(0.5, p_end, n_samples)
            noise = np.random.normal(0,
                                     np.sqrt(np.mean(p_lin**2)) / (10**(snr_db/20)),
                                     p_lin.shape)
            p_noisy = np.clip(p_lin + noise, 0, 1)
            
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
            
            # ---------- pointer setup ------------
            angle_step     = 360 / (n_samples-1)          # CCW 9 → 3 o’clock
            pointer.pos    = (0, 0)                       # anchored at centre
            
            current_i      = -1                           # “no flash yet”
            
            sweep_dur = flash_onsets[-1] + soa[-1]
            deg_per_sec       = sweep_span / sweep_dur  # constant angular speed
            # create starting attributes for give_response_2
            give_response_2.keys = []
            give_response_2.rt = []
            _give_response_2_allKeys = []
            # keep track of which components have finished
            instruction_decision_windowComponents = [give_response_2, reminder_text]
            for thisComponent in instruction_decision_windowComponents:
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
            
            # --- Run Routine "instruction_decision_window" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from update_searchloc_2
                # --- Pointer Movement ---
                pointer.ori = pointer_start_angle + (deg_per_sec * t)
                pointer_blocker.ori = pointer.ori
                rewardbar.ori = pointer.ori
                
                # --- Flash schedule logic ---
                # This code controls when the central dot appears and disappears
                
                # Start the next flash
                if current_i + 1 < n_samples and t >= flash_onsets[current_i + 1]:
                    current_i += 1
                    rewardbar.height = inner_radii[current_i]
                    rewardbar.opacity = 1
                    inner.size = (inner_radii[current_i], inner_radii[current_i])
                    inner.opacity = 1
                
                # End the current flash
                if current_i >= 0 and t >= flash_offsets[current_i]:
                    inner.opacity = 0
                    rewardbar.opacity = 0
                
                # End the trial if the sweep is complete
                if t > sum(soa):
                    continueRoutine = False
                # Run 'Each Frame' code from update_outline_cont1_2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # *give_response_2* updates
                waitOnFlip = False
                
                # if give_response_2 is starting this frame...
                if give_response_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    give_response_2.frameNStart = frameN  # exact frame index
                    give_response_2.tStart = t  # local t and not account for scr refresh
                    give_response_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(give_response_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'give_response_2.started')
                    # update status
                    give_response_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(give_response_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(give_response_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if give_response_2.status == STARTED and not waitOnFlip:
                    theseKeys = give_response_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _give_response_2_allKeys.extend(theseKeys)
                    if len(_give_response_2_allKeys):
                        give_response_2.keys = _give_response_2_allKeys[-1].name  # just the last key pressed
                        give_response_2.rt = _give_response_2_allKeys[-1].rt
                        give_response_2.duration = _give_response_2_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *reminder_text* updates
                
                # if reminder_text is starting this frame...
                if reminder_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    reminder_text.frameNStart = frameN  # exact frame index
                    reminder_text.tStart = t  # local t and not account for scr refresh
                    reminder_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(reminder_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reminder_text.started')
                    # update status
                    reminder_text.status = STARTED
                    reminder_text.setAutoDraw(True)
                
                # if reminder_text is active this frame...
                if reminder_text.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruction_decision_windowComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruction_decision_window" ---
            for thisComponent in instruction_decision_windowComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instruction_decision_window.stopped', globalClock.getTime(format='float'))
            # the Routine "instruction_decision_window" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instruction_hold" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instruction_hold.started', globalClock.getTime(format='float'))
            # keep track of which components have finished
            instruction_holdComponents = [blank_post_response_2]
            for thisComponent in instruction_holdComponents:
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
            
            # --- Run Routine "instruction_hold" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_post_response_2* updates
                
                # if blank_post_response_2 is starting this frame...
                if blank_post_response_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_post_response_2.frameNStart = frameN  # exact frame index
                    blank_post_response_2.tStart = t  # local t and not account for scr refresh
                    blank_post_response_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_post_response_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_post_response_2.started')
                    # update status
                    blank_post_response_2.status = STARTED
                    blank_post_response_2.setAutoDraw(True)
                
                # if blank_post_response_2 is active this frame...
                if blank_post_response_2.status == STARTED:
                    # update params
                    pass
                
                # if blank_post_response_2 is stopping this frame...
                if blank_post_response_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_post_response_2.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_post_response_2.tStop = t  # not accounting for scr refresh
                        blank_post_response_2.tStopRefresh = tThisFlipGlobal  # on global time
                        blank_post_response_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_post_response_2.stopped')
                        # update status
                        blank_post_response_2.status = FINISHED
                        blank_post_response_2.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline_cont2_2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruction_holdComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruction_hold" ---
            for thisComponent in instruction_holdComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instruction_hold.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "instruction_reward_reveal" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instruction_reward_reveal.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from calc_reward_reveal_time_2
            reward_reveal_time = 2.0 #changed from 1 --> 2, to allow to read added text output during instructions
            #jitter          = np.random.uniform(-0.1, +0.1)
            #reward_reveal_time += jitter
            # Run 'Begin Routine' code from calculate_reward_2
            # --- 1. Determine the outcome ---
            
            time_cutoff = sweep_dur / 2 
            reward = False  # Default to a loss
            
            # Check if a key was pressed AT ALL
            if not give_response_2.keys:
                # This is the case where the user let the time run out
                feedback_message = "You missed the chance to respond! Try to press SPACE while the clock is running."
                # 'reward' remains False
            else:
                # A key was pressed, so now check the timing
                rt = give_response_2.rt
            
                if current_tutorial_type == 'positive':
                    if rt > time_cutoff:
                        reward = True
                        feedback_message = "You waited for a better chance and you won. Easy money!"
                    else:
                        reward = False
                        feedback_message = "You went for it early and lost. When the indicator grows, it pays to be patient."
            
                elif current_tutorial_type == 'negative':
                    if rt < time_cutoff:
                        reward = True
                        feedback_message = "Great reaction! You won before the odds got worse."
                    else:
                        reward = False
                        feedback_message = "You waited too long as your odds became slim, and you lost."
            
                elif current_tutorial_type == 'high_noise':
                    if random.random() < 0.5:
                        reward = True
                        feedback_message = "You took a gamble in an unclear situation and it paid off!"
                    else:
                        reward = False
                        feedback_message = "You took a gamble in an unclear situation and were unlucky."
            
            # --- 2. Update the visual text components ---
            if reward:
                current_win_text_2.text = "+ 1"
            else:
                current_win_text_2.text = "+ 0"
            
            # This sets the text of your feedback_text component for the routine
            feedback_text.text = feedback_message
            # Run 'Begin Routine' code from hide_rewardbar_2
            rewardbar.opacity = 0
            cross.opacity = 0
            pointer.opacity = 0
            pointer_blocker.opacity = 0
            # keep track of which components have finished
            instruction_reward_revealComponents = [current_win_text_2, feedback_text]
            for thisComponent in instruction_reward_revealComponents:
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
            
            # --- Run Routine "instruction_reward_reveal" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *current_win_text_2* updates
                
                # if current_win_text_2 is starting this frame...
                if current_win_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    current_win_text_2.frameNStart = frameN  # exact frame index
                    current_win_text_2.tStart = t  # local t and not account for scr refresh
                    current_win_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(current_win_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'current_win_text_2.started')
                    # update status
                    current_win_text_2.status = STARTED
                    current_win_text_2.setAutoDraw(True)
                
                # if current_win_text_2 is active this frame...
                if current_win_text_2.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from calc_reward_reveal_time_2
                if t > reward_reveal_time:
                    continueRoutine = False
                # Run 'Each Frame' code from update_outline_cont3_2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # *feedback_text* updates
                
                # if feedback_text is starting this frame...
                if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text.frameNStart = frameN  # exact frame index
                    feedback_text.tStart = t  # local t and not account for scr refresh
                    feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text.started')
                    # update status
                    feedback_text.status = STARTED
                    feedback_text.setAutoDraw(True)
                
                # if feedback_text is active this frame...
                if feedback_text.status == STARTED:
                    # update params
                    feedback_text.setText(feedback_message, log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instruction_reward_revealComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instruction_reward_reveal" ---
            for thisComponent in instruction_reward_revealComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instruction_reward_reveal.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from calculate_reward_2
            # Hide all clock visuals AFTER the feedback has been shown,
            # ensuring the screen is clean for the next instruction page.
            pointer.setAutoDraw(False)
            pointer_blocker.setAutoDraw(False)
            outline.setAutoDraw(False)
            outline_bg.setAutoDraw(False)
            rewardbar.setAutoDraw(False)
            cross.setAutoDraw(False)
            inner.setAutoDraw(False)
            
            # Run 'End Routine' code from hide_rewardbar_2
            rewardbar.opacity = 1
            cross.opacity = 1
            # the Routine "instruction_reward_reveal" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed run_instruction_trial repeats of 'instruction_scripted_trials'
        
        
        # --- Prepare to start Routine "instruction_visualize_breaks" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('instruction_visualize_breaks.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        instruction_visualize_breaksComponents = []
        for thisComponent in instruction_visualize_breaksComponents:
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
        
        # --- Run Routine "instruction_visualize_breaks" ---
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
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruction_visualize_breaksComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruction_visualize_breaks" ---
        for thisComponent in instruction_visualize_breaksComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('instruction_visualize_breaks.stopped', globalClock.getTime(format='float'))
        # the Routine "instruction_visualize_breaks" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 99.0 repeats of 'tutorial_handler'
    
    
    # --- Prepare to start Routine "cleanup_instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('cleanup_instruction.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code
    # This is meant to keep the tutorial as a separate entity.
    # This could be moved to begin routine of the routine immediatly following the instruction_handler loop
    
    # --- Final Cleanup of all Tutorial Visuals ---
    # This ensures the tutorial leaves the screen completely clean.
    
    # Hide all visual components if they were created
    if 'pointer' in globals(): 
        pointer.setAutoDraw(False)
        pointer_blocker.setAutoDraw(False)
        outline.setAutoDraw(False)
        outline_bg.setAutoDraw(False)
        rewardbar.setAutoDraw(False)
        cross.setAutoDraw(False)
        inner.setAutoDraw(False)
    
    # Also ensure tutorial-specific text components are hidden
    if 'hint_text' in globals():
        hint_text.setAutoDraw(False)
    if 'feedback_text' in globals():
        feedback_text.setAutoDraw(False)
    
    # Make the routine end on the very next frame, making it instantaneous
    continueRoutine = False
    # keep track of which components have finished
    cleanup_instructionComponents = []
    for thisComponent in cleanup_instructionComponents:
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
    
    # --- Run Routine "cleanup_instruction" ---
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
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in cleanup_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "cleanup_instruction" ---
    for thisComponent in cleanup_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('cleanup_instruction.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "cleanup_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "resting_state" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state.started', globalClock.getTime(format='float'))
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
                fix_cross_resting_1.tStopRefresh = tThisFlipGlobal  # on global time
                fix_cross_resting_1.frameNStop = frameN  # exact frame index
                # update status
                fix_cross_resting_1.status = FINISHED
                fix_cross_resting_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
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
    thisExp.addData('resting_state.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "resting_state" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "TUS" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('TUS.started', globalClock.getTime(format='float'))
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
                fix_cross_TUS.tStopRefresh = tThisFlipGlobal  # on global time
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
            endExperiment(thisExp, win=win)
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
    thisExp.addData('TUS.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "resting_state" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state.started', globalClock.getTime(format='float'))
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
                fix_cross_resting_1.tStopRefresh = tThisFlipGlobal  # on global time
                fix_cross_resting_1.frameNStop = frameN  # exact frame index
                # update status
                fix_cross_resting_1.status = FINISHED
                fix_cross_resting_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
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
    thisExp.addData('resting_state.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "resting_state" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "task_instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('task_instruction.started', globalClock.getTime(format='float'))
    # create starting attributes for go_to_task
    go_to_task.keys = []
    go_to_task.rt = []
    _go_to_task_allKeys = []
    # keep track of which components have finished
    task_instructionComponents = [task_instruction_text, continue_to_task, go_to_task]
    for thisComponent in task_instructionComponents:
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
    
    # --- Run Routine "task_instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *task_instruction_text* updates
        
        # if task_instruction_text is starting this frame...
        if task_instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            task_instruction_text.frameNStart = frameN  # exact frame index
            task_instruction_text.tStart = t  # local t and not account for scr refresh
            task_instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(task_instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'task_instruction_text.started')
            # update status
            task_instruction_text.status = STARTED
            task_instruction_text.setAutoDraw(True)
        
        # if task_instruction_text is active this frame...
        if task_instruction_text.status == STARTED:
            # update params
            pass
        
        # *continue_to_task* updates
        
        # if continue_to_task is starting this frame...
        if continue_to_task.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            continue_to_task.frameNStart = frameN  # exact frame index
            continue_to_task.tStart = t  # local t and not account for scr refresh
            continue_to_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_to_task, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_to_task.started')
            # update status
            continue_to_task.status = STARTED
            continue_to_task.setAutoDraw(True)
        
        # if continue_to_task is active this frame...
        if continue_to_task.status == STARTED:
            # update params
            pass
        
        # *go_to_task* updates
        waitOnFlip = False
        
        # if go_to_task is starting this frame...
        if go_to_task.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            go_to_task.frameNStart = frameN  # exact frame index
            go_to_task.tStart = t  # local t and not account for scr refresh
            go_to_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(go_to_task, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'go_to_task.started')
            # update status
            go_to_task.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(go_to_task.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(go_to_task.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if go_to_task.status == STARTED and not waitOnFlip:
            theseKeys = go_to_task.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _go_to_task_allKeys.extend(theseKeys)
            if len(_go_to_task_allKeys):
                go_to_task.keys = _go_to_task_allKeys[-1].name  # just the last key pressed
                go_to_task.rt = _go_to_task_allKeys[-1].rt
                go_to_task.duration = _go_to_task_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "task_instruction" ---
    for thisComponent in task_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('task_instruction.stopped', globalClock.getTime(format='float'))
    # check responses
    if go_to_task.keys in ['', [], None]:  # No response was made
        go_to_task.keys = None
    thisExp.addData('go_to_task.keys',go_to_task.keys)
    if go_to_task.keys != None:  # we had a response
        thisExp.addData('go_to_task.rt', go_to_task.rt)
        thisExp.addData('go_to_task.duration', go_to_task.duration)
    thisExp.nextEntry()
    # the Routine "task_instruction" was not non-slip safe, so reset the non-slip timer
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
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
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
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
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
            thisExp.addData('trial_baseline.started', globalClock.getTime(format='float'))
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
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                radius = pointer_len 
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                outline.size = (1,1)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
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
            thisExp.addData('trial_baseline.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from setup_rewardbar
            rewardbar.ori = pointer_start_angle
            rewardbar.height = pointer_len*0.5
            rewardbar.opacity = 1
            # the Routine "trial_baseline" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "reward_size_cue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('reward_size_cue.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from setup_inner
            inner.setAutoDraw(False)
            inner_radius_start = inner_r_min + (inner_r_max-inner_r_min) * 0.5
            inner.size  = (inner_radius_start, inner_radius_start)
            inner.opacity = 1
            # Run 'Begin Routine' code from set_reward_size
            reward_size = random.choice(reward_sizes)
            
            bb.sendMarker(val=100+reward_size)
            
            inner.color = reward_size_colors[reward_size]
            rewardbar.color = reward_size_colors[reward_size]
            
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
                        blank_text.tStopRefresh = tThisFlipGlobal  # on global time
                        blank_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_text.stopped')
                        # update status
                        blank_text.status = FINISHED
                        blank_text.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
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
            thisExp.addData('reward_size_cue.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.600000)
            
            # --- Prepare to start Routine "decision_window" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('decision_window.started', globalClock.getTime(format='float'))
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
            
            # ---------- pointer setup ------------
            angle_step     = 360 / (n_samples-1)          # CCW 9 → 3 o’clock
            pointer.pos    = (0, 0)                       # anchored at centre
            
            current_i      = -1                           # “no flash yet”
            
            sweep_dur = flash_onsets[-1] + soa[-1]
            deg_per_sec       = sweep_span / sweep_dur  # constant angular speed
            # create starting attributes for give_response
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
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
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
            thisExp.addData('decision_window.stopped', globalClock.getTime(format='float'))
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
            thisExp.addData('hold.started', globalClock.getTime(format='float'))
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
                        blank_post_response.tStopRefresh = tThisFlipGlobal  # on global time
                        blank_post_response.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_post_response.stopped')
                        # update status
                        blank_post_response.status = FINISHED
                        blank_post_response.setAutoDraw(False)
                # Run 'Each Frame' code from update_outline_cont2
                progress_perc = global_clock.getTime()/task_time
                
                start_angle = np.pi / 2
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
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
            thisExp.addData('hold.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "reward_reveal" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('reward_reveal.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from calculate_reward
            # TODO: yes or no based on bubble size
            
            current_win_text.depth = -10
            
            if current_i + 1 > len(p_noisy):
                reward_prob = 0 # waited until end of trial
            else:
                reward_prob = p_noisy[current_i]
            
            rand_float = random.random()
            if rand_float > reward_prob:
                reward = False
                current_win_text.text = "+ 0" # ""
                bb.sendMarker(val=200)
            else:
                reward = True
                # reward_sound.play()
                current_win_text.text = f"+ {reward_size}"
                score += reward_size
                bb.sendMarker(val=210)
            # Run 'Begin Routine' code from calc_reward_reveal_time
            reward_reveal_time = 1.0
            jitter          = np.random.uniform(-0.1, +0.1)
            reward_reveal_time += jitter
            # Run 'Begin Routine' code from hide_rewardbar
            rewardbar.opacity = 0
            cross.opacity = 0
            pointer.opacity = 0
            pointer_blocker.opacity = 0
            # keep track of which components have finished
            reward_revealComponents = [current_win_text]
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
                end_angle = start_angle + 2 * np.pi * max(0, (1 - progress_perc))
                
                theta = np.linspace(start_angle, end_angle, 100)
                
                vertices = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
                
                outline.vertices = vertices
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
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
            thisExp.addData('reward_reveal.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from store_reward_info
            thisExp.addData('reward', reward)
            thisExp.addData('reward_size', reward_size)
            thisExp.addData('reward_chance_endpoint', p_end)
            thisExp.addData('snr_db', snr_db)
            
            thisExp.nextEntry()
            # Run 'End Routine' code from check_for_break
            # check if it is time for a break
            if (global_clock.getTime() - task_start_time) // block_dur > breaks_given:
                trials.finished = True
            # Run 'End Routine' code from hide_rewardbar
            rewardbar.opacity = 1
            cross.opacity = 1
            # Run 'End Routine' code from track_exp_time
            if global_clock.getTime() >= task_time:
                trials.finished = True
            # the Routine "reward_reveal" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 10000.0 repeats of 'trials'
        
        
        # --- Prepare to start Routine "break_relax" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_relax.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from update_break_text
        
            
        # Run 'Begin Routine' code from check_if_last
        breaks_given += 1
        
        if int(breaks_given) == int(n_blocks):
            continueRoutine = False
            breaks.finished = True
        # Run 'Begin Routine' code from hide_pointer_outline
        outline.opacity = 0
        outline_bg.opacity = 0
        rewardbar.opacity = 0
        # keep track of which components have finished
        break_relaxComponents = [break_text, score_text]
        for thisComponent in break_relaxComponents:
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
        
        # --- Run Routine "break_relax" ---
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
                break_text.setText('', log=False)
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + break_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.tStopRefresh = tThisFlipGlobal  # on global time
                    break_text.frameNStop = frameN  # exact frame index
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *score_text* updates
            
            # if score_text is starting this frame...
            if score_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                score_text.frameNStart = frameN  # exact frame index
                score_text.tStart = t  # local t and not account for scr refresh
                score_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(score_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'score_text.started')
                # update status
                score_text.status = STARTED
                score_text.setAutoDraw(True)
            
            # if score_text is active this frame...
            if score_text.status == STARTED:
                # update params
                score_text.setText('', log=False)
            
            # if score_text is stopping this frame...
            if score_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > score_text.tStartRefresh + break_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    score_text.tStop = t  # not accounting for scr refresh
                    score_text.tStopRefresh = tThisFlipGlobal  # on global time
                    score_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'score_text.stopped')
                    # update status
                    score_text.status = FINISHED
                    score_text.setAutoDraw(False)
            # Run 'Each Frame' code from update_break_text
            """ # TODO alignment did not work.
            break_text.anchorText = 'center'
            score_text.anchorText = 'center'
            break_text.alignText = 'center'
            score_text.alignText = 'right'
            
            break_text.text = 'Short break!\n\n\n\n\nContinuing in...\n' + str(int(break_dur + 1 - t))
            score_text.text = f'\n\nYour score: {score}\nComputer\'s score: {score_computer}\n\n\n'
            
            text_width = score_text.size[0]
            score_text.pos = (-text_width / 4, 0) # guesstimated division number, could not understand maths behind it - and still does not look good. drop.
            
            #"""
            
            break_text.text = f'Short break!\n\nYour score: \n{score}\nComputer\'s score: \n{score_computer}\n\nContinuing in...\n' + str(int(break_dur + 1 - t))
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_relaxComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_relax" ---
        for thisComponent in break_relaxComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_relax.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from hide_pointer_outline
        pointer.opacity = 1
        pointer_blocker.opacity = 1
        outline.opacity = 1
        outline_bg.opacity = 1
        cross.opacity = 1
        # the Routine "break_relax" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 4.0 repeats of 'breaks'
    
    
    # --- Prepare to start Routine "goodbye_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('goodbye_screen.started', globalClock.getTime(format='float'))
    # create starting attributes for quit_exp
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
            endExperiment(thisExp, win=win)
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
    thisExp.addData('goodbye_screen.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "goodbye_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


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


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
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
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
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
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
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
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
