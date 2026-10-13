#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on octubre 11, 2026, at 20:42
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
from psychopy.hardware.eyetracker import EyetrackerControl

# Run 'Before Experiment' code from code_start
import time
from datetime import datetime
import random
import pandas as pd
import math

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'Pilot'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
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
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

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
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Psychodevo\\Dropbox\\visual_illusion_task_adults 2 for github\\VI_task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
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
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
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
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='current', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Comenzando...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
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
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
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
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
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
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


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
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
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
    
    # --- Initialize components for Routine "hello" ---
    text = visual.TextStim(win=win, name='text',
        text='\nA continuación, te presentaremos una serie de imágenes.\n\nDebes elegir la imagen que mejor cumpla \nlo que la instrucción te indicará. \nEscucha con atención cada instrucción y \ntrata de responder tan rápido como puedas.',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.06, wrapWidth=10.0, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    buttonhello = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0,-0.25],
        letterHeight=0.08,
        size=[0.45,0.25], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='red', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonhello',
        depth=-1
    )
    buttonhello.buttonClock = core.Clock()
    # set audio backend
    sound.Sound.backend = 'ptb'
    sound_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_1'
    )
    sound_1.setVolume(1.0)
    # Run 'Begin Experiment' code from code_start
    def build_rows(
        path,
        total_trials,
        group_col="clase",
        variant_col="variant",
        pair_col="pair_id",
        group_quota=None,
    ):
        df = pd.read_excel(path)
        g = df[group_col].astype(str).str.strip().str.lower()
        v = df[variant_col].astype(str).str.strip().str.lower()
        p = df[pair_col].astype(str).str.strip().str.lower()
        df = df.assign(_group=g, _variant=v, _pair=p)
    
        picked_idx = []
        used_pairs = set()
    
        for gv, q in group_quota.items():
            # Handle odd quotas by randomly assigning the remainder (or splitting floor/ceil)
            need_N = math.ceil(q / 2) if random.random() < 0.5 else math.floor(q / 2)
            need_R = q - need_N
    
            sub = df[df._group == gv]
    
            # --- NORMAL VARIANT ---
            subN = sub[(sub._variant == "normal") & (~sub._pair.isin(used_pairs))]
            pairsN = subN._pair.drop_duplicates().tolist()
            random.shuffle(pairsN)
            chosenN = pairsN[:need_N]
    
            if chosenN:
                pickN = subN[subN._pair.isin(chosenN)].groupby(
                    "_pair", group_keys=False
                ).apply(lambda x: x.sample(n=1, random_state=None))
                used_pairs.update(chosenN)
                picked_idx.extend(pickN.index.tolist())
    
            # --- REVERSED VARIANT ---
            subR = sub[(sub._variant == "reversed") & (~sub._pair.isin(used_pairs))]
            pairsR = subR._pair.drop_duplicates().tolist()
            random.shuffle(pairsR)
            chosenR = pairsR[:need_R]
    
            if chosenR:
                pickR = subR[subR._pair.isin(chosenR)].groupby(
                    "_pair", group_keys=False
                ).apply(lambda x: x.sample(n=1, random_state=None))
                used_pairs.update(chosenR)
                picked_idx.extend(pickR.index.tolist())
    
        picked = df.loc[picked_idx].sample(frac=1.0, random_state=None)
        return ",".join(map(str, picked.index.tolist()))
        
    
    set_rows_ML = build_rows(
        "Stimuli/Müller-Lyer trials/ML_condition_file.xlsx",
        total_trials=18,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'control_inward': 6, 'control_outward': 6,  'misleading': 6}
    )
    
    set_rows_contrast = build_rows(
        "Stimuli/Contrast trials/Contrast_condition_file.xlsx",
        total_trials=15,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'control': 10, 'misleading': 5}
    )
    
    
    set_rows_ebbinghaus = build_rows(
        "Stimuli/Ebbinghaus trials/Ebbinghaus_condition_file.xlsx",
        total_trials=18,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'control_large': 6, 'control_small': 6,  'misleading': 6}
    )
    
    set_rows_kanizsa = build_rows(
        "Stimuli/Kanizsa trials/KN_condition_file.xlsx",
        total_trials=25,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'illusion_10': 5, 'illusion_20': 5, 'illusion_30': 5, 'illusion_40': 5,'illusion_50': 5}
    )
    
    
    set_rows_delboeuf = build_rows(
        "Stimuli/Delboeuf trials/Delboeuf_condition_file.xlsx",
        total_trials=18,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'control_large': 6, 'control_small': 6, 'misleading': 6}
    )
    
    
    set_rows_solitaire = build_rows(
        "Stimuli/Solitaire trials/Solitaire_condition_file.xlsx",
        total_trials=16,
        group_col='name',
        variant_col='variant',
        pair_col='pair_id',
        group_quota={'control':2, 'misleading': 14}
    )
    
    print("[DEBUG] ML rows   ->", set_rows_ML)
    print("[DEBUG] Contrast rows   ->", set_rows_contrast)
    print("[DEBUG] Ebbinghaus rows   ->", set_rows_ebbinghaus)
    print("[DEBUG] Kanizsa rows   ->", set_rows_kanizsa)
    print("[DEBUG] Delboeuf rows   ->", set_rows_delboeuf)
    print("[DEBUG] Solitaire rows   ->", set_rows_solitaire)
    
    
    
    # --- Initialize components for Routine "hello_training" ---
    hellotraining_poly = visual.ShapeStim(
        win=win, name='hellotraining_poly',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, -0.20), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    hellomouse = event.Mouse(win=win)
    x, y = [None, None]
    hellomouse.mouseClock = core.Clock()
    hellotraining = visual.TextStim(win=win, name='hellotraining',
        text='Pero primero, \n¡un poco de entrenamiento!\n\n\nHaz click sobre el CÍRCULO ROJO',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-0.4902, -0.1765, 0.7647], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "longerline_1" ---
    longerline1pic = visual.ImageStim(
        win=win,
        name='longerline1pic', 
        image='Stimuli/Training/longerline1.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[2,1],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    longerline1area = visual.Rect(
        win=win, name='longerline1area',
        width=(0.8, 0.32)[0], height=(0.8, 0.32)[1],
        ori=0.0, pos=[-0.23, -0.30], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=None, fillColor=None,
        opacity=None, depth=-1.0, interpolate=True)
    Longerlinetext1 = visual.TextBox2(
         win, text='Elige la LINEA MÁS LARGA!', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Longerlinetext1',
         depth=-2, autoLog=True,
    )
    longerlinemouse1 = event.Mouse(win=win)
    x, y = [None, None]
    longerlinemouse1.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "longerline_2" ---
    longerline2pic = visual.ImageStim(
        win=win,
        name='longerline2pic', 
        image='Stimuli/Training/longerline2.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[1.9,1],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    longerline2area = visual.Rect(
        win=win, name='longerline2area',
        width=(0.72, 0.37)[0], height=(0.72, 0.37)[1],
        ori=0.0, pos=[0.5, 0.24], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=None, fillColor=None,
        opacity=None, depth=-1.0, interpolate=True)
    Longerlinetext2 = visual.TextBox2(
         win, text='Elige la LINEA MÁS LARGA!', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Longerlinetext2',
         depth=-2, autoLog=True,
    )
    longerline2mouse = event.Mouse(win=win)
    x, y = [None, None]
    longerline2mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "longerline_3" ---
    longerline3pic = visual.ImageStim(
        win=win,
        name='longerline3pic', 
        image='Stimuli/Training/longerline3.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[1.9,1],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    longerline3area = visual.Rect(
        win=win, name='longerline3area',
        width=(1, 0.17)[0], height=(1, 0.17)[1],
        ori=0.0, pos=[0.49, -0.3], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=None, fillColor=None,
        opacity=None, depth=-1.0, interpolate=True)
    Longerlinetext3 = visual.TextBox2(
         win, text='Elige la LINEA MÁS LARGA!', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Longerlinetext3',
         depth=-2, autoLog=True,
    )
    longerline3mouse = event.Mouse(win=win)
    x, y = [None, None]
    longerline3mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle1" ---
    circle_click_text1 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1',
         depth=0, autoLog=True,
    )
    circle_click = visual.ShapeStim(
        win=win, name='circle_click',
        size=(0.35, 0.35), vertices='circle',
        ori=0.0, pos=(0.25, 0.22), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=-1.0, interpolate=True)
    circle_mouse1 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse1.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle2" ---
    circle_click_2 = visual.ShapeStim(
        win=win, name='circle_click_2',
        size=(0.28, 0.28), vertices='circle',
        ori=0.0, pos=(-0.3, 0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_2 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_2',
         depth=-1, autoLog=True,
    )
    circle_mouse2 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle3" ---
    circle_click_3 = visual.ShapeStim(
        win=win, name='circle_click_3',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0.2, -0.45), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_3 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_3',
         depth=-1, autoLog=True,
    )
    circle_mouse3 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle4" ---
    circle_click_4 = visual.ShapeStim(
        win=win, name='circle_click_4',
        size=(0.1, 0.1), vertices='circle',
        ori=0.0, pos=(0.45, 0.4), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_4 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_4',
         depth=-1, autoLog=True,
    )
    circle_mouse4 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse4.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle5" ---
    circle_click_5 = visual.ShapeStim(
        win=win, name='circle_click_5',
        size=(0.075, 0.075), vertices='circle',
        ori=0.0, pos=(-0.25, 0.1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_5 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_5',
         depth=-1, autoLog=True,
    )
    circle_mouse5 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse5.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle6" ---
    circle_click_6 = visual.ShapeStim(
        win=win, name='circle_click_6',
        size=(0.075, 0.075), vertices='circle',
        ori=0.0, pos=(-0.45, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_6 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_6',
         depth=-1, autoLog=True,
    )
    circle_mouse6 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse6.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle7" ---
    circle_click_7 = visual.ShapeStim(
        win=win, name='circle_click_7',
        size=(0.075, 0.075), vertices='circle',
        ori=0.0, pos=(-0.4, 0.36), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_7 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_7',
         depth=-1, autoLog=True,
    )
    circle_mouse7 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse7.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle8" ---
    circle_click_8 = visual.ShapeStim(
        win=win, name='circle_click_8',
        size=(0.055, 0.055), vertices='circle',
        ori=0.0, pos=(0.1, 0.38), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_8 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_8',
         depth=-1, autoLog=True,
    )
    circle_mouse8 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse8.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle9" ---
    circle_click_9 = visual.ShapeStim(
        win=win, name='circle_click_9',
        size=(0.055, 0.055), vertices='circle',
        ori=0.0, pos=(0.4, -0.42), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_9 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_9',
         depth=-1, autoLog=True,
    )
    circle_mouse9 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse9.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "clickcircle10" ---
    circle_click_10 = visual.ShapeStim(
        win=win, name='circle_click_10',
        size=(0.055, 0.055), vertices='circle',
        ori=0.0, pos=(0.43, 0.42), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    circle_click_text1_10 = visual.TextBox2(
         win, text='Haz click sobre el CÍRCULO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.45], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='circle_click_text1_10',
         depth=-1, autoLog=True,
    )
    circle_mouse10 = event.Mouse(win=win)
    x, y = [None, None]
    circle_mouse10.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "ML_INTRO" ---
    ml_intro_text = visual.TextStim(win=win, name='ml_intro_text',
        text='\nDe las dos líneas que observarás \na continuación, \nelige la ¡MÁS LARGA!\n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color=[-0.4902, -0.1765, 0.7647], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_2'
    )
    sound_2.setVolume(1.0)
    START_ML = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.05,
        size=[0.3,0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor=[1.0000, -0.4588, -1.0000], borderColor=[-1.0000, -1.0000, -1.0000],
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_ML',
        depth=-2
    )
    START_ML.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_ML" ---
    image_ML = visual.ImageStim(
        win=win,
        name='image_ML', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    LEFT_ML = visual.Rect(
        win=win, name='LEFT_ML',
        width=[0.60, 0.25][0], height=[0.60, 0.25][1],
        ori=0.0, pos=[-0.40,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    RIGHT_ML = visual.Rect(
        win=win, name='RIGHT_ML',
        width=[0.60, 0.25][0], height=[0.60, 0.25][1],
        ori=0.0, pos=[0.40, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_ML = event.Mouse(win=win)
    x, y = [None, None]
    mouse_ML.mouseClock = core.Clock()
    sound_3 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_3'
    )
    sound_3.setVolume(1.0)
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Contrast_illusion_INTRO" ---
    contrast_intro_text = visual.TextStim(win=win, name='contrast_intro_text',
        text='\nSelecciona la imagen en la \nque el círculo sea\n¡MÁS OSCURO!\n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_4 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_4'
    )
    sound_4.setVolume(1.0)
    START_contrast = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.05,
        size=[0.3,0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_contrast',
        depth=-2
    )
    START_contrast.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_contrast" ---
    image_cont = visual.ImageStim(
        win=win,
        name='image_cont', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    LEFT_cont = visual.Rect(
        win=win, name='LEFT_cont',
        width=[0.75, 0.5][0], height=[0.75, 0.5][1],
        ori=0.0, pos=[-0.40,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    RIGHT_cont = visual.Rect(
        win=win, name='RIGHT_cont',
        width=[0.75, 0.5][0], height=[0.75, 0.5][1],
        ori=0.0, pos=[0.40, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_cont = event.Mouse(win=win)
    x, y = [None, None]
    mouse_cont.mouseClock = core.Clock()
    sound_8 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_8'
    )
    sound_8.setVolume(1.0)
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Ebbing_INTRO" ---
    ebbing_intro_text = visual.TextStim(win=win, name='ebbing_intro_text',
        text='\nSelecciona el círculo\n naranja de \n¡MAYOR TAMAÑO!\n\n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_5 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_5'
    )
    sound_5.setVolume(1.0)
    START_ebbing = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_ebbing',
        depth=-2
    )
    START_ebbing.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_ebbing" ---
    image_ebbing = visual.ImageStim(
        win=win,
        name='image_ebbing', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    RIGHT_ebbing = visual.Rect(
        win=win, name='RIGHT_ebbing',
        width=[0.54, 0.54][0], height=[0.54, 0.54][1],
        ori=0.0, pos=[0.36, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    LEFT_ebbing = visual.Rect(
        win=win, name='LEFT_ebbing',
        width=[0.54, 0.54][0], height=[0.54, 0.54][1],
        ori=0.0, pos=[-0.36,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_ebbing = event.Mouse(win=win)
    x, y = [None, None]
    mouse_ebbing.mouseClock = core.Clock()
    sound_9 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_9'
    )
    sound_9.setVolume(1.0)
    key_resp_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Kanizsa_INTRO" ---
    kanizsa_intro_text = visual.TextStim(win=win, name='kanizsa_intro_text',
        text='\nDe las imágenes que \naparecerán a continuación,\n elige aquella donde se forme un \nCUADRADO BLANCO',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_6 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_6'
    )
    sound_6.setVolume(1.0)
    START_KN = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0, -0.3],
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_KN',
        depth=-2
    )
    START_KN.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "trainingKZS" ---
    triangle = visual.ImageStim(
        win=win,
        name='triangle', 
        image='Stimuli/Training/triang.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, 0.25), draggable=False, size=(0.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    threefourth = visual.ImageStim(
        win=win,
        name='threefourth', 
        image='Stimuli/Training/whitesquare3p4.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, 0.25), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    twofourth = visual.ImageStim(
        win=win,
        name='twofourth', 
        image='Stimuli/Training/whitesquare2p4.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.25), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    whitesquare = visual.ImageStim(
        win=win,
        name='whitesquare', 
        image='Stimuli/Training/whitesquare.png', mask=None, anchor='center',
        ori=0.0, pos=(0.5, -0.15), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "trainingKZS_2" ---
    triangle_2 = visual.ImageStim(
        win=win,
        name='triangle_2', 
        image='Stimuli/Training/triang.png', mask=None, anchor='center',
        ori=23.0, pos=[-0.3, -0.25], draggable=False, size=[0.5,0.5],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    threefourth_2 = visual.ImageStim(
        win=win,
        name='threefourth_2', 
        image='Stimuli/Training/whitesquare3p4.png', mask=None, anchor='center',
        ori=350.0, pos=(0.5, -0.15), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    twofourth_2 = visual.ImageStim(
        win=win,
        name='twofourth_2', 
        image='Stimuli/Training/whitesquare2p4.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, 0.25), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    whitesquare_2 = visual.ImageStim(
        win=win,
        name='whitesquare_2', 
        image='Stimuli/Training/whitesquare.png', mask=None, anchor='center',
        ori=45.0, pos=(-0.5, 0.25), draggable=False, size=(0.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_knzs" ---
    image_KN = visual.ImageStim(
        win=win,
        name='image_KN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    LEFT_KN = visual.Rect(
        win=win, name='LEFT_KN',
        width=[0.8, 0.8][0], height=[0.8, 0.8][1],
        ori=0.0, pos=[-0.44,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    RIGHT_KN = visual.Rect(
        win=win, name='RIGHT_KN',
        width=[0.8, 0.8][0], height=[0.8, 0.8][1],
        ori=0.0, pos=[0.44, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_KN = event.Mouse(win=win)
    x, y = [None, None]
    mouse_KN.mouseClock = core.Clock()
    sound_11 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_11'
    )
    sound_11.setVolume(1.0)
    key_resp_5 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "MS_INTRO_2" ---
    MS_2_intro_text = visual.TextStim(win=win, name='MS_2_intro_text',
        text='\nSi logras percibir movimiento \nen la siguiente imagen \nselecciona “SI”,\n de lo contrario \nselecciona “NO”. \n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_19 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_19'
    )
    sound_19.setVolume(1.0)
    START_MS_2 = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0, -0.3],
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_MS_2',
        depth=-2
    )
    START_MS_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_MS_2" ---
    image_MS_2 = visual.ImageStim(
        win=win,
        name='image_MS_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    mouse_MS_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_MS_2.mouseClock = core.Clock()
    sound_18 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_18'
    )
    sound_18.setVolume(1.0)
    key_resp_10 = keyboard.Keyboard(deviceName='defaultKeyboard')
    button_YES = visual.ButtonStim(win, 
        text='SI', font='Arvo',
        pos=(-0.55, -0.35),
        letterHeight=0.05,
        size=(0.15, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='red', borderColor=[-1.0000, -1.0000, -1.0000],
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_YES',
        depth=-4
    )
    button_YES.buttonClock = core.Clock()
    button_NO = visual.ButtonStim(win, 
        text='NO', font='Arvo',
        pos=(0.55, -0.34),
        letterHeight=0.05,
        size=(0.15, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='red', borderColor=[-1.0000, -1.0000, -1.0000],
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_NO',
        depth=-5
    )
    button_NO.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "DELBOUF_INTRO" ---
    DELBOUF_intro_text = visual.TextStim(win=win, name='DELBOUF_intro_text',
        text='\nSelecciona la imagen en la que el \ncirculo negro central\n que tenga\n ¡MAYOR TAMAÑO!',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_12 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_12'
    )
    sound_12.setVolume(1.0)
    START_DELBOUF = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0, -0.3],
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_DELBOUF',
        depth=-2
    )
    START_DELBOUF.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_DELBOUF" ---
    image_DELBOUF = visual.ImageStim(
        win=win,
        name='image_DELBOUF', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    LEFT_DELBOUF = visual.Rect(
        win=win, name='LEFT_DELBOUF',
        width=[0.72, 0.72][0], height=[0.72, 0.72][1],
        ori=0.0, pos=[-0.38,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    RIGHT_DELBOUF = visual.Rect(
        win=win, name='RIGHT_DELBOUF',
        width=[0.72, 0.72][0], height=[0.72, 0.72][1],
        ori=0.0, pos=[0.38, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_DELBOUF = event.Mouse(win=win)
    x, y = [None, None]
    mouse_DELBOUF.mouseClock = core.Clock()
    sound_13 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_13'
    )
    sound_13.setVolume(1.0)
    key_resp_7 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "PONZO_INTRO" ---
    PONZO_intro_text = visual.TextStim(win=win, name='PONZO_intro_text',
        text='\nElige la linea naranja\n¡MAS LARGA!\n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_15 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_15'
    )
    sound_15.setVolume(1.0)
    START_PONZO = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0, -0.3],
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_PONZO',
        depth=-2
    )
    START_PONZO.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_PONZO" ---
    image_PONZO = visual.ImageStim(
        win=win,
        name='image_PONZO', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    UP_PONZO = visual.Rect(
        win=win, name='UP_PONZO',
        width=(0.8, 0.25)[0], height=(0.8, 0.25)[1],
        ori=0.0, pos=(0,0.25), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    DOWN_PONZO = visual.Rect(
        win=win, name='DOWN_PONZO',
        width=(0.8, 0.25)[0], height=(0.8, 0.25)[1],
        ori=0.0, pos=(0,-0.25), draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_PONZO = event.Mouse(win=win)
    x, y = [None, None]
    mouse_PONZO.mouseClock = core.Clock()
    sound_14 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_14'
    )
    sound_14.setVolume(1.0)
    key_resp_8 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "SOLITAIRE_INTRO" ---
    SOLITAIRE_intro_text = visual.TextStim(win=win, name='SOLITAIRE_intro_text',
        text='\nElige la imagen que contenga\n la MAYOR \ncantidad de\n círculos negros.\n',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=20.0, ori=0.0, 
        color='royalblue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_17 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_17'
    )
    sound_17.setVolume(1.0)
    START_SOLITAIRE = visual.ButtonStim(win, 
        text='INICIAR', font='Arvo',
        pos=[0, -0.3],
        letterHeight=0.05,
        size=[0.3, 0.15], 
        ori=0.0
        ,borderWidth=1.0,
        fillColor='orangered', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='START_SOLITAIRE',
        depth=-2
    )
    START_SOLITAIRE.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "TEST_SOLITAIRE" ---
    image_SOLITAIRE = visual.ImageStim(
        win=win,
        name='image_SOLITAIRE', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    LEFT_SOLITAIRE = visual.Rect(
        win=win, name='LEFT_SOLITAIRE',
        width=[0.7, 0.7][0], height=[0.7, 0.7][1],
        ori=0.0, pos=[-0.4,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor=None,
        opacity=0.0, depth=-1.0, interpolate=True)
    RIGHT_SOLITAIRE = visual.Rect(
        win=win, name='RIGHT_SOLITAIRE',
        width=[0.7, 0.7][0], height=[0.7, 0.7][1],
        ori=0.0, pos=[0.4, 0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor=None,
        opacity=0.0, depth=-2.0, interpolate=True)
    mouse_SOLITAIRE = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SOLITAIRE.mouseClock = core.Clock()
    sound_16 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_16'
    )
    sound_16.setVolume(1.0)
    key_resp_9 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pressspace" ---
    Next_text = visual.TextBox2(
         win, text='Para ver la siguiente imagen\nHaz click sobre el CÍRCULO ROJO', placeholder='Type here...', font='Calibri',
         ori=0.0, pos=[0, 0.3], draggable=False,      letterHeight=0.06,
         size=(None, None), borderWidth=2.0,
         color=[-1.0000, -1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='Next_text',
         depth=0, autoLog=True,
    )
    Next_polygon = visual.ShapeStim(
        win=win, name='Next_polygon',
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    Next_mouse = event.Mouse(win=win)
    x, y = [None, None]
    Next_mouse.mouseClock = core.Clock()
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "bye" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='¡GRACIAS!',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=[-0.4902, -0.1765, 0.7647], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
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
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "hello" ---
    # create an object to store info about Routine hello
    hello = data.Routine(
        name='hello',
        components=[text, buttonhello, sound_1],
    )
    hello.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset buttonhello to account for continued clicks & clear times on/off
    buttonhello.reset()
    sound_1.setSound('Stimuli/Instructions/INTRO.ogg', hamming=True)
    sound_1.setVolume(1.0, log=False)
    sound_1.seek(0)
    # store start times for hello
    hello.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hello.tStart = globalClock.getTime(format='float')
    hello.status = STARTED
    thisExp.addData('hello.started', hello.tStart)
    hello.maxDuration = None
    # keep track of which components have finished
    helloComponents = hello.components
    for thisComponent in hello.components:
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
    
    # --- Run Routine "hello" ---
    thisExp.currentRoutine = hello
    hello.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        # *buttonhello* updates
        
        # if buttonhello is starting this frame...
        if buttonhello.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            buttonhello.frameNStart = frameN  # exact frame index
            buttonhello.tStart = t  # local t and not account for scr refresh
            buttonhello.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(buttonhello, 'tStartRefresh')  # time at next scr refresh
            # update status
            buttonhello.status = STARTED
            win.callOnFlip(buttonhello.buttonClock.reset)
            buttonhello.setAutoDraw(True)
        
        # if buttonhello is active this frame...
        if buttonhello.status == STARTED:
            # update params
            pass
            # check whether buttonhello has been pressed
            if buttonhello.isClicked:
                if not buttonhello.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    buttonhello.timesOn.append(globalClock.getTime())
                    buttonhello.timesOff.append(globalClock.getTime())
                elif len(buttonhello.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    buttonhello.timesOff[-1] = globalClock.getTime()
                if not buttonhello.wasClicked:
                    # end routine when buttonhello is clicked
                    continueRoutine = False
                if not buttonhello.wasClicked:
                    # run callback code when buttonhello is clicked
                    pass
        # take note of whether buttonhello was clicked, so that next frame we know if clicks are new
        buttonhello.wasClicked = buttonhello.isClicked and buttonhello.status == STARTED
        
        # *sound_1* updates
        
        # if sound_1 is starting this frame...
        if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sound_1.frameNStart = frameN  # exact frame index
            sound_1.tStart = t  # local t and not account for scr refresh
            sound_1.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            sound_1.status = STARTED
            sound_1.play(when=win)  # sync with win flip
        
        # if sound_1 is stopping this frame...
        if sound_1.status == STARTED:
            if bool(False) or sound_1.isFinished:
                # keep track of stop time/frame for later
                sound_1.tStop = t  # not accounting for scr refresh
                sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                sound_1.frameNStop = frameN  # exact frame index
                # update status
                sound_1.status = FINISHED
                sound_1.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=hello,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            hello.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if hello.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in hello.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hello" ---
    for thisComponent in hello.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hello
    hello.tStop = globalClock.getTime(format='float')
    hello.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hello.stopped', hello.tStop)
    thisExp.addData('buttonhello.numClicks', buttonhello.numClicks)
    if buttonhello.numClicks:
       thisExp.addData('buttonhello.timesOn', buttonhello.timesOn)
       thisExp.addData('buttonhello.timesOff', buttonhello.timesOff)
    else:
       thisExp.addData('buttonhello.timesOn', "")
       thisExp.addData('buttonhello.timesOff', "")
    sound_1.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "hello" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "hello_training" ---
    # create an object to store info about Routine hello_training
    hello_training = data.Routine(
        name='hello_training',
        components=[hellotraining_poly, hellomouse, hellotraining],
    )
    hello_training.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the hellomouse
    hellomouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for hello_training
    hello_training.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hello_training.tStart = globalClock.getTime(format='float')
    hello_training.status = STARTED
    thisExp.addData('hello_training.started', hello_training.tStart)
    hello_training.maxDuration = None
    # keep track of which components have finished
    hello_trainingComponents = hello_training.components
    for thisComponent in hello_training.components:
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
    
    # --- Run Routine "hello_training" ---
    thisExp.currentRoutine = hello_training
    hello_training.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *hellotraining_poly* updates
        
        # if hellotraining_poly is starting this frame...
        if hellotraining_poly.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hellotraining_poly.frameNStart = frameN  # exact frame index
            hellotraining_poly.tStart = t  # local t and not account for scr refresh
            hellotraining_poly.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hellotraining_poly, 'tStartRefresh')  # time at next scr refresh
            # update status
            hellotraining_poly.status = STARTED
            hellotraining_poly.setAutoDraw(True)
        
        # if hellotraining_poly is active this frame...
        if hellotraining_poly.status == STARTED:
            # update params
            pass
        # *hellomouse* updates
        
        # if hellomouse is starting this frame...
        if hellomouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hellomouse.frameNStart = frameN  # exact frame index
            hellomouse.tStart = t  # local t and not account for scr refresh
            hellomouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hellomouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('hellomouse.started', t)
            # update status
            hellomouse.status = STARTED
            prevButtonState = hellomouse.getPressed()  # if button is down already this ISN'T a new click
        if hellomouse.status == STARTED:  # only update if started and not finished!
            buttons = hellomouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(hellotraining_poly, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(hellomouse):
                            gotValidClick = True
                            hellomouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        hellomouse.clicked_name.append(None)
                    if gotValidClick:  
                        continueRoutine = False  # end routine on response
        
        # *hellotraining* updates
        
        # if hellotraining is starting this frame...
        if hellotraining.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hellotraining.frameNStart = frameN  # exact frame index
            hellotraining.tStart = t  # local t and not account for scr refresh
            hellotraining.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hellotraining, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hellotraining.started')
            # update status
            hellotraining.status = STARTED
            hellotraining.setAutoDraw(True)
        
        # if hellotraining is active this frame...
        if hellotraining.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=hello_training,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            hello_training.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if hello_training.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in hello_training.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hello_training" ---
    for thisComponent in hello_training.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hello_training
    hello_training.tStop = globalClock.getTime(format='float')
    hello_training.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hello_training.stopped', hello_training.tStop)
    # store data for thisExp (ExperimentHandler)
    x, y = hellomouse.getPos()
    buttons = hellomouse.getPressed()
    if sum(buttons):
        # check if the mouse was inside our 'clickable' objects
        gotValidClick = False
        clickableList = environmenttools.getFromNames(hellotraining_poly, namespace=locals())
        for obj in clickableList:
            # is this object clicked on?
            if obj.contains(hellomouse):
                gotValidClick = True
                hellomouse.clicked_name.append(obj.name)
        if not gotValidClick:
            hellomouse.clicked_name.append(None)
    thisExp.addData('hellomouse.x', x)
    thisExp.addData('hellomouse.y', y)
    thisExp.addData('hellomouse.leftButton', buttons[0])
    thisExp.addData('hellomouse.midButton', buttons[1])
    thisExp.addData('hellomouse.rightButton', buttons[2])
    if len(hellomouse.clicked_name):
        thisExp.addData('hellomouse.clicked_name', hellomouse.clicked_name[0])
    thisExp.nextEntry()
    # the Routine "hello_training" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "longerline_1" ---
    # create an object to store info about Routine longerline_1
    longerline_1 = data.Routine(
        name='longerline_1',
        components=[longerline1pic, longerline1area, Longerlinetext1, longerlinemouse1],
    )
    longerline_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    Longerlinetext1.reset()
    # setup some python lists for storing info about the longerlinemouse1
    longerlinemouse1.x = []
    longerlinemouse1.y = []
    longerlinemouse1.leftButton = []
    longerlinemouse1.midButton = []
    longerlinemouse1.rightButton = []
    longerlinemouse1.time = []
    longerlinemouse1.clicked_name = []
    gotValidClick = False  # until a click is received
    longerlinemouse1.mouseClock.reset()
    # store start times for longerline_1
    longerline_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    longerline_1.tStart = globalClock.getTime(format='float')
    longerline_1.status = STARTED
    thisExp.addData('longerline_1.started', longerline_1.tStart)
    longerline_1.maxDuration = None
    # keep track of which components have finished
    longerline_1Components = longerline_1.components
    for thisComponent in longerline_1.components:
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
    
    # --- Run Routine "longerline_1" ---
    thisExp.currentRoutine = longerline_1
    longerline_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *longerline1pic* updates
        
        # if longerline1pic is starting this frame...
        if longerline1pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline1pic.frameNStart = frameN  # exact frame index
            longerline1pic.tStart = t  # local t and not account for scr refresh
            longerline1pic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline1pic, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline1pic.status = STARTED
            longerline1pic.setAutoDraw(True)
        
        # if longerline1pic is active this frame...
        if longerline1pic.status == STARTED:
            # update params
            pass
        
        # *longerline1area* updates
        
        # if longerline1area is starting this frame...
        if longerline1area.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline1area.frameNStart = frameN  # exact frame index
            longerline1area.tStart = t  # local t and not account for scr refresh
            longerline1area.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline1area, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline1area.status = STARTED
            longerline1area.setAutoDraw(True)
        
        # if longerline1area is active this frame...
        if longerline1area.status == STARTED:
            # update params
            pass
        
        # *Longerlinetext1* updates
        
        # if Longerlinetext1 is starting this frame...
        if Longerlinetext1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Longerlinetext1.frameNStart = frameN  # exact frame index
            Longerlinetext1.tStart = t  # local t and not account for scr refresh
            Longerlinetext1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Longerlinetext1, 'tStartRefresh')  # time at next scr refresh
            # update status
            Longerlinetext1.status = STARTED
            Longerlinetext1.setAutoDraw(True)
        
        # if Longerlinetext1 is active this frame...
        if Longerlinetext1.status == STARTED:
            # update params
            pass
        # *longerlinemouse1* updates
        
        # if longerlinemouse1 is starting this frame...
        if longerlinemouse1.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerlinemouse1.frameNStart = frameN  # exact frame index
            longerlinemouse1.tStart = t  # local t and not account for scr refresh
            longerlinemouse1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerlinemouse1, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerlinemouse1.status = STARTED
            prevButtonState = longerlinemouse1.getPressed()  # if button is down already this ISN'T a new click
        if longerlinemouse1.status == STARTED:  # only update if started and not finished!
            buttons = longerlinemouse1.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(longerline1area, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(longerlinemouse1):
                            gotValidClick = True
                            longerlinemouse1.clicked_name.append(obj.name)
                            longerlinemouse1.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = longerlinemouse1.getPos()
                        longerlinemouse1.x.append(float(x))
                        longerlinemouse1.y.append(float(y))
                        buttons = longerlinemouse1.getPressed()
                        longerlinemouse1.leftButton.append(buttons[0])
                        longerlinemouse1.midButton.append(buttons[1])
                        longerlinemouse1.rightButton.append(buttons[2])
                        longerlinemouse1.time.append(longerlinemouse1.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=longerline_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            longerline_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if longerline_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in longerline_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "longerline_1" ---
    for thisComponent in longerline_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for longerline_1
    longerline_1.tStop = globalClock.getTime(format='float')
    longerline_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('longerline_1.stopped', longerline_1.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('longerlinemouse1.x', longerlinemouse1.x)
    thisExp.addData('longerlinemouse1.y', longerlinemouse1.y)
    thisExp.addData('longerlinemouse1.leftButton', longerlinemouse1.leftButton)
    thisExp.addData('longerlinemouse1.midButton', longerlinemouse1.midButton)
    thisExp.addData('longerlinemouse1.rightButton', longerlinemouse1.rightButton)
    thisExp.addData('longerlinemouse1.time', longerlinemouse1.time)
    thisExp.addData('longerlinemouse1.clicked_name', longerlinemouse1.clicked_name)
    thisExp.nextEntry()
    # the Routine "longerline_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "longerline_2" ---
    # create an object to store info about Routine longerline_2
    longerline_2 = data.Routine(
        name='longerline_2',
        components=[longerline2pic, longerline2area, Longerlinetext2, longerline2mouse],
    )
    longerline_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    Longerlinetext2.reset()
    # setup some python lists for storing info about the longerline2mouse
    longerline2mouse.x = []
    longerline2mouse.y = []
    longerline2mouse.leftButton = []
    longerline2mouse.midButton = []
    longerline2mouse.rightButton = []
    longerline2mouse.time = []
    longerline2mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    longerline2mouse.mouseClock.reset()
    # store start times for longerline_2
    longerline_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    longerline_2.tStart = globalClock.getTime(format='float')
    longerline_2.status = STARTED
    thisExp.addData('longerline_2.started', longerline_2.tStart)
    longerline_2.maxDuration = None
    # keep track of which components have finished
    longerline_2Components = longerline_2.components
    for thisComponent in longerline_2.components:
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
    
    # --- Run Routine "longerline_2" ---
    thisExp.currentRoutine = longerline_2
    longerline_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *longerline2pic* updates
        
        # if longerline2pic is starting this frame...
        if longerline2pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline2pic.frameNStart = frameN  # exact frame index
            longerline2pic.tStart = t  # local t and not account for scr refresh
            longerline2pic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline2pic, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline2pic.status = STARTED
            longerline2pic.setAutoDraw(True)
        
        # if longerline2pic is active this frame...
        if longerline2pic.status == STARTED:
            # update params
            pass
        
        # *longerline2area* updates
        
        # if longerline2area is starting this frame...
        if longerline2area.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline2area.frameNStart = frameN  # exact frame index
            longerline2area.tStart = t  # local t and not account for scr refresh
            longerline2area.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline2area, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline2area.status = STARTED
            longerline2area.setAutoDraw(True)
        
        # if longerline2area is active this frame...
        if longerline2area.status == STARTED:
            # update params
            pass
        
        # *Longerlinetext2* updates
        
        # if Longerlinetext2 is starting this frame...
        if Longerlinetext2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Longerlinetext2.frameNStart = frameN  # exact frame index
            Longerlinetext2.tStart = t  # local t and not account for scr refresh
            Longerlinetext2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Longerlinetext2, 'tStartRefresh')  # time at next scr refresh
            # update status
            Longerlinetext2.status = STARTED
            Longerlinetext2.setAutoDraw(True)
        
        # if Longerlinetext2 is active this frame...
        if Longerlinetext2.status == STARTED:
            # update params
            pass
        # *longerline2mouse* updates
        
        # if longerline2mouse is starting this frame...
        if longerline2mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline2mouse.frameNStart = frameN  # exact frame index
            longerline2mouse.tStart = t  # local t and not account for scr refresh
            longerline2mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline2mouse, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline2mouse.status = STARTED
            prevButtonState = longerline2mouse.getPressed()  # if button is down already this ISN'T a new click
        if longerline2mouse.status == STARTED:  # only update if started and not finished!
            buttons = longerline2mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(longerline2area, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(longerline2mouse):
                            gotValidClick = True
                            longerline2mouse.clicked_name.append(obj.name)
                            longerline2mouse.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = longerline2mouse.getPos()
                        longerline2mouse.x.append(float(x))
                        longerline2mouse.y.append(float(y))
                        buttons = longerline2mouse.getPressed()
                        longerline2mouse.leftButton.append(buttons[0])
                        longerline2mouse.midButton.append(buttons[1])
                        longerline2mouse.rightButton.append(buttons[2])
                        longerline2mouse.time.append(longerline2mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=longerline_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            longerline_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if longerline_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in longerline_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "longerline_2" ---
    for thisComponent in longerline_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for longerline_2
    longerline_2.tStop = globalClock.getTime(format='float')
    longerline_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('longerline_2.stopped', longerline_2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('longerline2mouse.x', longerline2mouse.x)
    thisExp.addData('longerline2mouse.y', longerline2mouse.y)
    thisExp.addData('longerline2mouse.leftButton', longerline2mouse.leftButton)
    thisExp.addData('longerline2mouse.midButton', longerline2mouse.midButton)
    thisExp.addData('longerline2mouse.rightButton', longerline2mouse.rightButton)
    thisExp.addData('longerline2mouse.time', longerline2mouse.time)
    thisExp.addData('longerline2mouse.clicked_name', longerline2mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "longerline_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "longerline_3" ---
    # create an object to store info about Routine longerline_3
    longerline_3 = data.Routine(
        name='longerline_3',
        components=[longerline3pic, longerline3area, Longerlinetext3, longerline3mouse],
    )
    longerline_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    Longerlinetext3.reset()
    # setup some python lists for storing info about the longerline3mouse
    longerline3mouse.x = []
    longerline3mouse.y = []
    longerline3mouse.leftButton = []
    longerline3mouse.midButton = []
    longerline3mouse.rightButton = []
    longerline3mouse.time = []
    longerline3mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    longerline3mouse.mouseClock.reset()
    # store start times for longerline_3
    longerline_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    longerline_3.tStart = globalClock.getTime(format='float')
    longerline_3.status = STARTED
    thisExp.addData('longerline_3.started', longerline_3.tStart)
    longerline_3.maxDuration = None
    # keep track of which components have finished
    longerline_3Components = longerline_3.components
    for thisComponent in longerline_3.components:
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
    
    # --- Run Routine "longerline_3" ---
    thisExp.currentRoutine = longerline_3
    longerline_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *longerline3pic* updates
        
        # if longerline3pic is starting this frame...
        if longerline3pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline3pic.frameNStart = frameN  # exact frame index
            longerline3pic.tStart = t  # local t and not account for scr refresh
            longerline3pic.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline3pic, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline3pic.status = STARTED
            longerline3pic.setAutoDraw(True)
        
        # if longerline3pic is active this frame...
        if longerline3pic.status == STARTED:
            # update params
            pass
        
        # *longerline3area* updates
        
        # if longerline3area is starting this frame...
        if longerline3area.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline3area.frameNStart = frameN  # exact frame index
            longerline3area.tStart = t  # local t and not account for scr refresh
            longerline3area.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline3area, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline3area.status = STARTED
            longerline3area.setAutoDraw(True)
        
        # if longerline3area is active this frame...
        if longerline3area.status == STARTED:
            # update params
            pass
        
        # *Longerlinetext3* updates
        
        # if Longerlinetext3 is starting this frame...
        if Longerlinetext3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Longerlinetext3.frameNStart = frameN  # exact frame index
            Longerlinetext3.tStart = t  # local t and not account for scr refresh
            Longerlinetext3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Longerlinetext3, 'tStartRefresh')  # time at next scr refresh
            # update status
            Longerlinetext3.status = STARTED
            Longerlinetext3.setAutoDraw(True)
        
        # if Longerlinetext3 is active this frame...
        if Longerlinetext3.status == STARTED:
            # update params
            pass
        # *longerline3mouse* updates
        
        # if longerline3mouse is starting this frame...
        if longerline3mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            longerline3mouse.frameNStart = frameN  # exact frame index
            longerline3mouse.tStart = t  # local t and not account for scr refresh
            longerline3mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(longerline3mouse, 'tStartRefresh')  # time at next scr refresh
            # update status
            longerline3mouse.status = STARTED
            prevButtonState = longerline3mouse.getPressed()  # if button is down already this ISN'T a new click
        if longerline3mouse.status == STARTED:  # only update if started and not finished!
            buttons = longerline3mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(longerline3area, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(longerline3mouse):
                            gotValidClick = True
                            longerline3mouse.clicked_name.append(obj.name)
                            longerline3mouse.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = longerline3mouse.getPos()
                        longerline3mouse.x.append(float(x))
                        longerline3mouse.y.append(float(y))
                        buttons = longerline3mouse.getPressed()
                        longerline3mouse.leftButton.append(buttons[0])
                        longerline3mouse.midButton.append(buttons[1])
                        longerline3mouse.rightButton.append(buttons[2])
                        longerline3mouse.time.append(longerline3mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=longerline_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            longerline_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if longerline_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in longerline_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "longerline_3" ---
    for thisComponent in longerline_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for longerline_3
    longerline_3.tStop = globalClock.getTime(format='float')
    longerline_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('longerline_3.stopped', longerline_3.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('longerline3mouse.x', longerline3mouse.x)
    thisExp.addData('longerline3mouse.y', longerline3mouse.y)
    thisExp.addData('longerline3mouse.leftButton', longerline3mouse.leftButton)
    thisExp.addData('longerline3mouse.midButton', longerline3mouse.midButton)
    thisExp.addData('longerline3mouse.rightButton', longerline3mouse.rightButton)
    thisExp.addData('longerline3mouse.time', longerline3mouse.time)
    thisExp.addData('longerline3mouse.clicked_name', longerline3mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "longerline_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle1" ---
    # create an object to store info about Routine clickcircle1
    clickcircle1 = data.Routine(
        name='clickcircle1',
        components=[circle_click_text1, circle_click, circle_mouse1],
    )
    clickcircle1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1.reset()
    # setup some python lists for storing info about the circle_mouse1
    circle_mouse1.x = []
    circle_mouse1.y = []
    circle_mouse1.leftButton = []
    circle_mouse1.midButton = []
    circle_mouse1.rightButton = []
    circle_mouse1.time = []
    circle_mouse1.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse1.mouseClock.reset()
    # store start times for clickcircle1
    clickcircle1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle1.tStart = globalClock.getTime(format='float')
    clickcircle1.status = STARTED
    thisExp.addData('clickcircle1.started', clickcircle1.tStart)
    clickcircle1.maxDuration = None
    # keep track of which components have finished
    clickcircle1Components = clickcircle1.components
    for thisComponent in clickcircle1.components:
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
    
    # --- Run Routine "clickcircle1" ---
    thisExp.currentRoutine = clickcircle1
    clickcircle1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_text1* updates
        
        # if circle_click_text1 is starting this frame...
        if circle_click_text1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1.frameNStart = frameN  # exact frame index
            circle_click_text1.tStart = t  # local t and not account for scr refresh
            circle_click_text1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1.status = STARTED
            circle_click_text1.setAutoDraw(True)
        
        # if circle_click_text1 is active this frame...
        if circle_click_text1.status == STARTED:
            # update params
            pass
        
        # *circle_click* updates
        
        # if circle_click is starting this frame...
        if circle_click.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click.frameNStart = frameN  # exact frame index
            circle_click.tStart = t  # local t and not account for scr refresh
            circle_click.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click.status = STARTED
            circle_click.setAutoDraw(True)
        
        # if circle_click is active this frame...
        if circle_click.status == STARTED:
            # update params
            pass
        # *circle_mouse1* updates
        
        # if circle_mouse1 is starting this frame...
        if circle_mouse1.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse1.frameNStart = frameN  # exact frame index
            circle_mouse1.tStart = t  # local t and not account for scr refresh
            circle_mouse1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse1, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse1.status = STARTED
            prevButtonState = circle_mouse1.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse1.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse1.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse1):
                            gotValidClick = True
                            circle_mouse1.clicked_name.append(obj.name)
                            circle_mouse1.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse1.getPos()
                        circle_mouse1.x.append(float(x))
                        circle_mouse1.y.append(float(y))
                        buttons = circle_mouse1.getPressed()
                        circle_mouse1.leftButton.append(buttons[0])
                        circle_mouse1.midButton.append(buttons[1])
                        circle_mouse1.rightButton.append(buttons[2])
                        circle_mouse1.time.append(circle_mouse1.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle1" ---
    for thisComponent in clickcircle1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle1
    clickcircle1.tStop = globalClock.getTime(format='float')
    clickcircle1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle1.stopped', clickcircle1.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse1.x', circle_mouse1.x)
    thisExp.addData('circle_mouse1.y', circle_mouse1.y)
    thisExp.addData('circle_mouse1.leftButton', circle_mouse1.leftButton)
    thisExp.addData('circle_mouse1.midButton', circle_mouse1.midButton)
    thisExp.addData('circle_mouse1.rightButton', circle_mouse1.rightButton)
    thisExp.addData('circle_mouse1.time', circle_mouse1.time)
    thisExp.addData('circle_mouse1.clicked_name', circle_mouse1.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle2" ---
    # create an object to store info about Routine clickcircle2
    clickcircle2 = data.Routine(
        name='clickcircle2',
        components=[circle_click_2, circle_click_text1_2, circle_mouse2],
    )
    clickcircle2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_2.reset()
    # setup some python lists for storing info about the circle_mouse2
    circle_mouse2.x = []
    circle_mouse2.y = []
    circle_mouse2.leftButton = []
    circle_mouse2.midButton = []
    circle_mouse2.rightButton = []
    circle_mouse2.time = []
    circle_mouse2.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse2.mouseClock.reset()
    # store start times for clickcircle2
    clickcircle2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle2.tStart = globalClock.getTime(format='float')
    clickcircle2.status = STARTED
    thisExp.addData('clickcircle2.started', clickcircle2.tStart)
    clickcircle2.maxDuration = None
    # keep track of which components have finished
    clickcircle2Components = clickcircle2.components
    for thisComponent in clickcircle2.components:
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
    
    # --- Run Routine "clickcircle2" ---
    thisExp.currentRoutine = clickcircle2
    clickcircle2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_2* updates
        
        # if circle_click_2 is starting this frame...
        if circle_click_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_2.frameNStart = frameN  # exact frame index
            circle_click_2.tStart = t  # local t and not account for scr refresh
            circle_click_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_2.status = STARTED
            circle_click_2.setAutoDraw(True)
        
        # if circle_click_2 is active this frame...
        if circle_click_2.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_2* updates
        
        # if circle_click_text1_2 is starting this frame...
        if circle_click_text1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_2.frameNStart = frameN  # exact frame index
            circle_click_text1_2.tStart = t  # local t and not account for scr refresh
            circle_click_text1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_2.status = STARTED
            circle_click_text1_2.setAutoDraw(True)
        
        # if circle_click_text1_2 is active this frame...
        if circle_click_text1_2.status == STARTED:
            # update params
            pass
        # *circle_mouse2* updates
        
        # if circle_mouse2 is starting this frame...
        if circle_mouse2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse2.frameNStart = frameN  # exact frame index
            circle_mouse2.tStart = t  # local t and not account for scr refresh
            circle_mouse2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse2, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse2.status = STARTED
            prevButtonState = circle_mouse2.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse2.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse2):
                            gotValidClick = True
                            circle_mouse2.clicked_name.append(obj.name)
                            circle_mouse2.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse2.getPos()
                        circle_mouse2.x.append(float(x))
                        circle_mouse2.y.append(float(y))
                        buttons = circle_mouse2.getPressed()
                        circle_mouse2.leftButton.append(buttons[0])
                        circle_mouse2.midButton.append(buttons[1])
                        circle_mouse2.rightButton.append(buttons[2])
                        circle_mouse2.time.append(circle_mouse2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle2" ---
    for thisComponent in clickcircle2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle2
    clickcircle2.tStop = globalClock.getTime(format='float')
    clickcircle2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle2.stopped', clickcircle2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse2.x', circle_mouse2.x)
    thisExp.addData('circle_mouse2.y', circle_mouse2.y)
    thisExp.addData('circle_mouse2.leftButton', circle_mouse2.leftButton)
    thisExp.addData('circle_mouse2.midButton', circle_mouse2.midButton)
    thisExp.addData('circle_mouse2.rightButton', circle_mouse2.rightButton)
    thisExp.addData('circle_mouse2.time', circle_mouse2.time)
    thisExp.addData('circle_mouse2.clicked_name', circle_mouse2.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle3" ---
    # create an object to store info about Routine clickcircle3
    clickcircle3 = data.Routine(
        name='clickcircle3',
        components=[circle_click_3, circle_click_text1_3, circle_mouse3],
    )
    clickcircle3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_3.reset()
    # setup some python lists for storing info about the circle_mouse3
    circle_mouse3.x = []
    circle_mouse3.y = []
    circle_mouse3.leftButton = []
    circle_mouse3.midButton = []
    circle_mouse3.rightButton = []
    circle_mouse3.time = []
    circle_mouse3.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse3.mouseClock.reset()
    # store start times for clickcircle3
    clickcircle3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle3.tStart = globalClock.getTime(format='float')
    clickcircle3.status = STARTED
    thisExp.addData('clickcircle3.started', clickcircle3.tStart)
    clickcircle3.maxDuration = None
    # keep track of which components have finished
    clickcircle3Components = clickcircle3.components
    for thisComponent in clickcircle3.components:
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
    
    # --- Run Routine "clickcircle3" ---
    thisExp.currentRoutine = clickcircle3
    clickcircle3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_3* updates
        
        # if circle_click_3 is starting this frame...
        if circle_click_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_3.frameNStart = frameN  # exact frame index
            circle_click_3.tStart = t  # local t and not account for scr refresh
            circle_click_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_3.status = STARTED
            circle_click_3.setAutoDraw(True)
        
        # if circle_click_3 is active this frame...
        if circle_click_3.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_3* updates
        
        # if circle_click_text1_3 is starting this frame...
        if circle_click_text1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_3.frameNStart = frameN  # exact frame index
            circle_click_text1_3.tStart = t  # local t and not account for scr refresh
            circle_click_text1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_3.status = STARTED
            circle_click_text1_3.setAutoDraw(True)
        
        # if circle_click_text1_3 is active this frame...
        if circle_click_text1_3.status == STARTED:
            # update params
            pass
        # *circle_mouse3* updates
        
        # if circle_mouse3 is starting this frame...
        if circle_mouse3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse3.frameNStart = frameN  # exact frame index
            circle_mouse3.tStart = t  # local t and not account for scr refresh
            circle_mouse3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse3, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse3.status = STARTED
            prevButtonState = circle_mouse3.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse3.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse3.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_3, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse3):
                            gotValidClick = True
                            circle_mouse3.clicked_name.append(obj.name)
                            circle_mouse3.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse3.getPos()
                        circle_mouse3.x.append(float(x))
                        circle_mouse3.y.append(float(y))
                        buttons = circle_mouse3.getPressed()
                        circle_mouse3.leftButton.append(buttons[0])
                        circle_mouse3.midButton.append(buttons[1])
                        circle_mouse3.rightButton.append(buttons[2])
                        circle_mouse3.time.append(circle_mouse3.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle3" ---
    for thisComponent in clickcircle3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle3
    clickcircle3.tStop = globalClock.getTime(format='float')
    clickcircle3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle3.stopped', clickcircle3.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse3.x', circle_mouse3.x)
    thisExp.addData('circle_mouse3.y', circle_mouse3.y)
    thisExp.addData('circle_mouse3.leftButton', circle_mouse3.leftButton)
    thisExp.addData('circle_mouse3.midButton', circle_mouse3.midButton)
    thisExp.addData('circle_mouse3.rightButton', circle_mouse3.rightButton)
    thisExp.addData('circle_mouse3.time', circle_mouse3.time)
    thisExp.addData('circle_mouse3.clicked_name', circle_mouse3.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle4" ---
    # create an object to store info about Routine clickcircle4
    clickcircle4 = data.Routine(
        name='clickcircle4',
        components=[circle_click_4, circle_click_text1_4, circle_mouse4],
    )
    clickcircle4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_4.reset()
    # setup some python lists for storing info about the circle_mouse4
    circle_mouse4.x = []
    circle_mouse4.y = []
    circle_mouse4.leftButton = []
    circle_mouse4.midButton = []
    circle_mouse4.rightButton = []
    circle_mouse4.time = []
    circle_mouse4.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse4.mouseClock.reset()
    # store start times for clickcircle4
    clickcircle4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle4.tStart = globalClock.getTime(format='float')
    clickcircle4.status = STARTED
    thisExp.addData('clickcircle4.started', clickcircle4.tStart)
    clickcircle4.maxDuration = None
    # keep track of which components have finished
    clickcircle4Components = clickcircle4.components
    for thisComponent in clickcircle4.components:
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
    
    # --- Run Routine "clickcircle4" ---
    thisExp.currentRoutine = clickcircle4
    clickcircle4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_4* updates
        
        # if circle_click_4 is starting this frame...
        if circle_click_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_4.frameNStart = frameN  # exact frame index
            circle_click_4.tStart = t  # local t and not account for scr refresh
            circle_click_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_4.status = STARTED
            circle_click_4.setAutoDraw(True)
        
        # if circle_click_4 is active this frame...
        if circle_click_4.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_4* updates
        
        # if circle_click_text1_4 is starting this frame...
        if circle_click_text1_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_4.frameNStart = frameN  # exact frame index
            circle_click_text1_4.tStart = t  # local t and not account for scr refresh
            circle_click_text1_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_4.status = STARTED
            circle_click_text1_4.setAutoDraw(True)
        
        # if circle_click_text1_4 is active this frame...
        if circle_click_text1_4.status == STARTED:
            # update params
            pass
        # *circle_mouse4* updates
        
        # if circle_mouse4 is starting this frame...
        if circle_mouse4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse4.frameNStart = frameN  # exact frame index
            circle_mouse4.tStart = t  # local t and not account for scr refresh
            circle_mouse4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse4, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse4.status = STARTED
            prevButtonState = circle_mouse4.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse4.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_4, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse4):
                            gotValidClick = True
                            circle_mouse4.clicked_name.append(obj.name)
                            circle_mouse4.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse4.getPos()
                        circle_mouse4.x.append(float(x))
                        circle_mouse4.y.append(float(y))
                        buttons = circle_mouse4.getPressed()
                        circle_mouse4.leftButton.append(buttons[0])
                        circle_mouse4.midButton.append(buttons[1])
                        circle_mouse4.rightButton.append(buttons[2])
                        circle_mouse4.time.append(circle_mouse4.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle4,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle4.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle4.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle4" ---
    for thisComponent in clickcircle4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle4
    clickcircle4.tStop = globalClock.getTime(format='float')
    clickcircle4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle4.stopped', clickcircle4.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse4.x', circle_mouse4.x)
    thisExp.addData('circle_mouse4.y', circle_mouse4.y)
    thisExp.addData('circle_mouse4.leftButton', circle_mouse4.leftButton)
    thisExp.addData('circle_mouse4.midButton', circle_mouse4.midButton)
    thisExp.addData('circle_mouse4.rightButton', circle_mouse4.rightButton)
    thisExp.addData('circle_mouse4.time', circle_mouse4.time)
    thisExp.addData('circle_mouse4.clicked_name', circle_mouse4.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle5" ---
    # create an object to store info about Routine clickcircle5
    clickcircle5 = data.Routine(
        name='clickcircle5',
        components=[circle_click_5, circle_click_text1_5, circle_mouse5],
    )
    clickcircle5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_5.reset()
    # setup some python lists for storing info about the circle_mouse5
    circle_mouse5.x = []
    circle_mouse5.y = []
    circle_mouse5.leftButton = []
    circle_mouse5.midButton = []
    circle_mouse5.rightButton = []
    circle_mouse5.time = []
    circle_mouse5.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse5.mouseClock.reset()
    # store start times for clickcircle5
    clickcircle5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle5.tStart = globalClock.getTime(format='float')
    clickcircle5.status = STARTED
    thisExp.addData('clickcircle5.started', clickcircle5.tStart)
    clickcircle5.maxDuration = None
    # keep track of which components have finished
    clickcircle5Components = clickcircle5.components
    for thisComponent in clickcircle5.components:
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
    
    # --- Run Routine "clickcircle5" ---
    thisExp.currentRoutine = clickcircle5
    clickcircle5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_5* updates
        
        # if circle_click_5 is starting this frame...
        if circle_click_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_5.frameNStart = frameN  # exact frame index
            circle_click_5.tStart = t  # local t and not account for scr refresh
            circle_click_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_5.status = STARTED
            circle_click_5.setAutoDraw(True)
        
        # if circle_click_5 is active this frame...
        if circle_click_5.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_5* updates
        
        # if circle_click_text1_5 is starting this frame...
        if circle_click_text1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_5.frameNStart = frameN  # exact frame index
            circle_click_text1_5.tStart = t  # local t and not account for scr refresh
            circle_click_text1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_5.status = STARTED
            circle_click_text1_5.setAutoDraw(True)
        
        # if circle_click_text1_5 is active this frame...
        if circle_click_text1_5.status == STARTED:
            # update params
            pass
        # *circle_mouse5* updates
        
        # if circle_mouse5 is starting this frame...
        if circle_mouse5.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse5.frameNStart = frameN  # exact frame index
            circle_mouse5.tStart = t  # local t and not account for scr refresh
            circle_mouse5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse5, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse5.status = STARTED
            prevButtonState = circle_mouse5.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse5.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse5.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_5, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse5):
                            gotValidClick = True
                            circle_mouse5.clicked_name.append(obj.name)
                            circle_mouse5.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse5.getPos()
                        circle_mouse5.x.append(float(x))
                        circle_mouse5.y.append(float(y))
                        buttons = circle_mouse5.getPressed()
                        circle_mouse5.leftButton.append(buttons[0])
                        circle_mouse5.midButton.append(buttons[1])
                        circle_mouse5.rightButton.append(buttons[2])
                        circle_mouse5.time.append(circle_mouse5.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle5,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle5.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle5.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle5" ---
    for thisComponent in clickcircle5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle5
    clickcircle5.tStop = globalClock.getTime(format='float')
    clickcircle5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle5.stopped', clickcircle5.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse5.x', circle_mouse5.x)
    thisExp.addData('circle_mouse5.y', circle_mouse5.y)
    thisExp.addData('circle_mouse5.leftButton', circle_mouse5.leftButton)
    thisExp.addData('circle_mouse5.midButton', circle_mouse5.midButton)
    thisExp.addData('circle_mouse5.rightButton', circle_mouse5.rightButton)
    thisExp.addData('circle_mouse5.time', circle_mouse5.time)
    thisExp.addData('circle_mouse5.clicked_name', circle_mouse5.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle6" ---
    # create an object to store info about Routine clickcircle6
    clickcircle6 = data.Routine(
        name='clickcircle6',
        components=[circle_click_6, circle_click_text1_6, circle_mouse6],
    )
    clickcircle6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_6.reset()
    # setup some python lists for storing info about the circle_mouse6
    circle_mouse6.x = []
    circle_mouse6.y = []
    circle_mouse6.leftButton = []
    circle_mouse6.midButton = []
    circle_mouse6.rightButton = []
    circle_mouse6.time = []
    circle_mouse6.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse6.mouseClock.reset()
    # store start times for clickcircle6
    clickcircle6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle6.tStart = globalClock.getTime(format='float')
    clickcircle6.status = STARTED
    thisExp.addData('clickcircle6.started', clickcircle6.tStart)
    clickcircle6.maxDuration = None
    # keep track of which components have finished
    clickcircle6Components = clickcircle6.components
    for thisComponent in clickcircle6.components:
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
    
    # --- Run Routine "clickcircle6" ---
    thisExp.currentRoutine = clickcircle6
    clickcircle6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_6* updates
        
        # if circle_click_6 is starting this frame...
        if circle_click_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_6.frameNStart = frameN  # exact frame index
            circle_click_6.tStart = t  # local t and not account for scr refresh
            circle_click_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_6.status = STARTED
            circle_click_6.setAutoDraw(True)
        
        # if circle_click_6 is active this frame...
        if circle_click_6.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_6* updates
        
        # if circle_click_text1_6 is starting this frame...
        if circle_click_text1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_6.frameNStart = frameN  # exact frame index
            circle_click_text1_6.tStart = t  # local t and not account for scr refresh
            circle_click_text1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_6.status = STARTED
            circle_click_text1_6.setAutoDraw(True)
        
        # if circle_click_text1_6 is active this frame...
        if circle_click_text1_6.status == STARTED:
            # update params
            pass
        # *circle_mouse6* updates
        
        # if circle_mouse6 is starting this frame...
        if circle_mouse6.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse6.frameNStart = frameN  # exact frame index
            circle_mouse6.tStart = t  # local t and not account for scr refresh
            circle_mouse6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse6, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse6.status = STARTED
            prevButtonState = circle_mouse6.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse6.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse6.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_6, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse6):
                            gotValidClick = True
                            circle_mouse6.clicked_name.append(obj.name)
                            circle_mouse6.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse6.getPos()
                        circle_mouse6.x.append(float(x))
                        circle_mouse6.y.append(float(y))
                        buttons = circle_mouse6.getPressed()
                        circle_mouse6.leftButton.append(buttons[0])
                        circle_mouse6.midButton.append(buttons[1])
                        circle_mouse6.rightButton.append(buttons[2])
                        circle_mouse6.time.append(circle_mouse6.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle6,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle6.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle6.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle6" ---
    for thisComponent in clickcircle6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle6
    clickcircle6.tStop = globalClock.getTime(format='float')
    clickcircle6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle6.stopped', clickcircle6.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse6.x', circle_mouse6.x)
    thisExp.addData('circle_mouse6.y', circle_mouse6.y)
    thisExp.addData('circle_mouse6.leftButton', circle_mouse6.leftButton)
    thisExp.addData('circle_mouse6.midButton', circle_mouse6.midButton)
    thisExp.addData('circle_mouse6.rightButton', circle_mouse6.rightButton)
    thisExp.addData('circle_mouse6.time', circle_mouse6.time)
    thisExp.addData('circle_mouse6.clicked_name', circle_mouse6.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle7" ---
    # create an object to store info about Routine clickcircle7
    clickcircle7 = data.Routine(
        name='clickcircle7',
        components=[circle_click_7, circle_click_text1_7, circle_mouse7],
    )
    clickcircle7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_7.reset()
    # setup some python lists for storing info about the circle_mouse7
    circle_mouse7.x = []
    circle_mouse7.y = []
    circle_mouse7.leftButton = []
    circle_mouse7.midButton = []
    circle_mouse7.rightButton = []
    circle_mouse7.time = []
    circle_mouse7.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse7.mouseClock.reset()
    # store start times for clickcircle7
    clickcircle7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle7.tStart = globalClock.getTime(format='float')
    clickcircle7.status = STARTED
    thisExp.addData('clickcircle7.started', clickcircle7.tStart)
    clickcircle7.maxDuration = None
    # keep track of which components have finished
    clickcircle7Components = clickcircle7.components
    for thisComponent in clickcircle7.components:
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
    
    # --- Run Routine "clickcircle7" ---
    thisExp.currentRoutine = clickcircle7
    clickcircle7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_7* updates
        
        # if circle_click_7 is starting this frame...
        if circle_click_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_7.frameNStart = frameN  # exact frame index
            circle_click_7.tStart = t  # local t and not account for scr refresh
            circle_click_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_7.status = STARTED
            circle_click_7.setAutoDraw(True)
        
        # if circle_click_7 is active this frame...
        if circle_click_7.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_7* updates
        
        # if circle_click_text1_7 is starting this frame...
        if circle_click_text1_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_7.frameNStart = frameN  # exact frame index
            circle_click_text1_7.tStart = t  # local t and not account for scr refresh
            circle_click_text1_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_7.status = STARTED
            circle_click_text1_7.setAutoDraw(True)
        
        # if circle_click_text1_7 is active this frame...
        if circle_click_text1_7.status == STARTED:
            # update params
            pass
        # *circle_mouse7* updates
        
        # if circle_mouse7 is starting this frame...
        if circle_mouse7.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse7.frameNStart = frameN  # exact frame index
            circle_mouse7.tStart = t  # local t and not account for scr refresh
            circle_mouse7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse7, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse7.status = STARTED
            prevButtonState = circle_mouse7.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse7.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse7.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_7, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse7):
                            gotValidClick = True
                            circle_mouse7.clicked_name.append(obj.name)
                            circle_mouse7.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse7.getPos()
                        circle_mouse7.x.append(float(x))
                        circle_mouse7.y.append(float(y))
                        buttons = circle_mouse7.getPressed()
                        circle_mouse7.leftButton.append(buttons[0])
                        circle_mouse7.midButton.append(buttons[1])
                        circle_mouse7.rightButton.append(buttons[2])
                        circle_mouse7.time.append(circle_mouse7.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle7,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle7.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle7.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle7" ---
    for thisComponent in clickcircle7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle7
    clickcircle7.tStop = globalClock.getTime(format='float')
    clickcircle7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle7.stopped', clickcircle7.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse7.x', circle_mouse7.x)
    thisExp.addData('circle_mouse7.y', circle_mouse7.y)
    thisExp.addData('circle_mouse7.leftButton', circle_mouse7.leftButton)
    thisExp.addData('circle_mouse7.midButton', circle_mouse7.midButton)
    thisExp.addData('circle_mouse7.rightButton', circle_mouse7.rightButton)
    thisExp.addData('circle_mouse7.time', circle_mouse7.time)
    thisExp.addData('circle_mouse7.clicked_name', circle_mouse7.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle8" ---
    # create an object to store info about Routine clickcircle8
    clickcircle8 = data.Routine(
        name='clickcircle8',
        components=[circle_click_8, circle_click_text1_8, circle_mouse8],
    )
    clickcircle8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_8.reset()
    # setup some python lists for storing info about the circle_mouse8
    circle_mouse8.x = []
    circle_mouse8.y = []
    circle_mouse8.leftButton = []
    circle_mouse8.midButton = []
    circle_mouse8.rightButton = []
    circle_mouse8.time = []
    circle_mouse8.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse8.mouseClock.reset()
    # store start times for clickcircle8
    clickcircle8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle8.tStart = globalClock.getTime(format='float')
    clickcircle8.status = STARTED
    thisExp.addData('clickcircle8.started', clickcircle8.tStart)
    clickcircle8.maxDuration = None
    # keep track of which components have finished
    clickcircle8Components = clickcircle8.components
    for thisComponent in clickcircle8.components:
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
    
    # --- Run Routine "clickcircle8" ---
    thisExp.currentRoutine = clickcircle8
    clickcircle8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_8* updates
        
        # if circle_click_8 is starting this frame...
        if circle_click_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_8.frameNStart = frameN  # exact frame index
            circle_click_8.tStart = t  # local t and not account for scr refresh
            circle_click_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_8, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_8.status = STARTED
            circle_click_8.setAutoDraw(True)
        
        # if circle_click_8 is active this frame...
        if circle_click_8.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_8* updates
        
        # if circle_click_text1_8 is starting this frame...
        if circle_click_text1_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_8.frameNStart = frameN  # exact frame index
            circle_click_text1_8.tStart = t  # local t and not account for scr refresh
            circle_click_text1_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_8, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_8.status = STARTED
            circle_click_text1_8.setAutoDraw(True)
        
        # if circle_click_text1_8 is active this frame...
        if circle_click_text1_8.status == STARTED:
            # update params
            pass
        # *circle_mouse8* updates
        
        # if circle_mouse8 is starting this frame...
        if circle_mouse8.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse8.frameNStart = frameN  # exact frame index
            circle_mouse8.tStart = t  # local t and not account for scr refresh
            circle_mouse8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse8, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse8.status = STARTED
            prevButtonState = circle_mouse8.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse8.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse8.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_8, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse8):
                            gotValidClick = True
                            circle_mouse8.clicked_name.append(obj.name)
                            circle_mouse8.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse8.getPos()
                        circle_mouse8.x.append(float(x))
                        circle_mouse8.y.append(float(y))
                        buttons = circle_mouse8.getPressed()
                        circle_mouse8.leftButton.append(buttons[0])
                        circle_mouse8.midButton.append(buttons[1])
                        circle_mouse8.rightButton.append(buttons[2])
                        circle_mouse8.time.append(circle_mouse8.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle8,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle8.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle8.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle8" ---
    for thisComponent in clickcircle8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle8
    clickcircle8.tStop = globalClock.getTime(format='float')
    clickcircle8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle8.stopped', clickcircle8.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse8.x', circle_mouse8.x)
    thisExp.addData('circle_mouse8.y', circle_mouse8.y)
    thisExp.addData('circle_mouse8.leftButton', circle_mouse8.leftButton)
    thisExp.addData('circle_mouse8.midButton', circle_mouse8.midButton)
    thisExp.addData('circle_mouse8.rightButton', circle_mouse8.rightButton)
    thisExp.addData('circle_mouse8.time', circle_mouse8.time)
    thisExp.addData('circle_mouse8.clicked_name', circle_mouse8.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle9" ---
    # create an object to store info about Routine clickcircle9
    clickcircle9 = data.Routine(
        name='clickcircle9',
        components=[circle_click_9, circle_click_text1_9, circle_mouse9],
    )
    clickcircle9.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_9.reset()
    # setup some python lists for storing info about the circle_mouse9
    circle_mouse9.x = []
    circle_mouse9.y = []
    circle_mouse9.leftButton = []
    circle_mouse9.midButton = []
    circle_mouse9.rightButton = []
    circle_mouse9.time = []
    circle_mouse9.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse9.mouseClock.reset()
    # store start times for clickcircle9
    clickcircle9.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle9.tStart = globalClock.getTime(format='float')
    clickcircle9.status = STARTED
    thisExp.addData('clickcircle9.started', clickcircle9.tStart)
    clickcircle9.maxDuration = None
    # keep track of which components have finished
    clickcircle9Components = clickcircle9.components
    for thisComponent in clickcircle9.components:
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
    
    # --- Run Routine "clickcircle9" ---
    thisExp.currentRoutine = clickcircle9
    clickcircle9.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_9* updates
        
        # if circle_click_9 is starting this frame...
        if circle_click_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_9.frameNStart = frameN  # exact frame index
            circle_click_9.tStart = t  # local t and not account for scr refresh
            circle_click_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_9, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_9.status = STARTED
            circle_click_9.setAutoDraw(True)
        
        # if circle_click_9 is active this frame...
        if circle_click_9.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_9* updates
        
        # if circle_click_text1_9 is starting this frame...
        if circle_click_text1_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_9.frameNStart = frameN  # exact frame index
            circle_click_text1_9.tStart = t  # local t and not account for scr refresh
            circle_click_text1_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_9, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_9.status = STARTED
            circle_click_text1_9.setAutoDraw(True)
        
        # if circle_click_text1_9 is active this frame...
        if circle_click_text1_9.status == STARTED:
            # update params
            pass
        # *circle_mouse9* updates
        
        # if circle_mouse9 is starting this frame...
        if circle_mouse9.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse9.frameNStart = frameN  # exact frame index
            circle_mouse9.tStart = t  # local t and not account for scr refresh
            circle_mouse9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse9, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse9.status = STARTED
            prevButtonState = circle_mouse9.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse9.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse9.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_9, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse9):
                            gotValidClick = True
                            circle_mouse9.clicked_name.append(obj.name)
                            circle_mouse9.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse9.getPos()
                        circle_mouse9.x.append(float(x))
                        circle_mouse9.y.append(float(y))
                        buttons = circle_mouse9.getPressed()
                        circle_mouse9.leftButton.append(buttons[0])
                        circle_mouse9.midButton.append(buttons[1])
                        circle_mouse9.rightButton.append(buttons[2])
                        circle_mouse9.time.append(circle_mouse9.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle9,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle9.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle9.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle9.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle9" ---
    for thisComponent in clickcircle9.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle9
    clickcircle9.tStop = globalClock.getTime(format='float')
    clickcircle9.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle9.stopped', clickcircle9.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse9.x', circle_mouse9.x)
    thisExp.addData('circle_mouse9.y', circle_mouse9.y)
    thisExp.addData('circle_mouse9.leftButton', circle_mouse9.leftButton)
    thisExp.addData('circle_mouse9.midButton', circle_mouse9.midButton)
    thisExp.addData('circle_mouse9.rightButton', circle_mouse9.rightButton)
    thisExp.addData('circle_mouse9.time', circle_mouse9.time)
    thisExp.addData('circle_mouse9.clicked_name', circle_mouse9.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle9" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clickcircle10" ---
    # create an object to store info about Routine clickcircle10
    clickcircle10 = data.Routine(
        name='clickcircle10',
        components=[circle_click_10, circle_click_text1_10, circle_mouse10],
    )
    clickcircle10.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    circle_click_text1_10.reset()
    # setup some python lists for storing info about the circle_mouse10
    circle_mouse10.x = []
    circle_mouse10.y = []
    circle_mouse10.leftButton = []
    circle_mouse10.midButton = []
    circle_mouse10.rightButton = []
    circle_mouse10.time = []
    circle_mouse10.clicked_name = []
    gotValidClick = False  # until a click is received
    circle_mouse10.mouseClock.reset()
    # store start times for clickcircle10
    clickcircle10.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clickcircle10.tStart = globalClock.getTime(format='float')
    clickcircle10.status = STARTED
    thisExp.addData('clickcircle10.started', clickcircle10.tStart)
    clickcircle10.maxDuration = None
    # keep track of which components have finished
    clickcircle10Components = clickcircle10.components
    for thisComponent in clickcircle10.components:
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
    
    # --- Run Routine "clickcircle10" ---
    thisExp.currentRoutine = clickcircle10
    clickcircle10.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_click_10* updates
        
        # if circle_click_10 is starting this frame...
        if circle_click_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_10.frameNStart = frameN  # exact frame index
            circle_click_10.tStart = t  # local t and not account for scr refresh
            circle_click_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_10, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_10.status = STARTED
            circle_click_10.setAutoDraw(True)
        
        # if circle_click_10 is active this frame...
        if circle_click_10.status == STARTED:
            # update params
            pass
        
        # *circle_click_text1_10* updates
        
        # if circle_click_text1_10 is starting this frame...
        if circle_click_text1_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_click_text1_10.frameNStart = frameN  # exact frame index
            circle_click_text1_10.tStart = t  # local t and not account for scr refresh
            circle_click_text1_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_click_text1_10, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_click_text1_10.status = STARTED
            circle_click_text1_10.setAutoDraw(True)
        
        # if circle_click_text1_10 is active this frame...
        if circle_click_text1_10.status == STARTED:
            # update params
            pass
        # *circle_mouse10* updates
        
        # if circle_mouse10 is starting this frame...
        if circle_mouse10.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_mouse10.frameNStart = frameN  # exact frame index
            circle_mouse10.tStart = t  # local t and not account for scr refresh
            circle_mouse10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_mouse10, 'tStartRefresh')  # time at next scr refresh
            # update status
            circle_mouse10.status = STARTED
            prevButtonState = circle_mouse10.getPressed()  # if button is down already this ISN'T a new click
        if circle_mouse10.status == STARTED:  # only update if started and not finished!
            buttons = circle_mouse10.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(circle_click_10, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(circle_mouse10):
                            gotValidClick = True
                            circle_mouse10.clicked_name.append(obj.name)
                            circle_mouse10.clicked_name.append(obj.name)
                    if gotValidClick:
                        x, y = circle_mouse10.getPos()
                        circle_mouse10.x.append(float(x))
                        circle_mouse10.y.append(float(y))
                        buttons = circle_mouse10.getPressed()
                        circle_mouse10.leftButton.append(buttons[0])
                        circle_mouse10.midButton.append(buttons[1])
                        circle_mouse10.rightButton.append(buttons[2])
                        circle_mouse10.time.append(circle_mouse10.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clickcircle10,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            clickcircle10.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if clickcircle10.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in clickcircle10.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clickcircle10" ---
    for thisComponent in clickcircle10.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clickcircle10
    clickcircle10.tStop = globalClock.getTime(format='float')
    clickcircle10.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clickcircle10.stopped', clickcircle10.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('circle_mouse10.x', circle_mouse10.x)
    thisExp.addData('circle_mouse10.y', circle_mouse10.y)
    thisExp.addData('circle_mouse10.leftButton', circle_mouse10.leftButton)
    thisExp.addData('circle_mouse10.midButton', circle_mouse10.midButton)
    thisExp.addData('circle_mouse10.rightButton', circle_mouse10.rightButton)
    thisExp.addData('circle_mouse10.time', circle_mouse10.time)
    thisExp.addData('circle_mouse10.clicked_name', circle_mouse10.clicked_name)
    thisExp.nextEntry()
    # the Routine "clickcircle10" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Stimuli/randomblock.xlsx'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        A = data.TrialHandler2(
            name='A',
            nReps=ML_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(A)  # add the loop to the experiment
        thisA = A.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisA.rgb)
        if thisA != None:
            for paramName in thisA:
                globals()[paramName] = thisA[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisA in A:
            A.status = STARTED
            if hasattr(thisA, 'status'):
                thisA.status = STARTED
            currentLoop = A
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisA.rgb)
            if thisA != None:
                for paramName in thisA:
                    globals()[paramName] = thisA[paramName]
            
            # --- Prepare to start Routine "ML_INTRO" ---
            # create an object to store info about Routine ML_INTRO
            ML_INTRO = data.Routine(
                name='ML_INTRO',
                components=[ml_intro_text, sound_2, START_ML],
            )
            ML_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_2.setSound('Stimuli/Instructions/MUL_INTRO.ogg', hamming=True)
            sound_2.setVolume(1.0, log=False)
            sound_2.seek(0)
            # reset START_ML to account for continued clicks & clear times on/off
            START_ML.reset()
            # store start times for ML_INTRO
            ML_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ML_INTRO.tStart = globalClock.getTime(format='float')
            ML_INTRO.status = STARTED
            thisExp.addData('ML_INTRO.started', ML_INTRO.tStart)
            ML_INTRO.maxDuration = None
            # keep track of which components have finished
            ML_INTROComponents = ML_INTRO.components
            for thisComponent in ML_INTRO.components:
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
            
            # --- Run Routine "ML_INTRO" ---
            thisExp.currentRoutine = ML_INTRO
            ML_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisA, 'status') and thisA.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *ml_intro_text* updates
                
                # if ml_intro_text is starting this frame...
                if ml_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    ml_intro_text.frameNStart = frameN  # exact frame index
                    ml_intro_text.tStart = t  # local t and not account for scr refresh
                    ml_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ml_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    ml_intro_text.status = STARTED
                    ml_intro_text.setAutoDraw(True)
                
                # if ml_intro_text is active this frame...
                if ml_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_2* updates
                
                # if sound_2 is starting this frame...
                if sound_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_2.frameNStart = frameN  # exact frame index
                    sound_2.tStart = t  # local t and not account for scr refresh
                    sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_2.status = STARTED
                    sound_2.play(when=win)  # sync with win flip
                
                # if sound_2 is stopping this frame...
                if sound_2.status == STARTED:
                    if bool(False) or sound_2.isFinished:
                        # keep track of stop time/frame for later
                        sound_2.tStop = t  # not accounting for scr refresh
                        sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_2.frameNStop = frameN  # exact frame index
                        # update status
                        sound_2.status = FINISHED
                        sound_2.stop()
                # *START_ML* updates
                
                # if START_ML is starting this frame...
                if START_ML.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_ML.frameNStart = frameN  # exact frame index
                    START_ML.tStart = t  # local t and not account for scr refresh
                    START_ML.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_ML, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_ML.status = STARTED
                    win.callOnFlip(START_ML.buttonClock.reset)
                    START_ML.setAutoDraw(True)
                
                # if START_ML is active this frame...
                if START_ML.status == STARTED:
                    # update params
                    pass
                    # check whether START_ML has been pressed
                    if START_ML.isClicked:
                        if not START_ML.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_ML.timesOn.append(globalClock.getTime())
                            START_ML.timesOff.append(globalClock.getTime())
                        elif len(START_ML.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_ML.timesOff[-1] = globalClock.getTime()
                        if not START_ML.wasClicked:
                            # end routine when START_ML is clicked
                            continueRoutine = False
                        if not START_ML.wasClicked:
                            # run callback code when START_ML is clicked
                            pass
                # take note of whether START_ML was clicked, so that next frame we know if clicks are new
                START_ML.wasClicked = START_ML.isClicked and START_ML.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=ML_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    ML_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if ML_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in ML_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ML_INTRO" ---
            for thisComponent in ML_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ML_INTRO
            ML_INTRO.tStop = globalClock.getTime(format='float')
            ML_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('ML_INTRO.stopped', ML_INTRO.tStop)
            sound_2.pause()  # ensure sound has stopped at end of Routine
            A.addData('START_ML.numClicks', START_ML.numClicks)
            if START_ML.numClicks:
               A.addData('START_ML.timesOn', START_ML.timesOn)
               A.addData('START_ML.timesOff', START_ML.timesOff)
            else:
               A.addData('START_ML.timesOn', "")
               A.addData('START_ML.timesOff', "")
            # the Routine "ML_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            ML_rep = data.TrialHandler2(
                name='ML_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Müller-Lyer trials/ML_condition_file.xlsx', 
                selection=set_rows_ML
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(ML_rep)  # add the loop to the experiment
            thisML_rep = ML_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisML_rep.rgb)
            if thisML_rep != None:
                for paramName in thisML_rep:
                    globals()[paramName] = thisML_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisML_rep in ML_rep:
                ML_rep.status = STARTED
                if hasattr(thisML_rep, 'status'):
                    thisML_rep.status = STARTED
                currentLoop = ML_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisML_rep.rgb)
                if thisML_rep != None:
                    for paramName in thisML_rep:
                        globals()[paramName] = thisML_rep[paramName]
                
                # --- Prepare to start Routine "TEST_ML" ---
                # create an object to store info about Routine TEST_ML
                TEST_ML = data.Routine(
                    name='TEST_ML',
                    components=[image_ML, LEFT_ML, RIGHT_ML, mouse_ML, sound_3, key_resp],
                )
                TEST_ML.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_ML.setImage(path)
                # setup some python lists for storing info about the mouse_ML
                mouse_ML.x = []
                mouse_ML.y = []
                mouse_ML.leftButton = []
                mouse_ML.midButton = []
                mouse_ML.rightButton = []
                mouse_ML.time = []
                mouse_ML.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_3.setSound('Stimuli/Instructions/MUL_SHORT.ogg', secs=10, hamming=True)
                sound_3.setVolume(1.0, log=False)
                sound_3.seek(0)
                # create starting attributes for key_resp
                key_resp.keys = []
                key_resp.rt = []
                _key_resp_allKeys = []
                # store start times for TEST_ML
                TEST_ML.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_ML.tStart = globalClock.getTime(format='float')
                TEST_ML.status = STARTED
                thisExp.addData('TEST_ML.started', TEST_ML.tStart)
                TEST_ML.maxDuration = None
                # keep track of which components have finished
                TEST_MLComponents = TEST_ML.components
                for thisComponent in TEST_ML.components:
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
                
                # --- Run Routine "TEST_ML" ---
                thisExp.currentRoutine = TEST_ML
                TEST_ML.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisML_rep, 'status') and thisML_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_ML* updates
                    
                    # if image_ML is starting this frame...
                    if image_ML.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_ML.frameNStart = frameN  # exact frame index
                        image_ML.tStart = t  # local t and not account for scr refresh
                        image_ML.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_ML, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_ML.status = STARTED
                        image_ML.setAutoDraw(True)
                    
                    # if image_ML is active this frame...
                    if image_ML.status == STARTED:
                        # update params
                        pass
                    
                    # if image_ML is stopping this frame...
                    if image_ML.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_ML.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_ML.tStop = t  # not accounting for scr refresh
                            image_ML.tStopRefresh = tThisFlipGlobal  # on global time
                            image_ML.frameNStop = frameN  # exact frame index
                            # update status
                            image_ML.status = FINISHED
                            image_ML.setAutoDraw(False)
                    
                    # *LEFT_ML* updates
                    
                    # if LEFT_ML is starting this frame...
                    if LEFT_ML.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_ML.frameNStart = frameN  # exact frame index
                        LEFT_ML.tStart = t  # local t and not account for scr refresh
                        LEFT_ML.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_ML, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_ML.status = STARTED
                        LEFT_ML.setAutoDraw(True)
                    
                    # if LEFT_ML is active this frame...
                    if LEFT_ML.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_ML is stopping this frame...
                    if LEFT_ML.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_ML.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_ML.tStop = t  # not accounting for scr refresh
                            LEFT_ML.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_ML.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_ML.status = FINISHED
                            LEFT_ML.setAutoDraw(False)
                    
                    # *RIGHT_ML* updates
                    
                    # if RIGHT_ML is starting this frame...
                    if RIGHT_ML.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_ML.frameNStart = frameN  # exact frame index
                        RIGHT_ML.tStart = t  # local t and not account for scr refresh
                        RIGHT_ML.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_ML, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_ML.status = STARTED
                        RIGHT_ML.setAutoDraw(True)
                    
                    # if RIGHT_ML is active this frame...
                    if RIGHT_ML.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_ML is stopping this frame...
                    if RIGHT_ML.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_ML.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_ML.tStop = t  # not accounting for scr refresh
                            RIGHT_ML.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_ML.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_ML.status = FINISHED
                            RIGHT_ML.setAutoDraw(False)
                    # *mouse_ML* updates
                    
                    # if mouse_ML is starting this frame...
                    if mouse_ML.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_ML.frameNStart = frameN  # exact frame index
                        mouse_ML.tStart = t  # local t and not account for scr refresh
                        mouse_ML.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_ML, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouse_ML.status = STARTED
                        mouse_ML.mouseClock.reset()
                        prevButtonState = mouse_ML.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_ML is stopping this frame...
                    if mouse_ML.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_ML.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_ML.tStop = t  # not accounting for scr refresh
                            mouse_ML.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_ML.frameNStop = frameN  # exact frame index
                            # update status
                            mouse_ML.status = FINISHED
                    if mouse_ML.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_ML.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_ML, RIGHT_ML], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_ML):
                                        gotValidClick = True
                                        mouse_ML.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_ML.clicked_name.append(None)
                                x, y = mouse_ML.getPos()
                                mouse_ML.x.append(float(x))
                                mouse_ML.y.append(float(y))
                                buttons = mouse_ML.getPressed()
                                mouse_ML.leftButton.append(buttons[0])
                                mouse_ML.midButton.append(buttons[1])
                                mouse_ML.rightButton.append(buttons[2])
                                mouse_ML.time.append(mouse_ML.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_3* updates
                    
                    # if sound_3 is starting this frame...
                    if sound_3.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_3.frameNStart = frameN  # exact frame index
                        sound_3.tStart = t  # local t and not account for scr refresh
                        sound_3.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_3.status = STARTED
                        sound_3.play(when=win)  # sync with win flip
                    
                    # if sound_3 is stopping this frame...
                    if sound_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_3.tStartRefresh + 10-frameTolerance or sound_3.isFinished:
                            # keep track of stop time/frame for later
                            sound_3.tStop = t  # not accounting for scr refresh
                            sound_3.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_3.frameNStop = frameN  # exact frame index
                            # update status
                            sound_3.status = FINISHED
                            sound_3.stop()
                    
                    # *key_resp* updates
                    
                    # if key_resp is starting this frame...
                    if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp.frameNStart = frameN  # exact frame index
                        key_resp.tStart = t  # local t and not account for scr refresh
                        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp.status = STARTED
                        # keyboard checking is just starting
                        key_resp.clock.reset()  # now t=0
                    
                    # if key_resp is stopping this frame...
                    if key_resp.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp.tStop = t  # not accounting for scr refresh
                            key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp.status = FINISHED
                            key_resp.status = FINISHED
                    if key_resp.status == STARTED:
                        theseKeys = key_resp.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_allKeys.extend(theseKeys)
                        if len(_key_resp_allKeys):
                            key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                            key_resp.rt = _key_resp_allKeys[-1].rt
                            key_resp.duration = _key_resp_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_ML,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_ML.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_ML.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_ML.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_ML" ---
                for thisComponent in TEST_ML.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_ML
                TEST_ML.tStop = globalClock.getTime(format='float')
                TEST_ML.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_ML.stopped', TEST_ML.tStop)
                # store data for ML_rep (TrialHandler)
                ML_rep.addData('mouse_ML.x', mouse_ML.x)
                ML_rep.addData('mouse_ML.y', mouse_ML.y)
                ML_rep.addData('mouse_ML.leftButton', mouse_ML.leftButton)
                ML_rep.addData('mouse_ML.midButton', mouse_ML.midButton)
                ML_rep.addData('mouse_ML.rightButton', mouse_ML.rightButton)
                ML_rep.addData('mouse_ML.time', mouse_ML.time)
                ML_rep.addData('mouse_ML.clicked_name', mouse_ML.clicked_name)
                sound_3.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_ML.maxDurationReached:
                    routineTimer.addTime(-TEST_ML.maxDuration)
                elif TEST_ML.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisML_rep, 'status') and thisML_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for ML_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                ML_rep.addData('Next_mouse.x', x)
                ML_rep.addData('Next_mouse.y', y)
                ML_rep.addData('Next_mouse.leftButton', buttons[0])
                ML_rep.addData('Next_mouse.midButton', buttons[1])
                ML_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    ML_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisML_rep as finished
                if hasattr(thisML_rep, 'status'):
                    thisML_rep.status = FINISHED
                # if awaiting a pause, pause now
                if ML_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    ML_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'ML_rep'
            ML_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if ML_rep.trialList in ([], [None], None):
                params = []
            else:
                params = ML_rep.trialList[0].keys()
            # save data for this loop
            ML_rep.saveAsExcel(filename + '.xlsx', sheetName='ML_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisA as finished
            if hasattr(thisA, 'status'):
                thisA.status = FINISHED
            # if awaiting a pause, pause now
            if A.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                A.status = STARTED
            thisExp.nextEntry()
            
        # completed ML_repeat repeats of 'A'
        A.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if A.trialList in ([], [None], None):
            params = []
        else:
            params = A.trialList[0].keys()
        # save data for this loop
        A.saveAsExcel(filename + '.xlsx', sheetName='A',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        B = data.TrialHandler2(
            name='B',
            nReps=contrast_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(B)  # add the loop to the experiment
        thisB = B.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisB.rgb)
        if thisB != None:
            for paramName in thisB:
                globals()[paramName] = thisB[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisB in B:
            B.status = STARTED
            if hasattr(thisB, 'status'):
                thisB.status = STARTED
            currentLoop = B
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisB.rgb)
            if thisB != None:
                for paramName in thisB:
                    globals()[paramName] = thisB[paramName]
            
            # --- Prepare to start Routine "Contrast_illusion_INTRO" ---
            # create an object to store info about Routine Contrast_illusion_INTRO
            Contrast_illusion_INTRO = data.Routine(
                name='Contrast_illusion_INTRO',
                components=[contrast_intro_text, sound_4, START_contrast],
            )
            Contrast_illusion_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_4.setSound('Stimuli/Instructions/CONT_INTRO.ogg', hamming=True)
            sound_4.setVolume(1.0, log=False)
            sound_4.seek(0)
            # reset START_contrast to account for continued clicks & clear times on/off
            START_contrast.reset()
            # store start times for Contrast_illusion_INTRO
            Contrast_illusion_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Contrast_illusion_INTRO.tStart = globalClock.getTime(format='float')
            Contrast_illusion_INTRO.status = STARTED
            thisExp.addData('Contrast_illusion_INTRO.started', Contrast_illusion_INTRO.tStart)
            Contrast_illusion_INTRO.maxDuration = None
            # keep track of which components have finished
            Contrast_illusion_INTROComponents = Contrast_illusion_INTRO.components
            for thisComponent in Contrast_illusion_INTRO.components:
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
            
            # --- Run Routine "Contrast_illusion_INTRO" ---
            thisExp.currentRoutine = Contrast_illusion_INTRO
            Contrast_illusion_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisB, 'status') and thisB.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *contrast_intro_text* updates
                
                # if contrast_intro_text is starting this frame...
                if contrast_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    contrast_intro_text.frameNStart = frameN  # exact frame index
                    contrast_intro_text.tStart = t  # local t and not account for scr refresh
                    contrast_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(contrast_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    contrast_intro_text.status = STARTED
                    contrast_intro_text.setAutoDraw(True)
                
                # if contrast_intro_text is active this frame...
                if contrast_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_4* updates
                
                # if sound_4 is starting this frame...
                if sound_4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_4.frameNStart = frameN  # exact frame index
                    sound_4.tStart = t  # local t and not account for scr refresh
                    sound_4.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_4.status = STARTED
                    sound_4.play(when=win)  # sync with win flip
                
                # if sound_4 is stopping this frame...
                if sound_4.status == STARTED:
                    if bool(False) or sound_4.isFinished:
                        # keep track of stop time/frame for later
                        sound_4.tStop = t  # not accounting for scr refresh
                        sound_4.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_4.frameNStop = frameN  # exact frame index
                        # update status
                        sound_4.status = FINISHED
                        sound_4.stop()
                # *START_contrast* updates
                
                # if START_contrast is starting this frame...
                if START_contrast.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_contrast.frameNStart = frameN  # exact frame index
                    START_contrast.tStart = t  # local t and not account for scr refresh
                    START_contrast.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_contrast, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_contrast.status = STARTED
                    win.callOnFlip(START_contrast.buttonClock.reset)
                    START_contrast.setAutoDraw(True)
                
                # if START_contrast is active this frame...
                if START_contrast.status == STARTED:
                    # update params
                    pass
                    # check whether START_contrast has been pressed
                    if START_contrast.isClicked:
                        if not START_contrast.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_contrast.timesOn.append(globalClock.getTime())
                            START_contrast.timesOff.append(globalClock.getTime())
                        elif len(START_contrast.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_contrast.timesOff[-1] = globalClock.getTime()
                        if not START_contrast.wasClicked:
                            # end routine when START_contrast is clicked
                            continueRoutine = False
                        if not START_contrast.wasClicked:
                            # run callback code when START_contrast is clicked
                            pass
                # take note of whether START_contrast was clicked, so that next frame we know if clicks are new
                START_contrast.wasClicked = START_contrast.isClicked and START_contrast.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Contrast_illusion_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    Contrast_illusion_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if Contrast_illusion_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in Contrast_illusion_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Contrast_illusion_INTRO" ---
            for thisComponent in Contrast_illusion_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Contrast_illusion_INTRO
            Contrast_illusion_INTRO.tStop = globalClock.getTime(format='float')
            Contrast_illusion_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Contrast_illusion_INTRO.stopped', Contrast_illusion_INTRO.tStop)
            sound_4.pause()  # ensure sound has stopped at end of Routine
            B.addData('START_contrast.numClicks', START_contrast.numClicks)
            if START_contrast.numClicks:
               B.addData('START_contrast.timesOn', START_contrast.timesOn)
               B.addData('START_contrast.timesOff', START_contrast.timesOff)
            else:
               B.addData('START_contrast.timesOn', "")
               B.addData('START_contrast.timesOff', "")
            # the Routine "Contrast_illusion_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            contrast_rep = data.TrialHandler2(
                name='contrast_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Contrast trials/Contrast_condition_file.xlsx', 
                selection=set_rows_contrast
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(contrast_rep)  # add the loop to the experiment
            thisContrast_rep = contrast_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisContrast_rep.rgb)
            if thisContrast_rep != None:
                for paramName in thisContrast_rep:
                    globals()[paramName] = thisContrast_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisContrast_rep in contrast_rep:
                contrast_rep.status = STARTED
                if hasattr(thisContrast_rep, 'status'):
                    thisContrast_rep.status = STARTED
                currentLoop = contrast_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisContrast_rep.rgb)
                if thisContrast_rep != None:
                    for paramName in thisContrast_rep:
                        globals()[paramName] = thisContrast_rep[paramName]
                
                # --- Prepare to start Routine "TEST_contrast" ---
                # create an object to store info about Routine TEST_contrast
                TEST_contrast = data.Routine(
                    name='TEST_contrast',
                    components=[image_cont, LEFT_cont, RIGHT_cont, mouse_cont, sound_8, key_resp_3],
                )
                TEST_contrast.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_cont.setImage(path)
                # setup some python lists for storing info about the mouse_cont
                mouse_cont.x = []
                mouse_cont.y = []
                mouse_cont.leftButton = []
                mouse_cont.midButton = []
                mouse_cont.rightButton = []
                mouse_cont.time = []
                mouse_cont.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_8.setSound('Stimuli/Instructions/CONT_SHORT.ogg', secs=10, hamming=True)
                sound_8.setVolume(1.0, log=False)
                sound_8.seek(0)
                # create starting attributes for key_resp_3
                key_resp_3.keys = []
                key_resp_3.rt = []
                _key_resp_3_allKeys = []
                # store start times for TEST_contrast
                TEST_contrast.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_contrast.tStart = globalClock.getTime(format='float')
                TEST_contrast.status = STARTED
                thisExp.addData('TEST_contrast.started', TEST_contrast.tStart)
                TEST_contrast.maxDuration = None
                # keep track of which components have finished
                TEST_contrastComponents = TEST_contrast.components
                for thisComponent in TEST_contrast.components:
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
                
                # --- Run Routine "TEST_contrast" ---
                thisExp.currentRoutine = TEST_contrast
                TEST_contrast.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisContrast_rep, 'status') and thisContrast_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_cont* updates
                    
                    # if image_cont is starting this frame...
                    if image_cont.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_cont.frameNStart = frameN  # exact frame index
                        image_cont.tStart = t  # local t and not account for scr refresh
                        image_cont.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_cont, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_cont.status = STARTED
                        image_cont.setAutoDraw(True)
                    
                    # if image_cont is active this frame...
                    if image_cont.status == STARTED:
                        # update params
                        pass
                    
                    # if image_cont is stopping this frame...
                    if image_cont.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_cont.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_cont.tStop = t  # not accounting for scr refresh
                            image_cont.tStopRefresh = tThisFlipGlobal  # on global time
                            image_cont.frameNStop = frameN  # exact frame index
                            # update status
                            image_cont.status = FINISHED
                            image_cont.setAutoDraw(False)
                    
                    # *LEFT_cont* updates
                    
                    # if LEFT_cont is starting this frame...
                    if LEFT_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_cont.frameNStart = frameN  # exact frame index
                        LEFT_cont.tStart = t  # local t and not account for scr refresh
                        LEFT_cont.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_cont, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_cont.status = STARTED
                        LEFT_cont.setAutoDraw(True)
                    
                    # if LEFT_cont is active this frame...
                    if LEFT_cont.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_cont is stopping this frame...
                    if LEFT_cont.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_cont.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_cont.tStop = t  # not accounting for scr refresh
                            LEFT_cont.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_cont.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_cont.status = FINISHED
                            LEFT_cont.setAutoDraw(False)
                    
                    # *RIGHT_cont* updates
                    
                    # if RIGHT_cont is starting this frame...
                    if RIGHT_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_cont.frameNStart = frameN  # exact frame index
                        RIGHT_cont.tStart = t  # local t and not account for scr refresh
                        RIGHT_cont.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_cont, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_cont.status = STARTED
                        RIGHT_cont.setAutoDraw(True)
                    
                    # if RIGHT_cont is active this frame...
                    if RIGHT_cont.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_cont is stopping this frame...
                    if RIGHT_cont.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_cont.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_cont.tStop = t  # not accounting for scr refresh
                            RIGHT_cont.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_cont.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_cont.status = FINISHED
                            RIGHT_cont.setAutoDraw(False)
                    # *mouse_cont* updates
                    
                    # if mouse_cont is starting this frame...
                    if mouse_cont.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_cont.frameNStart = frameN  # exact frame index
                        mouse_cont.tStart = t  # local t and not account for scr refresh
                        mouse_cont.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_cont, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_cont.started', t)
                        # update status
                        mouse_cont.status = STARTED
                        mouse_cont.mouseClock.reset()
                        prevButtonState = mouse_cont.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_cont is stopping this frame...
                    if mouse_cont.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_cont.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_cont.tStop = t  # not accounting for scr refresh
                            mouse_cont.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_cont.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_cont.stopped', t)
                            # update status
                            mouse_cont.status = FINISHED
                    if mouse_cont.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_cont.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_cont, RIGHT_cont], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_cont):
                                        gotValidClick = True
                                        mouse_cont.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_cont.clicked_name.append(None)
                                x, y = mouse_cont.getPos()
                                mouse_cont.x.append(float(x))
                                mouse_cont.y.append(float(y))
                                buttons = mouse_cont.getPressed()
                                mouse_cont.leftButton.append(buttons[0])
                                mouse_cont.midButton.append(buttons[1])
                                mouse_cont.rightButton.append(buttons[2])
                                mouse_cont.time.append(mouse_cont.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_8* updates
                    
                    # if sound_8 is starting this frame...
                    if sound_8.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_8.frameNStart = frameN  # exact frame index
                        sound_8.tStart = t  # local t and not account for scr refresh
                        sound_8.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_8.status = STARTED
                        sound_8.play(when=win)  # sync with win flip
                    
                    # if sound_8 is stopping this frame...
                    if sound_8.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_8.tStartRefresh + 10-frameTolerance or sound_8.isFinished:
                            # keep track of stop time/frame for later
                            sound_8.tStop = t  # not accounting for scr refresh
                            sound_8.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_8.frameNStop = frameN  # exact frame index
                            # update status
                            sound_8.status = FINISHED
                            sound_8.stop()
                    
                    # *key_resp_3* updates
                    
                    # if key_resp_3 is starting this frame...
                    if key_resp_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_3.frameNStart = frameN  # exact frame index
                        key_resp_3.tStart = t  # local t and not account for scr refresh
                        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_3.status = STARTED
                        # keyboard checking is just starting
                        key_resp_3.clock.reset()  # now t=0
                    
                    # if key_resp_3 is stopping this frame...
                    if key_resp_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_3.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_3.tStop = t  # not accounting for scr refresh
                            key_resp_3.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_3.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_3.status = FINISHED
                            key_resp_3.status = FINISHED
                    if key_resp_3.status == STARTED:
                        theseKeys = key_resp_3.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_3_allKeys.extend(theseKeys)
                        if len(_key_resp_3_allKeys):
                            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                            key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_contrast,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_contrast.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_contrast.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_contrast.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_contrast" ---
                for thisComponent in TEST_contrast.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_contrast
                TEST_contrast.tStop = globalClock.getTime(format='float')
                TEST_contrast.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_contrast.stopped', TEST_contrast.tStop)
                # store data for contrast_rep (TrialHandler)
                contrast_rep.addData('mouse_cont.x', mouse_cont.x)
                contrast_rep.addData('mouse_cont.y', mouse_cont.y)
                contrast_rep.addData('mouse_cont.leftButton', mouse_cont.leftButton)
                contrast_rep.addData('mouse_cont.midButton', mouse_cont.midButton)
                contrast_rep.addData('mouse_cont.rightButton', mouse_cont.rightButton)
                contrast_rep.addData('mouse_cont.time', mouse_cont.time)
                contrast_rep.addData('mouse_cont.clicked_name', mouse_cont.clicked_name)
                sound_8.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_contrast.maxDurationReached:
                    routineTimer.addTime(-TEST_contrast.maxDuration)
                elif TEST_contrast.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisContrast_rep, 'status') and thisContrast_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for contrast_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                contrast_rep.addData('Next_mouse.x', x)
                contrast_rep.addData('Next_mouse.y', y)
                contrast_rep.addData('Next_mouse.leftButton', buttons[0])
                contrast_rep.addData('Next_mouse.midButton', buttons[1])
                contrast_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    contrast_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisContrast_rep as finished
                if hasattr(thisContrast_rep, 'status'):
                    thisContrast_rep.status = FINISHED
                # if awaiting a pause, pause now
                if contrast_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    contrast_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'contrast_rep'
            contrast_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if contrast_rep.trialList in ([], [None], None):
                params = []
            else:
                params = contrast_rep.trialList[0].keys()
            # save data for this loop
            contrast_rep.saveAsExcel(filename + '.xlsx', sheetName='contrast_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisB as finished
            if hasattr(thisB, 'status'):
                thisB.status = FINISHED
            # if awaiting a pause, pause now
            if B.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                B.status = STARTED
            thisExp.nextEntry()
            
        # completed contrast_repeat repeats of 'B'
        B.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if B.trialList in ([], [None], None):
            params = []
        else:
            params = B.trialList[0].keys()
        # save data for this loop
        B.saveAsExcel(filename + '.xlsx', sheetName='B',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        C = data.TrialHandler2(
            name='C',
            nReps=ebbing_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(C)  # add the loop to the experiment
        thisC = C.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisC.rgb)
        if thisC != None:
            for paramName in thisC:
                globals()[paramName] = thisC[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisC in C:
            C.status = STARTED
            if hasattr(thisC, 'status'):
                thisC.status = STARTED
            currentLoop = C
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisC.rgb)
            if thisC != None:
                for paramName in thisC:
                    globals()[paramName] = thisC[paramName]
            
            # --- Prepare to start Routine "Ebbing_INTRO" ---
            # create an object to store info about Routine Ebbing_INTRO
            Ebbing_INTRO = data.Routine(
                name='Ebbing_INTRO',
                components=[ebbing_intro_text, sound_5, START_ebbing],
            )
            Ebbing_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_5.setSound('Stimuli/Instructions/EBB_INTRO.ogg', hamming=True)
            sound_5.setVolume(1.0, log=False)
            sound_5.seek(0)
            # reset START_ebbing to account for continued clicks & clear times on/off
            START_ebbing.reset()
            # store start times for Ebbing_INTRO
            Ebbing_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Ebbing_INTRO.tStart = globalClock.getTime(format='float')
            Ebbing_INTRO.status = STARTED
            thisExp.addData('Ebbing_INTRO.started', Ebbing_INTRO.tStart)
            Ebbing_INTRO.maxDuration = None
            # keep track of which components have finished
            Ebbing_INTROComponents = Ebbing_INTRO.components
            for thisComponent in Ebbing_INTRO.components:
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
            
            # --- Run Routine "Ebbing_INTRO" ---
            thisExp.currentRoutine = Ebbing_INTRO
            Ebbing_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisC, 'status') and thisC.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *ebbing_intro_text* updates
                
                # if ebbing_intro_text is starting this frame...
                if ebbing_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    ebbing_intro_text.frameNStart = frameN  # exact frame index
                    ebbing_intro_text.tStart = t  # local t and not account for scr refresh
                    ebbing_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ebbing_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    ebbing_intro_text.status = STARTED
                    ebbing_intro_text.setAutoDraw(True)
                
                # if ebbing_intro_text is active this frame...
                if ebbing_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_5* updates
                
                # if sound_5 is starting this frame...
                if sound_5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_5.frameNStart = frameN  # exact frame index
                    sound_5.tStart = t  # local t and not account for scr refresh
                    sound_5.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_5.status = STARTED
                    sound_5.play(when=win)  # sync with win flip
                
                # if sound_5 is stopping this frame...
                if sound_5.status == STARTED:
                    if bool(False) or sound_5.isFinished:
                        # keep track of stop time/frame for later
                        sound_5.tStop = t  # not accounting for scr refresh
                        sound_5.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_5.frameNStop = frameN  # exact frame index
                        # update status
                        sound_5.status = FINISHED
                        sound_5.stop()
                # *START_ebbing* updates
                
                # if START_ebbing is starting this frame...
                if START_ebbing.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_ebbing.frameNStart = frameN  # exact frame index
                    START_ebbing.tStart = t  # local t and not account for scr refresh
                    START_ebbing.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_ebbing, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_ebbing.status = STARTED
                    win.callOnFlip(START_ebbing.buttonClock.reset)
                    START_ebbing.setAutoDraw(True)
                
                # if START_ebbing is active this frame...
                if START_ebbing.status == STARTED:
                    # update params
                    pass
                    # check whether START_ebbing has been pressed
                    if START_ebbing.isClicked:
                        if not START_ebbing.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_ebbing.timesOn.append(globalClock.getTime())
                            START_ebbing.timesOff.append(globalClock.getTime())
                        elif len(START_ebbing.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_ebbing.timesOff[-1] = globalClock.getTime()
                        if not START_ebbing.wasClicked:
                            # end routine when START_ebbing is clicked
                            continueRoutine = False
                        if not START_ebbing.wasClicked:
                            # run callback code when START_ebbing is clicked
                            pass
                # take note of whether START_ebbing was clicked, so that next frame we know if clicks are new
                START_ebbing.wasClicked = START_ebbing.isClicked and START_ebbing.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Ebbing_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    Ebbing_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if Ebbing_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in Ebbing_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Ebbing_INTRO" ---
            for thisComponent in Ebbing_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Ebbing_INTRO
            Ebbing_INTRO.tStop = globalClock.getTime(format='float')
            Ebbing_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Ebbing_INTRO.stopped', Ebbing_INTRO.tStop)
            sound_5.pause()  # ensure sound has stopped at end of Routine
            C.addData('START_ebbing.numClicks', START_ebbing.numClicks)
            if START_ebbing.numClicks:
               C.addData('START_ebbing.timesOn', START_ebbing.timesOn)
               C.addData('START_ebbing.timesOff', START_ebbing.timesOff)
            else:
               C.addData('START_ebbing.timesOn', "")
               C.addData('START_ebbing.timesOff', "")
            # the Routine "Ebbing_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            ebbing_rep = data.TrialHandler2(
                name='ebbing_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Ebbinghaus trials/Ebbinghaus_condition_file.xlsx', 
                selection=set_rows_ebbinghaus
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(ebbing_rep)  # add the loop to the experiment
            thisEbbing_rep = ebbing_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisEbbing_rep.rgb)
            if thisEbbing_rep != None:
                for paramName in thisEbbing_rep:
                    globals()[paramName] = thisEbbing_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisEbbing_rep in ebbing_rep:
                ebbing_rep.status = STARTED
                if hasattr(thisEbbing_rep, 'status'):
                    thisEbbing_rep.status = STARTED
                currentLoop = ebbing_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisEbbing_rep.rgb)
                if thisEbbing_rep != None:
                    for paramName in thisEbbing_rep:
                        globals()[paramName] = thisEbbing_rep[paramName]
                
                # --- Prepare to start Routine "TEST_ebbing" ---
                # create an object to store info about Routine TEST_ebbing
                TEST_ebbing = data.Routine(
                    name='TEST_ebbing',
                    components=[image_ebbing, RIGHT_ebbing, LEFT_ebbing, mouse_ebbing, sound_9, key_resp_4],
                )
                TEST_ebbing.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_ebbing.setImage(path)
                # setup some python lists for storing info about the mouse_ebbing
                mouse_ebbing.x = []
                mouse_ebbing.y = []
                mouse_ebbing.leftButton = []
                mouse_ebbing.midButton = []
                mouse_ebbing.rightButton = []
                mouse_ebbing.time = []
                mouse_ebbing.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_9.setSound('Stimuli/Instructions/EBB_SHORT.ogg', secs=10, hamming=True)
                sound_9.setVolume(1.0, log=False)
                sound_9.seek(0)
                # create starting attributes for key_resp_4
                key_resp_4.keys = []
                key_resp_4.rt = []
                _key_resp_4_allKeys = []
                # store start times for TEST_ebbing
                TEST_ebbing.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_ebbing.tStart = globalClock.getTime(format='float')
                TEST_ebbing.status = STARTED
                thisExp.addData('TEST_ebbing.started', TEST_ebbing.tStart)
                TEST_ebbing.maxDuration = None
                # keep track of which components have finished
                TEST_ebbingComponents = TEST_ebbing.components
                for thisComponent in TEST_ebbing.components:
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
                
                # --- Run Routine "TEST_ebbing" ---
                thisExp.currentRoutine = TEST_ebbing
                TEST_ebbing.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisEbbing_rep, 'status') and thisEbbing_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_ebbing* updates
                    
                    # if image_ebbing is starting this frame...
                    if image_ebbing.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_ebbing.frameNStart = frameN  # exact frame index
                        image_ebbing.tStart = t  # local t and not account for scr refresh
                        image_ebbing.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_ebbing, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_ebbing.status = STARTED
                        image_ebbing.setAutoDraw(True)
                    
                    # if image_ebbing is active this frame...
                    if image_ebbing.status == STARTED:
                        # update params
                        pass
                    
                    # if image_ebbing is stopping this frame...
                    if image_ebbing.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_ebbing.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_ebbing.tStop = t  # not accounting for scr refresh
                            image_ebbing.tStopRefresh = tThisFlipGlobal  # on global time
                            image_ebbing.frameNStop = frameN  # exact frame index
                            # update status
                            image_ebbing.status = FINISHED
                            image_ebbing.setAutoDraw(False)
                    
                    # *RIGHT_ebbing* updates
                    
                    # if RIGHT_ebbing is starting this frame...
                    if RIGHT_ebbing.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_ebbing.frameNStart = frameN  # exact frame index
                        RIGHT_ebbing.tStart = t  # local t and not account for scr refresh
                        RIGHT_ebbing.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_ebbing, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_ebbing.status = STARTED
                        RIGHT_ebbing.setAutoDraw(True)
                    
                    # if RIGHT_ebbing is active this frame...
                    if RIGHT_ebbing.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_ebbing is stopping this frame...
                    if RIGHT_ebbing.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_ebbing.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_ebbing.tStop = t  # not accounting for scr refresh
                            RIGHT_ebbing.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_ebbing.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_ebbing.status = FINISHED
                            RIGHT_ebbing.setAutoDraw(False)
                    
                    # *LEFT_ebbing* updates
                    
                    # if LEFT_ebbing is starting this frame...
                    if LEFT_ebbing.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_ebbing.frameNStart = frameN  # exact frame index
                        LEFT_ebbing.tStart = t  # local t and not account for scr refresh
                        LEFT_ebbing.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_ebbing, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_ebbing.status = STARTED
                        LEFT_ebbing.setAutoDraw(True)
                    
                    # if LEFT_ebbing is active this frame...
                    if LEFT_ebbing.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_ebbing is stopping this frame...
                    if LEFT_ebbing.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_ebbing.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_ebbing.tStop = t  # not accounting for scr refresh
                            LEFT_ebbing.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_ebbing.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_ebbing.status = FINISHED
                            LEFT_ebbing.setAutoDraw(False)
                    # *mouse_ebbing* updates
                    
                    # if mouse_ebbing is starting this frame...
                    if mouse_ebbing.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_ebbing.frameNStart = frameN  # exact frame index
                        mouse_ebbing.tStart = t  # local t and not account for scr refresh
                        mouse_ebbing.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_ebbing, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_ebbing.started', t)
                        # update status
                        mouse_ebbing.status = STARTED
                        mouse_ebbing.mouseClock.reset()
                        prevButtonState = mouse_ebbing.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_ebbing is stopping this frame...
                    if mouse_ebbing.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_ebbing.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_ebbing.tStop = t  # not accounting for scr refresh
                            mouse_ebbing.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_ebbing.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_ebbing.stopped', t)
                            # update status
                            mouse_ebbing.status = FINISHED
                    if mouse_ebbing.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_ebbing.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_ebbing, RIGHT_ebbing], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_ebbing):
                                        gotValidClick = True
                                        mouse_ebbing.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_ebbing.clicked_name.append(None)
                                x, y = mouse_ebbing.getPos()
                                mouse_ebbing.x.append(float(x))
                                mouse_ebbing.y.append(float(y))
                                buttons = mouse_ebbing.getPressed()
                                mouse_ebbing.leftButton.append(buttons[0])
                                mouse_ebbing.midButton.append(buttons[1])
                                mouse_ebbing.rightButton.append(buttons[2])
                                mouse_ebbing.time.append(mouse_ebbing.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_9* updates
                    
                    # if sound_9 is starting this frame...
                    if sound_9.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_9.frameNStart = frameN  # exact frame index
                        sound_9.tStart = t  # local t and not account for scr refresh
                        sound_9.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_9.status = STARTED
                        sound_9.play(when=win)  # sync with win flip
                    
                    # if sound_9 is stopping this frame...
                    if sound_9.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_9.tStartRefresh + 10-frameTolerance or sound_9.isFinished:
                            # keep track of stop time/frame for later
                            sound_9.tStop = t  # not accounting for scr refresh
                            sound_9.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_9.frameNStop = frameN  # exact frame index
                            # update status
                            sound_9.status = FINISHED
                            sound_9.stop()
                    
                    # *key_resp_4* updates
                    
                    # if key_resp_4 is starting this frame...
                    if key_resp_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_4.frameNStart = frameN  # exact frame index
                        key_resp_4.tStart = t  # local t and not account for scr refresh
                        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_4.status = STARTED
                        # keyboard checking is just starting
                        key_resp_4.clock.reset()  # now t=0
                    
                    # if key_resp_4 is stopping this frame...
                    if key_resp_4.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_4.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_4.tStop = t  # not accounting for scr refresh
                            key_resp_4.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_4.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_4.status = FINISHED
                            key_resp_4.status = FINISHED
                    if key_resp_4.status == STARTED:
                        theseKeys = key_resp_4.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_4_allKeys.extend(theseKeys)
                        if len(_key_resp_4_allKeys):
                            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                            key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_ebbing,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_ebbing.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_ebbing.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_ebbing.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_ebbing" ---
                for thisComponent in TEST_ebbing.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_ebbing
                TEST_ebbing.tStop = globalClock.getTime(format='float')
                TEST_ebbing.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_ebbing.stopped', TEST_ebbing.tStop)
                # store data for ebbing_rep (TrialHandler)
                ebbing_rep.addData('mouse_ebbing.x', mouse_ebbing.x)
                ebbing_rep.addData('mouse_ebbing.y', mouse_ebbing.y)
                ebbing_rep.addData('mouse_ebbing.leftButton', mouse_ebbing.leftButton)
                ebbing_rep.addData('mouse_ebbing.midButton', mouse_ebbing.midButton)
                ebbing_rep.addData('mouse_ebbing.rightButton', mouse_ebbing.rightButton)
                ebbing_rep.addData('mouse_ebbing.time', mouse_ebbing.time)
                ebbing_rep.addData('mouse_ebbing.clicked_name', mouse_ebbing.clicked_name)
                sound_9.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_ebbing.maxDurationReached:
                    routineTimer.addTime(-TEST_ebbing.maxDuration)
                elif TEST_ebbing.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisEbbing_rep, 'status') and thisEbbing_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for ebbing_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                ebbing_rep.addData('Next_mouse.x', x)
                ebbing_rep.addData('Next_mouse.y', y)
                ebbing_rep.addData('Next_mouse.leftButton', buttons[0])
                ebbing_rep.addData('Next_mouse.midButton', buttons[1])
                ebbing_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    ebbing_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisEbbing_rep as finished
                if hasattr(thisEbbing_rep, 'status'):
                    thisEbbing_rep.status = FINISHED
                # if awaiting a pause, pause now
                if ebbing_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    ebbing_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'ebbing_rep'
            ebbing_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if ebbing_rep.trialList in ([], [None], None):
                params = []
            else:
                params = ebbing_rep.trialList[0].keys()
            # save data for this loop
            ebbing_rep.saveAsExcel(filename + '.xlsx', sheetName='ebbing_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisC as finished
            if hasattr(thisC, 'status'):
                thisC.status = FINISHED
            # if awaiting a pause, pause now
            if C.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                C.status = STARTED
            thisExp.nextEntry()
            
        # completed ebbing_repeat repeats of 'C'
        C.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if C.trialList in ([], [None], None):
            params = []
        else:
            params = C.trialList[0].keys()
        # save data for this loop
        C.saveAsExcel(filename + '.xlsx', sheetName='C',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        D = data.TrialHandler2(
            name='D',
            nReps=kanizsa_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(D)  # add the loop to the experiment
        thisD = D.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisD.rgb)
        if thisD != None:
            for paramName in thisD:
                globals()[paramName] = thisD[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisD in D:
            D.status = STARTED
            if hasattr(thisD, 'status'):
                thisD.status = STARTED
            currentLoop = D
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisD.rgb)
            if thisD != None:
                for paramName in thisD:
                    globals()[paramName] = thisD[paramName]
            
            # --- Prepare to start Routine "Kanizsa_INTRO" ---
            # create an object to store info about Routine Kanizsa_INTRO
            Kanizsa_INTRO = data.Routine(
                name='Kanizsa_INTRO',
                components=[kanizsa_intro_text, sound_6, START_KN],
            )
            Kanizsa_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_6.setSound('Stimuli/Instructions/KZ_INTRO.ogg', hamming=True)
            sound_6.setVolume(1.0, log=False)
            sound_6.seek(0)
            # reset START_KN to account for continued clicks & clear times on/off
            START_KN.reset()
            # store start times for Kanizsa_INTRO
            Kanizsa_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Kanizsa_INTRO.tStart = globalClock.getTime(format='float')
            Kanizsa_INTRO.status = STARTED
            thisExp.addData('Kanizsa_INTRO.started', Kanizsa_INTRO.tStart)
            Kanizsa_INTRO.maxDuration = None
            # keep track of which components have finished
            Kanizsa_INTROComponents = Kanizsa_INTRO.components
            for thisComponent in Kanizsa_INTRO.components:
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
            
            # --- Run Routine "Kanizsa_INTRO" ---
            thisExp.currentRoutine = Kanizsa_INTRO
            Kanizsa_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisD, 'status') and thisD.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *kanizsa_intro_text* updates
                
                # if kanizsa_intro_text is starting this frame...
                if kanizsa_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    kanizsa_intro_text.frameNStart = frameN  # exact frame index
                    kanizsa_intro_text.tStart = t  # local t and not account for scr refresh
                    kanizsa_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(kanizsa_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    kanizsa_intro_text.status = STARTED
                    kanizsa_intro_text.setAutoDraw(True)
                
                # if kanizsa_intro_text is active this frame...
                if kanizsa_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_6* updates
                
                # if sound_6 is starting this frame...
                if sound_6.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_6.frameNStart = frameN  # exact frame index
                    sound_6.tStart = t  # local t and not account for scr refresh
                    sound_6.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_6.status = STARTED
                    sound_6.play(when=win)  # sync with win flip
                
                # if sound_6 is stopping this frame...
                if sound_6.status == STARTED:
                    if bool(False) or sound_6.isFinished:
                        # keep track of stop time/frame for later
                        sound_6.tStop = t  # not accounting for scr refresh
                        sound_6.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_6.frameNStop = frameN  # exact frame index
                        # update status
                        sound_6.status = FINISHED
                        sound_6.stop()
                # *START_KN* updates
                
                # if START_KN is starting this frame...
                if START_KN.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_KN.frameNStart = frameN  # exact frame index
                    START_KN.tStart = t  # local t and not account for scr refresh
                    START_KN.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_KN, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_KN.status = STARTED
                    win.callOnFlip(START_KN.buttonClock.reset)
                    START_KN.setAutoDraw(True)
                
                # if START_KN is active this frame...
                if START_KN.status == STARTED:
                    # update params
                    pass
                    # check whether START_KN has been pressed
                    if START_KN.isClicked:
                        if not START_KN.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_KN.timesOn.append(globalClock.getTime())
                            START_KN.timesOff.append(globalClock.getTime())
                        elif len(START_KN.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_KN.timesOff[-1] = globalClock.getTime()
                        if not START_KN.wasClicked:
                            # end routine when START_KN is clicked
                            continueRoutine = False
                        if not START_KN.wasClicked:
                            # run callback code when START_KN is clicked
                            pass
                # take note of whether START_KN was clicked, so that next frame we know if clicks are new
                START_KN.wasClicked = START_KN.isClicked and START_KN.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Kanizsa_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    Kanizsa_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if Kanizsa_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in Kanizsa_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Kanizsa_INTRO" ---
            for thisComponent in Kanizsa_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Kanizsa_INTRO
            Kanizsa_INTRO.tStop = globalClock.getTime(format='float')
            Kanizsa_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Kanizsa_INTRO.stopped', Kanizsa_INTRO.tStop)
            sound_6.pause()  # ensure sound has stopped at end of Routine
            D.addData('START_KN.numClicks', START_KN.numClicks)
            if START_KN.numClicks:
               D.addData('START_KN.timesOn', START_KN.timesOn)
               D.addData('START_KN.timesOff', START_KN.timesOff)
            else:
               D.addData('START_KN.timesOn', "")
               D.addData('START_KN.timesOff', "")
            # the Routine "Kanizsa_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trainingKZS" ---
            # create an object to store info about Routine trainingKZS
            trainingKZS = data.Routine(
                name='trainingKZS',
                components=[triangle, threefourth, twofourth, whitesquare, mouse_2],
            )
            trainingKZS.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the mouse_2
            mouse_2.clicked_name = []
            gotValidClick = False  # until a click is received
            # store start times for trainingKZS
            trainingKZS.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trainingKZS.tStart = globalClock.getTime(format='float')
            trainingKZS.status = STARTED
            thisExp.addData('trainingKZS.started', trainingKZS.tStart)
            trainingKZS.maxDuration = None
            # keep track of which components have finished
            trainingKZSComponents = trainingKZS.components
            for thisComponent in trainingKZS.components:
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
            
            # --- Run Routine "trainingKZS" ---
            thisExp.currentRoutine = trainingKZS
            trainingKZS.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisD, 'status') and thisD.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *triangle* updates
                
                # if triangle is starting this frame...
                if triangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    triangle.frameNStart = frameN  # exact frame index
                    triangle.tStart = t  # local t and not account for scr refresh
                    triangle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(triangle, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    triangle.status = STARTED
                    triangle.setAutoDraw(True)
                
                # if triangle is active this frame...
                if triangle.status == STARTED:
                    # update params
                    pass
                
                # *threefourth* updates
                
                # if threefourth is starting this frame...
                if threefourth.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    threefourth.frameNStart = frameN  # exact frame index
                    threefourth.tStart = t  # local t and not account for scr refresh
                    threefourth.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(threefourth, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    threefourth.status = STARTED
                    threefourth.setAutoDraw(True)
                
                # if threefourth is active this frame...
                if threefourth.status == STARTED:
                    # update params
                    pass
                
                # *twofourth* updates
                
                # if twofourth is starting this frame...
                if twofourth.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    twofourth.frameNStart = frameN  # exact frame index
                    twofourth.tStart = t  # local t and not account for scr refresh
                    twofourth.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(twofourth, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    twofourth.status = STARTED
                    twofourth.setAutoDraw(True)
                
                # if twofourth is active this frame...
                if twofourth.status == STARTED:
                    # update params
                    pass
                
                # *whitesquare* updates
                
                # if whitesquare is starting this frame...
                if whitesquare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    whitesquare.frameNStart = frameN  # exact frame index
                    whitesquare.tStart = t  # local t and not account for scr refresh
                    whitesquare.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(whitesquare, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'whitesquare.started')
                    # update status
                    whitesquare.status = STARTED
                    whitesquare.setAutoDraw(True)
                
                # if whitesquare is active this frame...
                if whitesquare.status == STARTED:
                    # update params
                    pass
                # *mouse_2* updates
                
                # if mouse_2 is starting this frame...
                if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_2.frameNStart = frameN  # exact frame index
                    mouse_2.tStart = t  # local t and not account for scr refresh
                    mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouse_2.status = STARTED
                    mouse_2.mouseClock.reset()
                    prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
                if mouse_2.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_2.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(whitesquare, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse_2):
                                    gotValidClick = True
                                    mouse_2.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse_2.clicked_name.append(None)
                            if gotValidClick:  
                                continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=trainingKZS,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    trainingKZS.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if trainingKZS.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in trainingKZS.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trainingKZS" ---
            for thisComponent in trainingKZS.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trainingKZS
            trainingKZS.tStop = globalClock.getTime(format='float')
            trainingKZS.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trainingKZS.stopped', trainingKZS.tStop)
            # store data for D (TrialHandler)
            x, y = mouse_2.getPos()
            buttons = mouse_2.getPressed()
            if sum(buttons):
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                clickableList = environmenttools.getFromNames(whitesquare, namespace=locals())
                for obj in clickableList:
                    # is this object clicked on?
                    if obj.contains(mouse_2):
                        gotValidClick = True
                        mouse_2.clicked_name.append(obj.name)
                if not gotValidClick:
                    mouse_2.clicked_name.append(None)
            D.addData('mouse_2.x', x)
            D.addData('mouse_2.y', y)
            D.addData('mouse_2.leftButton', buttons[0])
            D.addData('mouse_2.midButton', buttons[1])
            D.addData('mouse_2.rightButton', buttons[2])
            if len(mouse_2.clicked_name):
                D.addData('mouse_2.clicked_name', mouse_2.clicked_name[0])
            # the Routine "trainingKZS" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trainingKZS_2" ---
            # create an object to store info about Routine trainingKZS_2
            trainingKZS_2 = data.Routine(
                name='trainingKZS_2',
                components=[triangle_2, threefourth_2, twofourth_2, whitesquare_2, mouse_3],
            )
            trainingKZS_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the mouse_3
            mouse_3.clicked_name = []
            gotValidClick = False  # until a click is received
            # store start times for trainingKZS_2
            trainingKZS_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trainingKZS_2.tStart = globalClock.getTime(format='float')
            trainingKZS_2.status = STARTED
            thisExp.addData('trainingKZS_2.started', trainingKZS_2.tStart)
            trainingKZS_2.maxDuration = None
            # keep track of which components have finished
            trainingKZS_2Components = trainingKZS_2.components
            for thisComponent in trainingKZS_2.components:
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
            
            # --- Run Routine "trainingKZS_2" ---
            thisExp.currentRoutine = trainingKZS_2
            trainingKZS_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisD, 'status') and thisD.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *triangle_2* updates
                
                # if triangle_2 is starting this frame...
                if triangle_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    triangle_2.frameNStart = frameN  # exact frame index
                    triangle_2.tStart = t  # local t and not account for scr refresh
                    triangle_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(triangle_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    triangle_2.status = STARTED
                    triangle_2.setAutoDraw(True)
                
                # if triangle_2 is active this frame...
                if triangle_2.status == STARTED:
                    # update params
                    pass
                
                # *threefourth_2* updates
                
                # if threefourth_2 is starting this frame...
                if threefourth_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    threefourth_2.frameNStart = frameN  # exact frame index
                    threefourth_2.tStart = t  # local t and not account for scr refresh
                    threefourth_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(threefourth_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    threefourth_2.status = STARTED
                    threefourth_2.setAutoDraw(True)
                
                # if threefourth_2 is active this frame...
                if threefourth_2.status == STARTED:
                    # update params
                    pass
                
                # *twofourth_2* updates
                
                # if twofourth_2 is starting this frame...
                if twofourth_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    twofourth_2.frameNStart = frameN  # exact frame index
                    twofourth_2.tStart = t  # local t and not account for scr refresh
                    twofourth_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(twofourth_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    twofourth_2.status = STARTED
                    twofourth_2.setAutoDraw(True)
                
                # if twofourth_2 is active this frame...
                if twofourth_2.status == STARTED:
                    # update params
                    pass
                
                # *whitesquare_2* updates
                
                # if whitesquare_2 is starting this frame...
                if whitesquare_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    whitesquare_2.frameNStart = frameN  # exact frame index
                    whitesquare_2.tStart = t  # local t and not account for scr refresh
                    whitesquare_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(whitesquare_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    whitesquare_2.status = STARTED
                    whitesquare_2.setAutoDraw(True)
                
                # if whitesquare_2 is active this frame...
                if whitesquare_2.status == STARTED:
                    # update params
                    pass
                # *mouse_3* updates
                
                # if mouse_3 is starting this frame...
                if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_3.frameNStart = frameN  # exact frame index
                    mouse_3.tStart = t  # local t and not account for scr refresh
                    mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse_3.started', t)
                    # update status
                    mouse_3.status = STARTED
                    mouse_3.mouseClock.reset()
                    prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
                if mouse_3.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_3.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(whitesquare_2, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse_3):
                                    gotValidClick = True
                                    mouse_3.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse_3.clicked_name.append(None)
                            if gotValidClick:  
                                continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=trainingKZS_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    trainingKZS_2.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if trainingKZS_2.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in trainingKZS_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trainingKZS_2" ---
            for thisComponent in trainingKZS_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trainingKZS_2
            trainingKZS_2.tStop = globalClock.getTime(format='float')
            trainingKZS_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trainingKZS_2.stopped', trainingKZS_2.tStop)
            # store data for D (TrialHandler)
            x, y = mouse_3.getPos()
            buttons = mouse_3.getPressed()
            if sum(buttons):
                # check if the mouse was inside our 'clickable' objects
                gotValidClick = False
                clickableList = environmenttools.getFromNames(whitesquare_2, namespace=locals())
                for obj in clickableList:
                    # is this object clicked on?
                    if obj.contains(mouse_3):
                        gotValidClick = True
                        mouse_3.clicked_name.append(obj.name)
                if not gotValidClick:
                    mouse_3.clicked_name.append(None)
            D.addData('mouse_3.x', x)
            D.addData('mouse_3.y', y)
            D.addData('mouse_3.leftButton', buttons[0])
            D.addData('mouse_3.midButton', buttons[1])
            D.addData('mouse_3.rightButton', buttons[2])
            if len(mouse_3.clicked_name):
                D.addData('mouse_3.clicked_name', mouse_3.clicked_name[0])
            # the Routine "trainingKZS_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            KN_rep = data.TrialHandler2(
                name='KN_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Kanizsa trials/KN_condition_file.xlsx', 
                selection=set_rows_kanizsa
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(KN_rep)  # add the loop to the experiment
            thisKN_rep = KN_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisKN_rep.rgb)
            if thisKN_rep != None:
                for paramName in thisKN_rep:
                    globals()[paramName] = thisKN_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisKN_rep in KN_rep:
                KN_rep.status = STARTED
                if hasattr(thisKN_rep, 'status'):
                    thisKN_rep.status = STARTED
                currentLoop = KN_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisKN_rep.rgb)
                if thisKN_rep != None:
                    for paramName in thisKN_rep:
                        globals()[paramName] = thisKN_rep[paramName]
                
                # --- Prepare to start Routine "TEST_knzs" ---
                # create an object to store info about Routine TEST_knzs
                TEST_knzs = data.Routine(
                    name='TEST_knzs',
                    components=[image_KN, LEFT_KN, RIGHT_KN, mouse_KN, sound_11, key_resp_5],
                )
                TEST_knzs.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_KN.setImage(path)
                # setup some python lists for storing info about the mouse_KN
                mouse_KN.x = []
                mouse_KN.y = []
                mouse_KN.leftButton = []
                mouse_KN.midButton = []
                mouse_KN.rightButton = []
                mouse_KN.time = []
                mouse_KN.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_11.setSound('Stimuli/Instructions/KZ_SHORT.ogg', secs=10, hamming=True)
                sound_11.setVolume(1.0, log=False)
                sound_11.seek(0)
                # create starting attributes for key_resp_5
                key_resp_5.keys = []
                key_resp_5.rt = []
                _key_resp_5_allKeys = []
                # store start times for TEST_knzs
                TEST_knzs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_knzs.tStart = globalClock.getTime(format='float')
                TEST_knzs.status = STARTED
                thisExp.addData('TEST_knzs.started', TEST_knzs.tStart)
                TEST_knzs.maxDuration = None
                # keep track of which components have finished
                TEST_knzsComponents = TEST_knzs.components
                for thisComponent in TEST_knzs.components:
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
                
                # --- Run Routine "TEST_knzs" ---
                thisExp.currentRoutine = TEST_knzs
                TEST_knzs.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisKN_rep, 'status') and thisKN_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_KN* updates
                    
                    # if image_KN is starting this frame...
                    if image_KN.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_KN.frameNStart = frameN  # exact frame index
                        image_KN.tStart = t  # local t and not account for scr refresh
                        image_KN.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_KN, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_KN.status = STARTED
                        image_KN.setAutoDraw(True)
                    
                    # if image_KN is active this frame...
                    if image_KN.status == STARTED:
                        # update params
                        pass
                    
                    # if image_KN is stopping this frame...
                    if image_KN.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_KN.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_KN.tStop = t  # not accounting for scr refresh
                            image_KN.tStopRefresh = tThisFlipGlobal  # on global time
                            image_KN.frameNStop = frameN  # exact frame index
                            # update status
                            image_KN.status = FINISHED
                            image_KN.setAutoDraw(False)
                    
                    # *LEFT_KN* updates
                    
                    # if LEFT_KN is starting this frame...
                    if LEFT_KN.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_KN.frameNStart = frameN  # exact frame index
                        LEFT_KN.tStart = t  # local t and not account for scr refresh
                        LEFT_KN.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_KN, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_KN.status = STARTED
                        LEFT_KN.setAutoDraw(True)
                    
                    # if LEFT_KN is active this frame...
                    if LEFT_KN.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_KN is stopping this frame...
                    if LEFT_KN.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_KN.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_KN.tStop = t  # not accounting for scr refresh
                            LEFT_KN.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_KN.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_KN.status = FINISHED
                            LEFT_KN.setAutoDraw(False)
                    
                    # *RIGHT_KN* updates
                    
                    # if RIGHT_KN is starting this frame...
                    if RIGHT_KN.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_KN.frameNStart = frameN  # exact frame index
                        RIGHT_KN.tStart = t  # local t and not account for scr refresh
                        RIGHT_KN.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_KN, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_KN.status = STARTED
                        RIGHT_KN.setAutoDraw(True)
                    
                    # if RIGHT_KN is active this frame...
                    if RIGHT_KN.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_KN is stopping this frame...
                    if RIGHT_KN.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_KN.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_KN.tStop = t  # not accounting for scr refresh
                            RIGHT_KN.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_KN.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_KN.status = FINISHED
                            RIGHT_KN.setAutoDraw(False)
                    # *mouse_KN* updates
                    
                    # if mouse_KN is starting this frame...
                    if mouse_KN.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_KN.frameNStart = frameN  # exact frame index
                        mouse_KN.tStart = t  # local t and not account for scr refresh
                        mouse_KN.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_KN, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_KN.started', t)
                        # update status
                        mouse_KN.status = STARTED
                        mouse_KN.mouseClock.reset()
                        prevButtonState = mouse_KN.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_KN is stopping this frame...
                    if mouse_KN.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_KN.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_KN.tStop = t  # not accounting for scr refresh
                            mouse_KN.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_KN.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_KN.stopped', t)
                            # update status
                            mouse_KN.status = FINISHED
                    if mouse_KN.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_KN.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_KN, RIGHT_KN], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_KN):
                                        gotValidClick = True
                                        mouse_KN.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_KN.clicked_name.append(None)
                                x, y = mouse_KN.getPos()
                                mouse_KN.x.append(float(x))
                                mouse_KN.y.append(float(y))
                                buttons = mouse_KN.getPressed()
                                mouse_KN.leftButton.append(buttons[0])
                                mouse_KN.midButton.append(buttons[1])
                                mouse_KN.rightButton.append(buttons[2])
                                mouse_KN.time.append(mouse_KN.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_11* updates
                    
                    # if sound_11 is starting this frame...
                    if sound_11.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_11.frameNStart = frameN  # exact frame index
                        sound_11.tStart = t  # local t and not account for scr refresh
                        sound_11.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_11.status = STARTED
                        sound_11.play(when=win)  # sync with win flip
                    
                    # if sound_11 is stopping this frame...
                    if sound_11.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_11.tStartRefresh + 10-frameTolerance or sound_11.isFinished:
                            # keep track of stop time/frame for later
                            sound_11.tStop = t  # not accounting for scr refresh
                            sound_11.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_11.frameNStop = frameN  # exact frame index
                            # update status
                            sound_11.status = FINISHED
                            sound_11.stop()
                    
                    # *key_resp_5* updates
                    
                    # if key_resp_5 is starting this frame...
                    if key_resp_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_5.frameNStart = frameN  # exact frame index
                        key_resp_5.tStart = t  # local t and not account for scr refresh
                        key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_5.status = STARTED
                        # keyboard checking is just starting
                        key_resp_5.clock.reset()  # now t=0
                    
                    # if key_resp_5 is stopping this frame...
                    if key_resp_5.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_5.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_5.tStop = t  # not accounting for scr refresh
                            key_resp_5.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_5.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_5.status = FINISHED
                            key_resp_5.status = FINISHED
                    if key_resp_5.status == STARTED:
                        theseKeys = key_resp_5.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_5_allKeys.extend(theseKeys)
                        if len(_key_resp_5_allKeys):
                            key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                            key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                            key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_knzs,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_knzs.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_knzs.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_knzs.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_knzs" ---
                for thisComponent in TEST_knzs.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_knzs
                TEST_knzs.tStop = globalClock.getTime(format='float')
                TEST_knzs.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_knzs.stopped', TEST_knzs.tStop)
                # store data for KN_rep (TrialHandler)
                KN_rep.addData('mouse_KN.x', mouse_KN.x)
                KN_rep.addData('mouse_KN.y', mouse_KN.y)
                KN_rep.addData('mouse_KN.leftButton', mouse_KN.leftButton)
                KN_rep.addData('mouse_KN.midButton', mouse_KN.midButton)
                KN_rep.addData('mouse_KN.rightButton', mouse_KN.rightButton)
                KN_rep.addData('mouse_KN.time', mouse_KN.time)
                KN_rep.addData('mouse_KN.clicked_name', mouse_KN.clicked_name)
                sound_11.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_knzs.maxDurationReached:
                    routineTimer.addTime(-TEST_knzs.maxDuration)
                elif TEST_knzs.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisKN_rep, 'status') and thisKN_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for KN_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                KN_rep.addData('Next_mouse.x', x)
                KN_rep.addData('Next_mouse.y', y)
                KN_rep.addData('Next_mouse.leftButton', buttons[0])
                KN_rep.addData('Next_mouse.midButton', buttons[1])
                KN_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    KN_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisKN_rep as finished
                if hasattr(thisKN_rep, 'status'):
                    thisKN_rep.status = FINISHED
                # if awaiting a pause, pause now
                if KN_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    KN_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'KN_rep'
            KN_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if KN_rep.trialList in ([], [None], None):
                params = []
            else:
                params = KN_rep.trialList[0].keys()
            # save data for this loop
            KN_rep.saveAsExcel(filename + '.xlsx', sheetName='KN_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisD as finished
            if hasattr(thisD, 'status'):
                thisD.status = FINISHED
            # if awaiting a pause, pause now
            if D.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                D.status = STARTED
            thisExp.nextEntry()
            
        # completed kanizsa_repeat repeats of 'D'
        D.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if D.trialList in ([], [None], None):
            params = []
        else:
            params = D.trialList[0].keys()
        # save data for this loop
        D.saveAsExcel(filename + '.xlsx', sheetName='D',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        E = data.TrialHandler2(
            name='E',
            nReps=MS_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(E)  # add the loop to the experiment
        thisE = E.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisE.rgb)
        if thisE != None:
            for paramName in thisE:
                globals()[paramName] = thisE[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisE in E:
            E.status = STARTED
            if hasattr(thisE, 'status'):
                thisE.status = STARTED
            currentLoop = E
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisE.rgb)
            if thisE != None:
                for paramName in thisE:
                    globals()[paramName] = thisE[paramName]
            
            # --- Prepare to start Routine "MS_INTRO_2" ---
            # create an object to store info about Routine MS_INTRO_2
            MS_INTRO_2 = data.Routine(
                name='MS_INTRO_2',
                components=[MS_2_intro_text, sound_19, START_MS_2],
            )
            MS_INTRO_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_19.setSound('Stimuli/Instructions/SNK_INTRO.ogg', hamming=True)
            sound_19.setVolume(1.0, log=False)
            sound_19.seek(0)
            # reset START_MS_2 to account for continued clicks & clear times on/off
            START_MS_2.reset()
            # store start times for MS_INTRO_2
            MS_INTRO_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            MS_INTRO_2.tStart = globalClock.getTime(format='float')
            MS_INTRO_2.status = STARTED
            thisExp.addData('MS_INTRO_2.started', MS_INTRO_2.tStart)
            MS_INTRO_2.maxDuration = None
            # keep track of which components have finished
            MS_INTRO_2Components = MS_INTRO_2.components
            for thisComponent in MS_INTRO_2.components:
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
            
            # --- Run Routine "MS_INTRO_2" ---
            thisExp.currentRoutine = MS_INTRO_2
            MS_INTRO_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisE, 'status') and thisE.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *MS_2_intro_text* updates
                
                # if MS_2_intro_text is starting this frame...
                if MS_2_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    MS_2_intro_text.frameNStart = frameN  # exact frame index
                    MS_2_intro_text.tStart = t  # local t and not account for scr refresh
                    MS_2_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(MS_2_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    MS_2_intro_text.status = STARTED
                    MS_2_intro_text.setAutoDraw(True)
                
                # if MS_2_intro_text is active this frame...
                if MS_2_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_19* updates
                
                # if sound_19 is starting this frame...
                if sound_19.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_19.frameNStart = frameN  # exact frame index
                    sound_19.tStart = t  # local t and not account for scr refresh
                    sound_19.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_19.status = STARTED
                    sound_19.play(when=win)  # sync with win flip
                
                # if sound_19 is stopping this frame...
                if sound_19.status == STARTED:
                    if bool(False) or sound_19.isFinished:
                        # keep track of stop time/frame for later
                        sound_19.tStop = t  # not accounting for scr refresh
                        sound_19.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_19.frameNStop = frameN  # exact frame index
                        # update status
                        sound_19.status = FINISHED
                        sound_19.stop()
                # *START_MS_2* updates
                
                # if START_MS_2 is starting this frame...
                if START_MS_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_MS_2.frameNStart = frameN  # exact frame index
                    START_MS_2.tStart = t  # local t and not account for scr refresh
                    START_MS_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_MS_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_MS_2.status = STARTED
                    win.callOnFlip(START_MS_2.buttonClock.reset)
                    START_MS_2.setAutoDraw(True)
                
                # if START_MS_2 is active this frame...
                if START_MS_2.status == STARTED:
                    # update params
                    pass
                    # check whether START_MS_2 has been pressed
                    if START_MS_2.isClicked:
                        if not START_MS_2.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_MS_2.timesOn.append(globalClock.getTime())
                            START_MS_2.timesOff.append(globalClock.getTime())
                        elif len(START_MS_2.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_MS_2.timesOff[-1] = globalClock.getTime()
                        if not START_MS_2.wasClicked:
                            # end routine when START_MS_2 is clicked
                            continueRoutine = False
                        if not START_MS_2.wasClicked:
                            # run callback code when START_MS_2 is clicked
                            pass
                # take note of whether START_MS_2 was clicked, so that next frame we know if clicks are new
                START_MS_2.wasClicked = START_MS_2.isClicked and START_MS_2.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=MS_INTRO_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    MS_INTRO_2.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if MS_INTRO_2.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in MS_INTRO_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "MS_INTRO_2" ---
            for thisComponent in MS_INTRO_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for MS_INTRO_2
            MS_INTRO_2.tStop = globalClock.getTime(format='float')
            MS_INTRO_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('MS_INTRO_2.stopped', MS_INTRO_2.tStop)
            sound_19.pause()  # ensure sound has stopped at end of Routine
            E.addData('START_MS_2.numClicks', START_MS_2.numClicks)
            if START_MS_2.numClicks:
               E.addData('START_MS_2.timesOn', START_MS_2.timesOn)
               E.addData('START_MS_2.timesOff', START_MS_2.timesOff)
            else:
               E.addData('START_MS_2.timesOn', "")
               E.addData('START_MS_2.timesOff', "")
            # the Routine "MS_INTRO_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            trials_2 = data.TrialHandler2(
                name='trials_2',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Stimuli/Moving snake trials v2/MS_condition_file.xlsx'), 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(trials_2)  # add the loop to the experiment
            thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
            if thisTrial_2 != None:
                for paramName in thisTrial_2:
                    globals()[paramName] = thisTrial_2[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisTrial_2 in trials_2:
                trials_2.status = STARTED
                if hasattr(thisTrial_2, 'status'):
                    thisTrial_2.status = STARTED
                currentLoop = trials_2
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
                if thisTrial_2 != None:
                    for paramName in thisTrial_2:
                        globals()[paramName] = thisTrial_2[paramName]
                
                # --- Prepare to start Routine "TEST_MS_2" ---
                # create an object to store info about Routine TEST_MS_2
                TEST_MS_2 = data.Routine(
                    name='TEST_MS_2',
                    components=[image_MS_2, mouse_MS_2, sound_18, key_resp_10, button_YES, button_NO],
                )
                TEST_MS_2.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_MS_2.setImage(path)
                # setup some python lists for storing info about the mouse_MS_2
                mouse_MS_2.x = []
                mouse_MS_2.y = []
                mouse_MS_2.leftButton = []
                mouse_MS_2.midButton = []
                mouse_MS_2.rightButton = []
                mouse_MS_2.time = []
                mouse_MS_2.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_18.setSound('Stimuli/Instructions/SNK_SHORT.ogg', secs=10, hamming=True)
                sound_18.setVolume(1.0, log=False)
                sound_18.seek(0)
                # create starting attributes for key_resp_10
                key_resp_10.keys = []
                key_resp_10.rt = []
                _key_resp_10_allKeys = []
                # reset button_YES to account for continued clicks & clear times on/off
                button_YES.reset()
                # reset button_NO to account for continued clicks & clear times on/off
                button_NO.reset()
                # store start times for TEST_MS_2
                TEST_MS_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_MS_2.tStart = globalClock.getTime(format='float')
                TEST_MS_2.status = STARTED
                thisExp.addData('TEST_MS_2.started', TEST_MS_2.tStart)
                TEST_MS_2.maxDuration = None
                # keep track of which components have finished
                TEST_MS_2Components = TEST_MS_2.components
                for thisComponent in TEST_MS_2.components:
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
                
                # --- Run Routine "TEST_MS_2" ---
                thisExp.currentRoutine = TEST_MS_2
                TEST_MS_2.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisTrial_2, 'status') and thisTrial_2.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_MS_2* updates
                    
                    # if image_MS_2 is starting this frame...
                    if image_MS_2.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_MS_2.frameNStart = frameN  # exact frame index
                        image_MS_2.tStart = t  # local t and not account for scr refresh
                        image_MS_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_MS_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_MS_2.status = STARTED
                        image_MS_2.setAutoDraw(True)
                    
                    # if image_MS_2 is active this frame...
                    if image_MS_2.status == STARTED:
                        # update params
                        pass
                    
                    # if image_MS_2 is stopping this frame...
                    if image_MS_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_MS_2.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_MS_2.tStop = t  # not accounting for scr refresh
                            image_MS_2.tStopRefresh = tThisFlipGlobal  # on global time
                            image_MS_2.frameNStop = frameN  # exact frame index
                            # update status
                            image_MS_2.status = FINISHED
                            image_MS_2.setAutoDraw(False)
                    # *mouse_MS_2* updates
                    
                    # if mouse_MS_2 is starting this frame...
                    if mouse_MS_2.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_MS_2.frameNStart = frameN  # exact frame index
                        mouse_MS_2.tStart = t  # local t and not account for scr refresh
                        mouse_MS_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_MS_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_MS_2.started', t)
                        # update status
                        mouse_MS_2.status = STARTED
                        mouse_MS_2.mouseClock.reset()
                        prevButtonState = mouse_MS_2.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_MS_2 is stopping this frame...
                    if mouse_MS_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_MS_2.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_MS_2.tStop = t  # not accounting for scr refresh
                            mouse_MS_2.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_MS_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_MS_2.stopped', t)
                            # update status
                            mouse_MS_2.status = FINISHED
                    if mouse_MS_2.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_MS_2.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([button_YES, button_NO], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_MS_2):
                                        gotValidClick = True
                                        mouse_MS_2.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_MS_2.clicked_name.append(None)
                                x, y = mouse_MS_2.getPos()
                                mouse_MS_2.x.append(float(x))
                                mouse_MS_2.y.append(float(y))
                                buttons = mouse_MS_2.getPressed()
                                mouse_MS_2.leftButton.append(buttons[0])
                                mouse_MS_2.midButton.append(buttons[1])
                                mouse_MS_2.rightButton.append(buttons[2])
                                mouse_MS_2.time.append(mouse_MS_2.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_18* updates
                    
                    # if sound_18 is starting this frame...
                    if sound_18.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_18.frameNStart = frameN  # exact frame index
                        sound_18.tStart = t  # local t and not account for scr refresh
                        sound_18.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_18.status = STARTED
                        sound_18.play(when=win)  # sync with win flip
                    
                    # if sound_18 is stopping this frame...
                    if sound_18.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_18.tStartRefresh + 10-frameTolerance or sound_18.isFinished:
                            # keep track of stop time/frame for later
                            sound_18.tStop = t  # not accounting for scr refresh
                            sound_18.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_18.frameNStop = frameN  # exact frame index
                            # update status
                            sound_18.status = FINISHED
                            sound_18.stop()
                    
                    # *key_resp_10* updates
                    
                    # if key_resp_10 is starting this frame...
                    if key_resp_10.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_10.frameNStart = frameN  # exact frame index
                        key_resp_10.tStart = t  # local t and not account for scr refresh
                        key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_10.status = STARTED
                        # keyboard checking is just starting
                        key_resp_10.clock.reset()  # now t=0
                    
                    # if key_resp_10 is stopping this frame...
                    if key_resp_10.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_10.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_10.tStop = t  # not accounting for scr refresh
                            key_resp_10.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_10.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_10.status = FINISHED
                            key_resp_10.status = FINISHED
                    if key_resp_10.status == STARTED:
                        theseKeys = key_resp_10.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_10_allKeys.extend(theseKeys)
                        if len(_key_resp_10_allKeys):
                            key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                            key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                            key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    # *button_YES* updates
                    
                    # if button_YES is starting this frame...
                    if button_YES.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        button_YES.frameNStart = frameN  # exact frame index
                        button_YES.tStart = t  # local t and not account for scr refresh
                        button_YES.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(button_YES, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'button_YES.started')
                        # update status
                        button_YES.status = STARTED
                        win.callOnFlip(button_YES.buttonClock.reset)
                        button_YES.setAutoDraw(True)
                    
                    # if button_YES is active this frame...
                    if button_YES.status == STARTED:
                        # update params
                        pass
                        # check whether button_YES has been pressed
                        if button_YES.isClicked:
                            if not button_YES.wasClicked:
                                # if this is a new click, store time of first click and clicked until
                                button_YES.timesOn.append(button_YES.buttonClock.getTime())
                                button_YES.timesOff.append(button_YES.buttonClock.getTime())
                            elif len(button_YES.timesOff):
                                # if click is continuing from last frame, update time of clicked until
                                button_YES.timesOff[-1] = button_YES.buttonClock.getTime()
                            if not button_YES.wasClicked:
                                # end routine when button_YES is clicked
                                continueRoutine = False
                            if not button_YES.wasClicked:
                                # run callback code when button_YES is clicked
                                pass
                    # take note of whether button_YES was clicked, so that next frame we know if clicks are new
                    button_YES.wasClicked = button_YES.isClicked and button_YES.status == STARTED
                    
                    # if button_YES is stopping this frame...
                    if button_YES.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > button_YES.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            button_YES.tStop = t  # not accounting for scr refresh
                            button_YES.tStopRefresh = tThisFlipGlobal  # on global time
                            button_YES.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'button_YES.stopped')
                            # update status
                            button_YES.status = FINISHED
                            button_YES.setAutoDraw(False)
                    # *button_NO* updates
                    
                    # if button_NO is starting this frame...
                    if button_NO.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        button_NO.frameNStart = frameN  # exact frame index
                        button_NO.tStart = t  # local t and not account for scr refresh
                        button_NO.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(button_NO, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'button_NO.started')
                        # update status
                        button_NO.status = STARTED
                        win.callOnFlip(button_NO.buttonClock.reset)
                        button_NO.setAutoDraw(True)
                    
                    # if button_NO is active this frame...
                    if button_NO.status == STARTED:
                        # update params
                        pass
                        # check whether button_NO has been pressed
                        if button_NO.isClicked:
                            if not button_NO.wasClicked:
                                # if this is a new click, store time of first click and clicked until
                                button_NO.timesOn.append(button_NO.buttonClock.getTime())
                                button_NO.timesOff.append(button_NO.buttonClock.getTime())
                            elif len(button_NO.timesOff):
                                # if click is continuing from last frame, update time of clicked until
                                button_NO.timesOff[-1] = button_NO.buttonClock.getTime()
                            if not button_NO.wasClicked:
                                # end routine when button_NO is clicked
                                continueRoutine = False
                            if not button_NO.wasClicked:
                                # run callback code when button_NO is clicked
                                pass
                    # take note of whether button_NO was clicked, so that next frame we know if clicks are new
                    button_NO.wasClicked = button_NO.isClicked and button_NO.status == STARTED
                    
                    # if button_NO is stopping this frame...
                    if button_NO.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > button_NO.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            button_NO.tStop = t  # not accounting for scr refresh
                            button_NO.tStopRefresh = tThisFlipGlobal  # on global time
                            button_NO.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'button_NO.stopped')
                            # update status
                            button_NO.status = FINISHED
                            button_NO.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_MS_2,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_MS_2.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_MS_2.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_MS_2.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_MS_2" ---
                for thisComponent in TEST_MS_2.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_MS_2
                TEST_MS_2.tStop = globalClock.getTime(format='float')
                TEST_MS_2.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_MS_2.stopped', TEST_MS_2.tStop)
                # store data for trials_2 (TrialHandler)
                trials_2.addData('mouse_MS_2.x', mouse_MS_2.x)
                trials_2.addData('mouse_MS_2.y', mouse_MS_2.y)
                trials_2.addData('mouse_MS_2.leftButton', mouse_MS_2.leftButton)
                trials_2.addData('mouse_MS_2.midButton', mouse_MS_2.midButton)
                trials_2.addData('mouse_MS_2.rightButton', mouse_MS_2.rightButton)
                trials_2.addData('mouse_MS_2.time', mouse_MS_2.time)
                trials_2.addData('mouse_MS_2.clicked_name', mouse_MS_2.clicked_name)
                sound_18.pause()  # ensure sound has stopped at end of Routine
                trials_2.addData('button_YES.numClicks', button_YES.numClicks)
                if button_YES.numClicks:
                   trials_2.addData('button_YES.timesOn', button_YES.timesOn)
                   trials_2.addData('button_YES.timesOff', button_YES.timesOff)
                else:
                   trials_2.addData('button_YES.timesOn', "")
                   trials_2.addData('button_YES.timesOff', "")
                trials_2.addData('button_NO.numClicks', button_NO.numClicks)
                if button_NO.numClicks:
                   trials_2.addData('button_NO.timesOn', button_NO.timesOn)
                   trials_2.addData('button_NO.timesOff', button_NO.timesOff)
                else:
                   trials_2.addData('button_NO.timesOn', "")
                   trials_2.addData('button_NO.timesOff', "")
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_MS_2.maxDurationReached:
                    routineTimer.addTime(-TEST_MS_2.maxDuration)
                elif TEST_MS_2.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisTrial_2, 'status') and thisTrial_2.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for trials_2 (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                trials_2.addData('Next_mouse.x', x)
                trials_2.addData('Next_mouse.y', y)
                trials_2.addData('Next_mouse.leftButton', buttons[0])
                trials_2.addData('Next_mouse.midButton', buttons[1])
                trials_2.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    trials_2.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisTrial_2 as finished
                if hasattr(thisTrial_2, 'status'):
                    thisTrial_2.status = FINISHED
                # if awaiting a pause, pause now
                if trials_2.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    trials_2.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'trials_2'
            trials_2.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if trials_2.trialList in ([], [None], None):
                params = []
            else:
                params = trials_2.trialList[0].keys()
            # save data for this loop
            trials_2.saveAsExcel(filename + '.xlsx', sheetName='trials_2',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisE as finished
            if hasattr(thisE, 'status'):
                thisE.status = FINISHED
            # if awaiting a pause, pause now
            if E.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                E.status = STARTED
            thisExp.nextEntry()
            
        # completed MS_repeat repeats of 'E'
        E.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if E.trialList in ([], [None], None):
            params = []
        else:
            params = E.trialList[0].keys()
        # save data for this loop
        E.saveAsExcel(filename + '.xlsx', sheetName='E',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        F = data.TrialHandler2(
            name='F',
            nReps=delbouf_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(F)  # add the loop to the experiment
        thisF = F.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisF.rgb)
        if thisF != None:
            for paramName in thisF:
                globals()[paramName] = thisF[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisF in F:
            F.status = STARTED
            if hasattr(thisF, 'status'):
                thisF.status = STARTED
            currentLoop = F
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisF.rgb)
            if thisF != None:
                for paramName in thisF:
                    globals()[paramName] = thisF[paramName]
            
            # --- Prepare to start Routine "DELBOUF_INTRO" ---
            # create an object to store info about Routine DELBOUF_INTRO
            DELBOUF_INTRO = data.Routine(
                name='DELBOUF_INTRO',
                components=[DELBOUF_intro_text, sound_12, START_DELBOUF],
            )
            DELBOUF_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_12.setSound('Stimuli/Instructions/DELB_INTRO.ogg', hamming=True)
            sound_12.setVolume(1.0, log=False)
            sound_12.seek(0)
            # reset START_DELBOUF to account for continued clicks & clear times on/off
            START_DELBOUF.reset()
            # store start times for DELBOUF_INTRO
            DELBOUF_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            DELBOUF_INTRO.tStart = globalClock.getTime(format='float')
            DELBOUF_INTRO.status = STARTED
            thisExp.addData('DELBOUF_INTRO.started', DELBOUF_INTRO.tStart)
            DELBOUF_INTRO.maxDuration = None
            # keep track of which components have finished
            DELBOUF_INTROComponents = DELBOUF_INTRO.components
            for thisComponent in DELBOUF_INTRO.components:
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
            
            # --- Run Routine "DELBOUF_INTRO" ---
            thisExp.currentRoutine = DELBOUF_INTRO
            DELBOUF_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisF, 'status') and thisF.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *DELBOUF_intro_text* updates
                
                # if DELBOUF_intro_text is starting this frame...
                if DELBOUF_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    DELBOUF_intro_text.frameNStart = frameN  # exact frame index
                    DELBOUF_intro_text.tStart = t  # local t and not account for scr refresh
                    DELBOUF_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(DELBOUF_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    DELBOUF_intro_text.status = STARTED
                    DELBOUF_intro_text.setAutoDraw(True)
                
                # if DELBOUF_intro_text is active this frame...
                if DELBOUF_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_12* updates
                
                # if sound_12 is starting this frame...
                if sound_12.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_12.frameNStart = frameN  # exact frame index
                    sound_12.tStart = t  # local t and not account for scr refresh
                    sound_12.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_12.status = STARTED
                    sound_12.play(when=win)  # sync with win flip
                
                # if sound_12 is stopping this frame...
                if sound_12.status == STARTED:
                    if bool(False) or sound_12.isFinished:
                        # keep track of stop time/frame for later
                        sound_12.tStop = t  # not accounting for scr refresh
                        sound_12.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_12.frameNStop = frameN  # exact frame index
                        # update status
                        sound_12.status = FINISHED
                        sound_12.stop()
                # *START_DELBOUF* updates
                
                # if START_DELBOUF is starting this frame...
                if START_DELBOUF.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_DELBOUF.frameNStart = frameN  # exact frame index
                    START_DELBOUF.tStart = t  # local t and not account for scr refresh
                    START_DELBOUF.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_DELBOUF, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_DELBOUF.status = STARTED
                    win.callOnFlip(START_DELBOUF.buttonClock.reset)
                    START_DELBOUF.setAutoDraw(True)
                
                # if START_DELBOUF is active this frame...
                if START_DELBOUF.status == STARTED:
                    # update params
                    pass
                    # check whether START_DELBOUF has been pressed
                    if START_DELBOUF.isClicked:
                        if not START_DELBOUF.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_DELBOUF.timesOn.append(globalClock.getTime())
                            START_DELBOUF.timesOff.append(globalClock.getTime())
                        elif len(START_DELBOUF.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_DELBOUF.timesOff[-1] = globalClock.getTime()
                        if not START_DELBOUF.wasClicked:
                            # end routine when START_DELBOUF is clicked
                            continueRoutine = False
                        if not START_DELBOUF.wasClicked:
                            # run callback code when START_DELBOUF is clicked
                            pass
                # take note of whether START_DELBOUF was clicked, so that next frame we know if clicks are new
                START_DELBOUF.wasClicked = START_DELBOUF.isClicked and START_DELBOUF.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=DELBOUF_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    DELBOUF_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if DELBOUF_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in DELBOUF_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "DELBOUF_INTRO" ---
            for thisComponent in DELBOUF_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for DELBOUF_INTRO
            DELBOUF_INTRO.tStop = globalClock.getTime(format='float')
            DELBOUF_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('DELBOUF_INTRO.stopped', DELBOUF_INTRO.tStop)
            sound_12.pause()  # ensure sound has stopped at end of Routine
            F.addData('START_DELBOUF.numClicks', START_DELBOUF.numClicks)
            if START_DELBOUF.numClicks:
               F.addData('START_DELBOUF.timesOn', START_DELBOUF.timesOn)
               F.addData('START_DELBOUF.timesOff', START_DELBOUF.timesOff)
            else:
               F.addData('START_DELBOUF.timesOn', "")
               F.addData('START_DELBOUF.timesOff', "")
            # the Routine "DELBOUF_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            DELBOUF_rep = data.TrialHandler2(
                name='DELBOUF_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Delboeuf trials/Delboeuf_condition_file.xlsx', 
                selection=set_rows_delboeuf
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(DELBOUF_rep)  # add the loop to the experiment
            thisDELBOUF_rep = DELBOUF_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisDELBOUF_rep.rgb)
            if thisDELBOUF_rep != None:
                for paramName in thisDELBOUF_rep:
                    globals()[paramName] = thisDELBOUF_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisDELBOUF_rep in DELBOUF_rep:
                DELBOUF_rep.status = STARTED
                if hasattr(thisDELBOUF_rep, 'status'):
                    thisDELBOUF_rep.status = STARTED
                currentLoop = DELBOUF_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisDELBOUF_rep.rgb)
                if thisDELBOUF_rep != None:
                    for paramName in thisDELBOUF_rep:
                        globals()[paramName] = thisDELBOUF_rep[paramName]
                
                # --- Prepare to start Routine "TEST_DELBOUF" ---
                # create an object to store info about Routine TEST_DELBOUF
                TEST_DELBOUF = data.Routine(
                    name='TEST_DELBOUF',
                    components=[image_DELBOUF, LEFT_DELBOUF, RIGHT_DELBOUF, mouse_DELBOUF, sound_13, key_resp_7],
                )
                TEST_DELBOUF.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_DELBOUF.setImage(path)
                # setup some python lists for storing info about the mouse_DELBOUF
                mouse_DELBOUF.x = []
                mouse_DELBOUF.y = []
                mouse_DELBOUF.leftButton = []
                mouse_DELBOUF.midButton = []
                mouse_DELBOUF.rightButton = []
                mouse_DELBOUF.time = []
                mouse_DELBOUF.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_13.setSound('Stimuli/Instructions/DELB_SHORT.ogg', secs=10, hamming=True)
                sound_13.setVolume(1.0, log=False)
                sound_13.seek(0)
                # create starting attributes for key_resp_7
                key_resp_7.keys = []
                key_resp_7.rt = []
                _key_resp_7_allKeys = []
                # store start times for TEST_DELBOUF
                TEST_DELBOUF.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_DELBOUF.tStart = globalClock.getTime(format='float')
                TEST_DELBOUF.status = STARTED
                thisExp.addData('TEST_DELBOUF.started', TEST_DELBOUF.tStart)
                TEST_DELBOUF.maxDuration = None
                # keep track of which components have finished
                TEST_DELBOUFComponents = TEST_DELBOUF.components
                for thisComponent in TEST_DELBOUF.components:
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
                
                # --- Run Routine "TEST_DELBOUF" ---
                thisExp.currentRoutine = TEST_DELBOUF
                TEST_DELBOUF.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisDELBOUF_rep, 'status') and thisDELBOUF_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_DELBOUF* updates
                    
                    # if image_DELBOUF is starting this frame...
                    if image_DELBOUF.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_DELBOUF.frameNStart = frameN  # exact frame index
                        image_DELBOUF.tStart = t  # local t and not account for scr refresh
                        image_DELBOUF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_DELBOUF, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_DELBOUF.status = STARTED
                        image_DELBOUF.setAutoDraw(True)
                    
                    # if image_DELBOUF is active this frame...
                    if image_DELBOUF.status == STARTED:
                        # update params
                        pass
                    
                    # if image_DELBOUF is stopping this frame...
                    if image_DELBOUF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_DELBOUF.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_DELBOUF.tStop = t  # not accounting for scr refresh
                            image_DELBOUF.tStopRefresh = tThisFlipGlobal  # on global time
                            image_DELBOUF.frameNStop = frameN  # exact frame index
                            # update status
                            image_DELBOUF.status = FINISHED
                            image_DELBOUF.setAutoDraw(False)
                    
                    # *LEFT_DELBOUF* updates
                    
                    # if LEFT_DELBOUF is starting this frame...
                    if LEFT_DELBOUF.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_DELBOUF.frameNStart = frameN  # exact frame index
                        LEFT_DELBOUF.tStart = t  # local t and not account for scr refresh
                        LEFT_DELBOUF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_DELBOUF, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_DELBOUF.status = STARTED
                        LEFT_DELBOUF.setAutoDraw(True)
                    
                    # if LEFT_DELBOUF is active this frame...
                    if LEFT_DELBOUF.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_DELBOUF is stopping this frame...
                    if LEFT_DELBOUF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_DELBOUF.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_DELBOUF.tStop = t  # not accounting for scr refresh
                            LEFT_DELBOUF.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_DELBOUF.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_DELBOUF.status = FINISHED
                            LEFT_DELBOUF.setAutoDraw(False)
                    
                    # *RIGHT_DELBOUF* updates
                    
                    # if RIGHT_DELBOUF is starting this frame...
                    if RIGHT_DELBOUF.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_DELBOUF.frameNStart = frameN  # exact frame index
                        RIGHT_DELBOUF.tStart = t  # local t and not account for scr refresh
                        RIGHT_DELBOUF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_DELBOUF, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_DELBOUF.status = STARTED
                        RIGHT_DELBOUF.setAutoDraw(True)
                    
                    # if RIGHT_DELBOUF is active this frame...
                    if RIGHT_DELBOUF.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_DELBOUF is stopping this frame...
                    if RIGHT_DELBOUF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_DELBOUF.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_DELBOUF.tStop = t  # not accounting for scr refresh
                            RIGHT_DELBOUF.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_DELBOUF.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_DELBOUF.status = FINISHED
                            RIGHT_DELBOUF.setAutoDraw(False)
                    # *mouse_DELBOUF* updates
                    
                    # if mouse_DELBOUF is starting this frame...
                    if mouse_DELBOUF.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_DELBOUF.frameNStart = frameN  # exact frame index
                        mouse_DELBOUF.tStart = t  # local t and not account for scr refresh
                        mouse_DELBOUF.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_DELBOUF, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_DELBOUF.started', t)
                        # update status
                        mouse_DELBOUF.status = STARTED
                        mouse_DELBOUF.mouseClock.reset()
                        prevButtonState = mouse_DELBOUF.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_DELBOUF is stopping this frame...
                    if mouse_DELBOUF.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_DELBOUF.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_DELBOUF.tStop = t  # not accounting for scr refresh
                            mouse_DELBOUF.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_DELBOUF.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_DELBOUF.stopped', t)
                            # update status
                            mouse_DELBOUF.status = FINISHED
                    if mouse_DELBOUF.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_DELBOUF.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_DELBOUF, RIGHT_DELBOUF], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_DELBOUF):
                                        gotValidClick = True
                                        mouse_DELBOUF.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_DELBOUF.clicked_name.append(None)
                                x, y = mouse_DELBOUF.getPos()
                                mouse_DELBOUF.x.append(float(x))
                                mouse_DELBOUF.y.append(float(y))
                                buttons = mouse_DELBOUF.getPressed()
                                mouse_DELBOUF.leftButton.append(buttons[0])
                                mouse_DELBOUF.midButton.append(buttons[1])
                                mouse_DELBOUF.rightButton.append(buttons[2])
                                mouse_DELBOUF.time.append(mouse_DELBOUF.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_13* updates
                    
                    # if sound_13 is starting this frame...
                    if sound_13.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_13.frameNStart = frameN  # exact frame index
                        sound_13.tStart = t  # local t and not account for scr refresh
                        sound_13.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_13.status = STARTED
                        sound_13.play(when=win)  # sync with win flip
                    
                    # if sound_13 is stopping this frame...
                    if sound_13.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_13.tStartRefresh + 10-frameTolerance or sound_13.isFinished:
                            # keep track of stop time/frame for later
                            sound_13.tStop = t  # not accounting for scr refresh
                            sound_13.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_13.frameNStop = frameN  # exact frame index
                            # update status
                            sound_13.status = FINISHED
                            sound_13.stop()
                    
                    # *key_resp_7* updates
                    
                    # if key_resp_7 is starting this frame...
                    if key_resp_7.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_7.frameNStart = frameN  # exact frame index
                        key_resp_7.tStart = t  # local t and not account for scr refresh
                        key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_7.status = STARTED
                        # keyboard checking is just starting
                        key_resp_7.clock.reset()  # now t=0
                    
                    # if key_resp_7 is stopping this frame...
                    if key_resp_7.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_7.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_7.tStop = t  # not accounting for scr refresh
                            key_resp_7.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_7.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_7.status = FINISHED
                            key_resp_7.status = FINISHED
                    if key_resp_7.status == STARTED:
                        theseKeys = key_resp_7.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_7_allKeys.extend(theseKeys)
                        if len(_key_resp_7_allKeys):
                            key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                            key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                            key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_DELBOUF,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_DELBOUF.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_DELBOUF.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_DELBOUF.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_DELBOUF" ---
                for thisComponent in TEST_DELBOUF.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_DELBOUF
                TEST_DELBOUF.tStop = globalClock.getTime(format='float')
                TEST_DELBOUF.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_DELBOUF.stopped', TEST_DELBOUF.tStop)
                # store data for DELBOUF_rep (TrialHandler)
                DELBOUF_rep.addData('mouse_DELBOUF.x', mouse_DELBOUF.x)
                DELBOUF_rep.addData('mouse_DELBOUF.y', mouse_DELBOUF.y)
                DELBOUF_rep.addData('mouse_DELBOUF.leftButton', mouse_DELBOUF.leftButton)
                DELBOUF_rep.addData('mouse_DELBOUF.midButton', mouse_DELBOUF.midButton)
                DELBOUF_rep.addData('mouse_DELBOUF.rightButton', mouse_DELBOUF.rightButton)
                DELBOUF_rep.addData('mouse_DELBOUF.time', mouse_DELBOUF.time)
                DELBOUF_rep.addData('mouse_DELBOUF.clicked_name', mouse_DELBOUF.clicked_name)
                sound_13.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_DELBOUF.maxDurationReached:
                    routineTimer.addTime(-TEST_DELBOUF.maxDuration)
                elif TEST_DELBOUF.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisDELBOUF_rep, 'status') and thisDELBOUF_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for DELBOUF_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                DELBOUF_rep.addData('Next_mouse.x', x)
                DELBOUF_rep.addData('Next_mouse.y', y)
                DELBOUF_rep.addData('Next_mouse.leftButton', buttons[0])
                DELBOUF_rep.addData('Next_mouse.midButton', buttons[1])
                DELBOUF_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    DELBOUF_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisDELBOUF_rep as finished
                if hasattr(thisDELBOUF_rep, 'status'):
                    thisDELBOUF_rep.status = FINISHED
                # if awaiting a pause, pause now
                if DELBOUF_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    DELBOUF_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'DELBOUF_rep'
            DELBOUF_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if DELBOUF_rep.trialList in ([], [None], None):
                params = []
            else:
                params = DELBOUF_rep.trialList[0].keys()
            # save data for this loop
            DELBOUF_rep.saveAsExcel(filename + '.xlsx', sheetName='DELBOUF_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisF as finished
            if hasattr(thisF, 'status'):
                thisF.status = FINISHED
            # if awaiting a pause, pause now
            if F.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                F.status = STARTED
            thisExp.nextEntry()
            
        # completed delbouf_repeat repeats of 'F'
        F.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if F.trialList in ([], [None], None):
            params = []
        else:
            params = F.trialList[0].keys()
        # save data for this loop
        F.saveAsExcel(filename + '.xlsx', sheetName='F',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        G = data.TrialHandler2(
            name='G',
            nReps=ponzo_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(G)  # add the loop to the experiment
        thisG = G.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisG.rgb)
        if thisG != None:
            for paramName in thisG:
                globals()[paramName] = thisG[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisG in G:
            G.status = STARTED
            if hasattr(thisG, 'status'):
                thisG.status = STARTED
            currentLoop = G
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisG.rgb)
            if thisG != None:
                for paramName in thisG:
                    globals()[paramName] = thisG[paramName]
            
            # --- Prepare to start Routine "PONZO_INTRO" ---
            # create an object to store info about Routine PONZO_INTRO
            PONZO_INTRO = data.Routine(
                name='PONZO_INTRO',
                components=[PONZO_intro_text, sound_15, START_PONZO],
            )
            PONZO_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_15.setSound('Stimuli/Instructions/PNZ_INTRO.ogg', hamming=True)
            sound_15.setVolume(1.0, log=False)
            sound_15.seek(0)
            # reset START_PONZO to account for continued clicks & clear times on/off
            START_PONZO.reset()
            # store start times for PONZO_INTRO
            PONZO_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            PONZO_INTRO.tStart = globalClock.getTime(format='float')
            PONZO_INTRO.status = STARTED
            thisExp.addData('PONZO_INTRO.started', PONZO_INTRO.tStart)
            PONZO_INTRO.maxDuration = None
            # keep track of which components have finished
            PONZO_INTROComponents = PONZO_INTRO.components
            for thisComponent in PONZO_INTRO.components:
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
            
            # --- Run Routine "PONZO_INTRO" ---
            thisExp.currentRoutine = PONZO_INTRO
            PONZO_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisG, 'status') and thisG.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *PONZO_intro_text* updates
                
                # if PONZO_intro_text is starting this frame...
                if PONZO_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    PONZO_intro_text.frameNStart = frameN  # exact frame index
                    PONZO_intro_text.tStart = t  # local t and not account for scr refresh
                    PONZO_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(PONZO_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    PONZO_intro_text.status = STARTED
                    PONZO_intro_text.setAutoDraw(True)
                
                # if PONZO_intro_text is active this frame...
                if PONZO_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_15* updates
                
                # if sound_15 is starting this frame...
                if sound_15.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_15.frameNStart = frameN  # exact frame index
                    sound_15.tStart = t  # local t and not account for scr refresh
                    sound_15.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_15.status = STARTED
                    sound_15.play(when=win)  # sync with win flip
                
                # if sound_15 is stopping this frame...
                if sound_15.status == STARTED:
                    if bool(False) or sound_15.isFinished:
                        # keep track of stop time/frame for later
                        sound_15.tStop = t  # not accounting for scr refresh
                        sound_15.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_15.frameNStop = frameN  # exact frame index
                        # update status
                        sound_15.status = FINISHED
                        sound_15.stop()
                # *START_PONZO* updates
                
                # if START_PONZO is starting this frame...
                if START_PONZO.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_PONZO.frameNStart = frameN  # exact frame index
                    START_PONZO.tStart = t  # local t and not account for scr refresh
                    START_PONZO.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_PONZO, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_PONZO.status = STARTED
                    win.callOnFlip(START_PONZO.buttonClock.reset)
                    START_PONZO.setAutoDraw(True)
                
                # if START_PONZO is active this frame...
                if START_PONZO.status == STARTED:
                    # update params
                    pass
                    # check whether START_PONZO has been pressed
                    if START_PONZO.isClicked:
                        if not START_PONZO.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_PONZO.timesOn.append(globalClock.getTime())
                            START_PONZO.timesOff.append(globalClock.getTime())
                        elif len(START_PONZO.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_PONZO.timesOff[-1] = globalClock.getTime()
                        if not START_PONZO.wasClicked:
                            # end routine when START_PONZO is clicked
                            continueRoutine = False
                        if not START_PONZO.wasClicked:
                            # run callback code when START_PONZO is clicked
                            pass
                # take note of whether START_PONZO was clicked, so that next frame we know if clicks are new
                START_PONZO.wasClicked = START_PONZO.isClicked and START_PONZO.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=PONZO_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    PONZO_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if PONZO_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in PONZO_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "PONZO_INTRO" ---
            for thisComponent in PONZO_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for PONZO_INTRO
            PONZO_INTRO.tStop = globalClock.getTime(format='float')
            PONZO_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('PONZO_INTRO.stopped', PONZO_INTRO.tStop)
            sound_15.pause()  # ensure sound has stopped at end of Routine
            G.addData('START_PONZO.numClicks', START_PONZO.numClicks)
            if START_PONZO.numClicks:
               G.addData('START_PONZO.timesOn', START_PONZO.timesOn)
               G.addData('START_PONZO.timesOff', START_PONZO.timesOff)
            else:
               G.addData('START_PONZO.timesOn', "")
               G.addData('START_PONZO.timesOff', "")
            # the Routine "PONZO_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            PONZO_rep = data.TrialHandler2(
                name='PONZO_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions('Stimuli/Ponzo trials/Ponzo_condition_file.xlsx'), 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(PONZO_rep)  # add the loop to the experiment
            thisPONZO_rep = PONZO_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisPONZO_rep.rgb)
            if thisPONZO_rep != None:
                for paramName in thisPONZO_rep:
                    globals()[paramName] = thisPONZO_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisPONZO_rep in PONZO_rep:
                PONZO_rep.status = STARTED
                if hasattr(thisPONZO_rep, 'status'):
                    thisPONZO_rep.status = STARTED
                currentLoop = PONZO_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisPONZO_rep.rgb)
                if thisPONZO_rep != None:
                    for paramName in thisPONZO_rep:
                        globals()[paramName] = thisPONZO_rep[paramName]
                
                # --- Prepare to start Routine "TEST_PONZO" ---
                # create an object to store info about Routine TEST_PONZO
                TEST_PONZO = data.Routine(
                    name='TEST_PONZO',
                    components=[image_PONZO, UP_PONZO, DOWN_PONZO, mouse_PONZO, sound_14, key_resp_8],
                )
                TEST_PONZO.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_PONZO.setImage(path)
                # setup some python lists for storing info about the mouse_PONZO
                mouse_PONZO.x = []
                mouse_PONZO.y = []
                mouse_PONZO.leftButton = []
                mouse_PONZO.midButton = []
                mouse_PONZO.rightButton = []
                mouse_PONZO.time = []
                mouse_PONZO.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_14.setSound('Stimuli/Instructions/PNZ_SHORT.ogg', secs=10, hamming=True)
                sound_14.setVolume(1.0, log=False)
                sound_14.seek(0)
                # create starting attributes for key_resp_8
                key_resp_8.keys = []
                key_resp_8.rt = []
                _key_resp_8_allKeys = []
                # store start times for TEST_PONZO
                TEST_PONZO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_PONZO.tStart = globalClock.getTime(format='float')
                TEST_PONZO.status = STARTED
                thisExp.addData('TEST_PONZO.started', TEST_PONZO.tStart)
                TEST_PONZO.maxDuration = None
                # keep track of which components have finished
                TEST_PONZOComponents = TEST_PONZO.components
                for thisComponent in TEST_PONZO.components:
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
                
                # --- Run Routine "TEST_PONZO" ---
                thisExp.currentRoutine = TEST_PONZO
                TEST_PONZO.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisPONZO_rep, 'status') and thisPONZO_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_PONZO* updates
                    
                    # if image_PONZO is starting this frame...
                    if image_PONZO.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_PONZO.frameNStart = frameN  # exact frame index
                        image_PONZO.tStart = t  # local t and not account for scr refresh
                        image_PONZO.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_PONZO, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_PONZO.status = STARTED
                        image_PONZO.setAutoDraw(True)
                    
                    # if image_PONZO is active this frame...
                    if image_PONZO.status == STARTED:
                        # update params
                        pass
                    
                    # if image_PONZO is stopping this frame...
                    if image_PONZO.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_PONZO.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_PONZO.tStop = t  # not accounting for scr refresh
                            image_PONZO.tStopRefresh = tThisFlipGlobal  # on global time
                            image_PONZO.frameNStop = frameN  # exact frame index
                            # update status
                            image_PONZO.status = FINISHED
                            image_PONZO.setAutoDraw(False)
                    
                    # *UP_PONZO* updates
                    
                    # if UP_PONZO is starting this frame...
                    if UP_PONZO.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        UP_PONZO.frameNStart = frameN  # exact frame index
                        UP_PONZO.tStart = t  # local t and not account for scr refresh
                        UP_PONZO.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(UP_PONZO, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        UP_PONZO.status = STARTED
                        UP_PONZO.setAutoDraw(True)
                    
                    # if UP_PONZO is active this frame...
                    if UP_PONZO.status == STARTED:
                        # update params
                        pass
                    
                    # if UP_PONZO is stopping this frame...
                    if UP_PONZO.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > UP_PONZO.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            UP_PONZO.tStop = t  # not accounting for scr refresh
                            UP_PONZO.tStopRefresh = tThisFlipGlobal  # on global time
                            UP_PONZO.frameNStop = frameN  # exact frame index
                            # update status
                            UP_PONZO.status = FINISHED
                            UP_PONZO.setAutoDraw(False)
                    
                    # *DOWN_PONZO* updates
                    
                    # if DOWN_PONZO is starting this frame...
                    if DOWN_PONZO.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        DOWN_PONZO.frameNStart = frameN  # exact frame index
                        DOWN_PONZO.tStart = t  # local t and not account for scr refresh
                        DOWN_PONZO.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(DOWN_PONZO, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        DOWN_PONZO.status = STARTED
                        DOWN_PONZO.setAutoDraw(True)
                    
                    # if DOWN_PONZO is active this frame...
                    if DOWN_PONZO.status == STARTED:
                        # update params
                        pass
                    
                    # if DOWN_PONZO is stopping this frame...
                    if DOWN_PONZO.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > DOWN_PONZO.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            DOWN_PONZO.tStop = t  # not accounting for scr refresh
                            DOWN_PONZO.tStopRefresh = tThisFlipGlobal  # on global time
                            DOWN_PONZO.frameNStop = frameN  # exact frame index
                            # update status
                            DOWN_PONZO.status = FINISHED
                            DOWN_PONZO.setAutoDraw(False)
                    # *mouse_PONZO* updates
                    
                    # if mouse_PONZO is starting this frame...
                    if mouse_PONZO.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_PONZO.frameNStart = frameN  # exact frame index
                        mouse_PONZO.tStart = t  # local t and not account for scr refresh
                        mouse_PONZO.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_PONZO, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_PONZO.started', t)
                        # update status
                        mouse_PONZO.status = STARTED
                        mouse_PONZO.mouseClock.reset()
                        prevButtonState = mouse_PONZO.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_PONZO is stopping this frame...
                    if mouse_PONZO.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_PONZO.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_PONZO.tStop = t  # not accounting for scr refresh
                            mouse_PONZO.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_PONZO.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_PONZO.stopped', t)
                            # update status
                            mouse_PONZO.status = FINISHED
                    if mouse_PONZO.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_PONZO.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([UP_PONZO, DOWN_PONZO], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_PONZO):
                                        gotValidClick = True
                                        mouse_PONZO.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_PONZO.clicked_name.append(None)
                                x, y = mouse_PONZO.getPos()
                                mouse_PONZO.x.append(float(x))
                                mouse_PONZO.y.append(float(y))
                                buttons = mouse_PONZO.getPressed()
                                mouse_PONZO.leftButton.append(buttons[0])
                                mouse_PONZO.midButton.append(buttons[1])
                                mouse_PONZO.rightButton.append(buttons[2])
                                mouse_PONZO.time.append(mouse_PONZO.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_14* updates
                    
                    # if sound_14 is starting this frame...
                    if sound_14.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_14.frameNStart = frameN  # exact frame index
                        sound_14.tStart = t  # local t and not account for scr refresh
                        sound_14.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_14.status = STARTED
                        sound_14.play(when=win)  # sync with win flip
                    
                    # if sound_14 is stopping this frame...
                    if sound_14.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_14.tStartRefresh + 10-frameTolerance or sound_14.isFinished:
                            # keep track of stop time/frame for later
                            sound_14.tStop = t  # not accounting for scr refresh
                            sound_14.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_14.frameNStop = frameN  # exact frame index
                            # update status
                            sound_14.status = FINISHED
                            sound_14.stop()
                    
                    # *key_resp_8* updates
                    
                    # if key_resp_8 is starting this frame...
                    if key_resp_8.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_8.frameNStart = frameN  # exact frame index
                        key_resp_8.tStart = t  # local t and not account for scr refresh
                        key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_8.status = STARTED
                        # keyboard checking is just starting
                        key_resp_8.clock.reset()  # now t=0
                    
                    # if key_resp_8 is stopping this frame...
                    if key_resp_8.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_8.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_8.tStop = t  # not accounting for scr refresh
                            key_resp_8.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_8.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_8.status = FINISHED
                            key_resp_8.status = FINISHED
                    if key_resp_8.status == STARTED:
                        theseKeys = key_resp_8.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_8_allKeys.extend(theseKeys)
                        if len(_key_resp_8_allKeys):
                            key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                            key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                            key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_PONZO,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_PONZO.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_PONZO.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_PONZO.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_PONZO" ---
                for thisComponent in TEST_PONZO.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_PONZO
                TEST_PONZO.tStop = globalClock.getTime(format='float')
                TEST_PONZO.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_PONZO.stopped', TEST_PONZO.tStop)
                # store data for PONZO_rep (TrialHandler)
                PONZO_rep.addData('mouse_PONZO.x', mouse_PONZO.x)
                PONZO_rep.addData('mouse_PONZO.y', mouse_PONZO.y)
                PONZO_rep.addData('mouse_PONZO.leftButton', mouse_PONZO.leftButton)
                PONZO_rep.addData('mouse_PONZO.midButton', mouse_PONZO.midButton)
                PONZO_rep.addData('mouse_PONZO.rightButton', mouse_PONZO.rightButton)
                PONZO_rep.addData('mouse_PONZO.time', mouse_PONZO.time)
                PONZO_rep.addData('mouse_PONZO.clicked_name', mouse_PONZO.clicked_name)
                sound_14.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_PONZO.maxDurationReached:
                    routineTimer.addTime(-TEST_PONZO.maxDuration)
                elif TEST_PONZO.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisPONZO_rep, 'status') and thisPONZO_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for PONZO_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                PONZO_rep.addData('Next_mouse.x', x)
                PONZO_rep.addData('Next_mouse.y', y)
                PONZO_rep.addData('Next_mouse.leftButton', buttons[0])
                PONZO_rep.addData('Next_mouse.midButton', buttons[1])
                PONZO_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    PONZO_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisPONZO_rep as finished
                if hasattr(thisPONZO_rep, 'status'):
                    thisPONZO_rep.status = FINISHED
                # if awaiting a pause, pause now
                if PONZO_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    PONZO_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'PONZO_rep'
            PONZO_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if PONZO_rep.trialList in ([], [None], None):
                params = []
            else:
                params = PONZO_rep.trialList[0].keys()
            # save data for this loop
            PONZO_rep.saveAsExcel(filename + '.xlsx', sheetName='PONZO_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisG as finished
            if hasattr(thisG, 'status'):
                thisG.status = FINISHED
            # if awaiting a pause, pause now
            if G.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                G.status = STARTED
            thisExp.nextEntry()
            
        # completed ponzo_repeat repeats of 'G'
        G.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if G.trialList in ([], [None], None):
            params = []
        else:
            params = G.trialList[0].keys()
        # save data for this loop
        G.saveAsExcel(filename + '.xlsx', sheetName='G',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        H = data.TrialHandler2(
            name='H',
            nReps=solitaire_repeat, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(H)  # add the loop to the experiment
        thisH = H.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisH.rgb)
        if thisH != None:
            for paramName in thisH:
                globals()[paramName] = thisH[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisH in H:
            H.status = STARTED
            if hasattr(thisH, 'status'):
                thisH.status = STARTED
            currentLoop = H
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisH.rgb)
            if thisH != None:
                for paramName in thisH:
                    globals()[paramName] = thisH[paramName]
            
            # --- Prepare to start Routine "SOLITAIRE_INTRO" ---
            # create an object to store info about Routine SOLITAIRE_INTRO
            SOLITAIRE_INTRO = data.Routine(
                name='SOLITAIRE_INTRO',
                components=[SOLITAIRE_intro_text, sound_17, START_SOLITAIRE],
            )
            SOLITAIRE_INTRO.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sound_17.setSound('Stimuli/Instructions/SOL_INTRO.ogg', hamming=True)
            sound_17.setVolume(1.0, log=False)
            sound_17.seek(0)
            # reset START_SOLITAIRE to account for continued clicks & clear times on/off
            START_SOLITAIRE.reset()
            # store start times for SOLITAIRE_INTRO
            SOLITAIRE_INTRO.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            SOLITAIRE_INTRO.tStart = globalClock.getTime(format='float')
            SOLITAIRE_INTRO.status = STARTED
            thisExp.addData('SOLITAIRE_INTRO.started', SOLITAIRE_INTRO.tStart)
            SOLITAIRE_INTRO.maxDuration = None
            # keep track of which components have finished
            SOLITAIRE_INTROComponents = SOLITAIRE_INTRO.components
            for thisComponent in SOLITAIRE_INTRO.components:
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
            
            # --- Run Routine "SOLITAIRE_INTRO" ---
            thisExp.currentRoutine = SOLITAIRE_INTRO
            SOLITAIRE_INTRO.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisH, 'status') and thisH.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *SOLITAIRE_intro_text* updates
                
                # if SOLITAIRE_intro_text is starting this frame...
                if SOLITAIRE_intro_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    SOLITAIRE_intro_text.frameNStart = frameN  # exact frame index
                    SOLITAIRE_intro_text.tStart = t  # local t and not account for scr refresh
                    SOLITAIRE_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(SOLITAIRE_intro_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    SOLITAIRE_intro_text.status = STARTED
                    SOLITAIRE_intro_text.setAutoDraw(True)
                
                # if SOLITAIRE_intro_text is active this frame...
                if SOLITAIRE_intro_text.status == STARTED:
                    # update params
                    pass
                
                # *sound_17* updates
                
                # if sound_17 is starting this frame...
                if sound_17.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    sound_17.frameNStart = frameN  # exact frame index
                    sound_17.tStart = t  # local t and not account for scr refresh
                    sound_17.tStartRefresh = tThisFlipGlobal  # on global time
                    # update status
                    sound_17.status = STARTED
                    sound_17.play(when=win)  # sync with win flip
                
                # if sound_17 is stopping this frame...
                if sound_17.status == STARTED:
                    if bool(False) or sound_17.isFinished:
                        # keep track of stop time/frame for later
                        sound_17.tStop = t  # not accounting for scr refresh
                        sound_17.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_17.frameNStop = frameN  # exact frame index
                        # update status
                        sound_17.status = FINISHED
                        sound_17.stop()
                # *START_SOLITAIRE* updates
                
                # if START_SOLITAIRE is starting this frame...
                if START_SOLITAIRE.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    START_SOLITAIRE.frameNStart = frameN  # exact frame index
                    START_SOLITAIRE.tStart = t  # local t and not account for scr refresh
                    START_SOLITAIRE.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(START_SOLITAIRE, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    START_SOLITAIRE.status = STARTED
                    win.callOnFlip(START_SOLITAIRE.buttonClock.reset)
                    START_SOLITAIRE.setAutoDraw(True)
                
                # if START_SOLITAIRE is active this frame...
                if START_SOLITAIRE.status == STARTED:
                    # update params
                    pass
                    # check whether START_SOLITAIRE has been pressed
                    if START_SOLITAIRE.isClicked:
                        if not START_SOLITAIRE.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            START_SOLITAIRE.timesOn.append(globalClock.getTime())
                            START_SOLITAIRE.timesOff.append(globalClock.getTime())
                        elif len(START_SOLITAIRE.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            START_SOLITAIRE.timesOff[-1] = globalClock.getTime()
                        if not START_SOLITAIRE.wasClicked:
                            # end routine when START_SOLITAIRE is clicked
                            continueRoutine = False
                        if not START_SOLITAIRE.wasClicked:
                            # run callback code when START_SOLITAIRE is clicked
                            pass
                # take note of whether START_SOLITAIRE was clicked, so that next frame we know if clicks are new
                START_SOLITAIRE.wasClicked = START_SOLITAIRE.isClicked and START_SOLITAIRE.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=SOLITAIRE_INTRO,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    SOLITAIRE_INTRO.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if SOLITAIRE_INTRO.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in SOLITAIRE_INTRO.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "SOLITAIRE_INTRO" ---
            for thisComponent in SOLITAIRE_INTRO.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for SOLITAIRE_INTRO
            SOLITAIRE_INTRO.tStop = globalClock.getTime(format='float')
            SOLITAIRE_INTRO.tStopRefresh = tThisFlipGlobal
            thisExp.addData('SOLITAIRE_INTRO.stopped', SOLITAIRE_INTRO.tStop)
            sound_17.pause()  # ensure sound has stopped at end of Routine
            H.addData('START_SOLITAIRE.numClicks', START_SOLITAIRE.numClicks)
            if START_SOLITAIRE.numClicks:
               H.addData('START_SOLITAIRE.timesOn', START_SOLITAIRE.timesOn)
               H.addData('START_SOLITAIRE.timesOff', START_SOLITAIRE.timesOff)
            else:
               H.addData('START_SOLITAIRE.timesOn', "")
               H.addData('START_SOLITAIRE.timesOff', "")
            # the Routine "SOLITAIRE_INTRO" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            SOLITAIRE_rep = data.TrialHandler2(
                name='SOLITAIRE_rep',
                nReps=1.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'Stimuli/Solitaire trials/Solitaire_condition_file.xlsx', 
                selection=set_rows_solitaire
            )
            , 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(SOLITAIRE_rep)  # add the loop to the experiment
            thisSOLITAIRE_rep = SOLITAIRE_rep.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSOLITAIRE_rep.rgb)
            if thisSOLITAIRE_rep != None:
                for paramName in thisSOLITAIRE_rep:
                    globals()[paramName] = thisSOLITAIRE_rep[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSOLITAIRE_rep in SOLITAIRE_rep:
                SOLITAIRE_rep.status = STARTED
                if hasattr(thisSOLITAIRE_rep, 'status'):
                    thisSOLITAIRE_rep.status = STARTED
                currentLoop = SOLITAIRE_rep
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSOLITAIRE_rep.rgb)
                if thisSOLITAIRE_rep != None:
                    for paramName in thisSOLITAIRE_rep:
                        globals()[paramName] = thisSOLITAIRE_rep[paramName]
                
                # --- Prepare to start Routine "TEST_SOLITAIRE" ---
                # create an object to store info about Routine TEST_SOLITAIRE
                TEST_SOLITAIRE = data.Routine(
                    name='TEST_SOLITAIRE',
                    components=[image_SOLITAIRE, LEFT_SOLITAIRE, RIGHT_SOLITAIRE, mouse_SOLITAIRE, sound_16, key_resp_9],
                )
                TEST_SOLITAIRE.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                image_SOLITAIRE.setImage(path)
                # setup some python lists for storing info about the mouse_SOLITAIRE
                mouse_SOLITAIRE.x = []
                mouse_SOLITAIRE.y = []
                mouse_SOLITAIRE.leftButton = []
                mouse_SOLITAIRE.midButton = []
                mouse_SOLITAIRE.rightButton = []
                mouse_SOLITAIRE.time = []
                mouse_SOLITAIRE.clicked_name = []
                gotValidClick = False  # until a click is received
                sound_16.setSound('Stimuli/Instructions/SOL_SHORT.ogg', secs=10, hamming=True)
                sound_16.setVolume(1.0, log=False)
                sound_16.seek(0)
                # create starting attributes for key_resp_9
                key_resp_9.keys = []
                key_resp_9.rt = []
                _key_resp_9_allKeys = []
                # store start times for TEST_SOLITAIRE
                TEST_SOLITAIRE.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                TEST_SOLITAIRE.tStart = globalClock.getTime(format='float')
                TEST_SOLITAIRE.status = STARTED
                thisExp.addData('TEST_SOLITAIRE.started', TEST_SOLITAIRE.tStart)
                TEST_SOLITAIRE.maxDuration = None
                # keep track of which components have finished
                TEST_SOLITAIREComponents = TEST_SOLITAIRE.components
                for thisComponent in TEST_SOLITAIRE.components:
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
                
                # --- Run Routine "TEST_SOLITAIRE" ---
                thisExp.currentRoutine = TEST_SOLITAIRE
                TEST_SOLITAIRE.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 15.1:
                    # if trial has changed, end Routine now
                    if hasattr(thisSOLITAIRE_rep, 'status') and thisSOLITAIRE_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *image_SOLITAIRE* updates
                    
                    # if image_SOLITAIRE is starting this frame...
                    if image_SOLITAIRE.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        image_SOLITAIRE.frameNStart = frameN  # exact frame index
                        image_SOLITAIRE.tStart = t  # local t and not account for scr refresh
                        image_SOLITAIRE.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image_SOLITAIRE, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        image_SOLITAIRE.status = STARTED
                        image_SOLITAIRE.setAutoDraw(True)
                    
                    # if image_SOLITAIRE is active this frame...
                    if image_SOLITAIRE.status == STARTED:
                        # update params
                        pass
                    
                    # if image_SOLITAIRE is stopping this frame...
                    if image_SOLITAIRE.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image_SOLITAIRE.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            image_SOLITAIRE.tStop = t  # not accounting for scr refresh
                            image_SOLITAIRE.tStopRefresh = tThisFlipGlobal  # on global time
                            image_SOLITAIRE.frameNStop = frameN  # exact frame index
                            # update status
                            image_SOLITAIRE.status = FINISHED
                            image_SOLITAIRE.setAutoDraw(False)
                    
                    # *LEFT_SOLITAIRE* updates
                    
                    # if LEFT_SOLITAIRE is starting this frame...
                    if LEFT_SOLITAIRE.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        LEFT_SOLITAIRE.frameNStart = frameN  # exact frame index
                        LEFT_SOLITAIRE.tStart = t  # local t and not account for scr refresh
                        LEFT_SOLITAIRE.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(LEFT_SOLITAIRE, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        LEFT_SOLITAIRE.status = STARTED
                        LEFT_SOLITAIRE.setAutoDraw(True)
                    
                    # if LEFT_SOLITAIRE is active this frame...
                    if LEFT_SOLITAIRE.status == STARTED:
                        # update params
                        pass
                    
                    # if LEFT_SOLITAIRE is stopping this frame...
                    if LEFT_SOLITAIRE.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > LEFT_SOLITAIRE.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            LEFT_SOLITAIRE.tStop = t  # not accounting for scr refresh
                            LEFT_SOLITAIRE.tStopRefresh = tThisFlipGlobal  # on global time
                            LEFT_SOLITAIRE.frameNStop = frameN  # exact frame index
                            # update status
                            LEFT_SOLITAIRE.status = FINISHED
                            LEFT_SOLITAIRE.setAutoDraw(False)
                    
                    # *RIGHT_SOLITAIRE* updates
                    
                    # if RIGHT_SOLITAIRE is starting this frame...
                    if RIGHT_SOLITAIRE.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        RIGHT_SOLITAIRE.frameNStart = frameN  # exact frame index
                        RIGHT_SOLITAIRE.tStart = t  # local t and not account for scr refresh
                        RIGHT_SOLITAIRE.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(RIGHT_SOLITAIRE, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        RIGHT_SOLITAIRE.status = STARTED
                        RIGHT_SOLITAIRE.setAutoDraw(True)
                    
                    # if RIGHT_SOLITAIRE is active this frame...
                    if RIGHT_SOLITAIRE.status == STARTED:
                        # update params
                        pass
                    
                    # if RIGHT_SOLITAIRE is stopping this frame...
                    if RIGHT_SOLITAIRE.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > RIGHT_SOLITAIRE.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            RIGHT_SOLITAIRE.tStop = t  # not accounting for scr refresh
                            RIGHT_SOLITAIRE.tStopRefresh = tThisFlipGlobal  # on global time
                            RIGHT_SOLITAIRE.frameNStop = frameN  # exact frame index
                            # update status
                            RIGHT_SOLITAIRE.status = FINISHED
                            RIGHT_SOLITAIRE.setAutoDraw(False)
                    # *mouse_SOLITAIRE* updates
                    
                    # if mouse_SOLITAIRE is starting this frame...
                    if mouse_SOLITAIRE.status == NOT_STARTED and t >= 0.1-frameTolerance:
                        # keep track of start time/frame for later
                        mouse_SOLITAIRE.frameNStart = frameN  # exact frame index
                        mouse_SOLITAIRE.tStart = t  # local t and not account for scr refresh
                        mouse_SOLITAIRE.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouse_SOLITAIRE, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('mouse_SOLITAIRE.started', t)
                        # update status
                        mouse_SOLITAIRE.status = STARTED
                        mouse_SOLITAIRE.mouseClock.reset()
                        prevButtonState = mouse_SOLITAIRE.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mouse_SOLITAIRE is stopping this frame...
                    if mouse_SOLITAIRE.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mouse_SOLITAIRE.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            mouse_SOLITAIRE.tStop = t  # not accounting for scr refresh
                            mouse_SOLITAIRE.tStopRefresh = tThisFlipGlobal  # on global time
                            mouse_SOLITAIRE.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.addData('mouse_SOLITAIRE.stopped', t)
                            # update status
                            mouse_SOLITAIRE.status = FINISHED
                    if mouse_SOLITAIRE.status == STARTED:  # only update if started and not finished!
                        buttons = mouse_SOLITAIRE.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames([LEFT_SOLITAIRE, RIGHT_SOLITAIRE], namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(mouse_SOLITAIRE):
                                        gotValidClick = True
                                        mouse_SOLITAIRE.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    mouse_SOLITAIRE.clicked_name.append(None)
                                x, y = mouse_SOLITAIRE.getPos()
                                mouse_SOLITAIRE.x.append(float(x))
                                mouse_SOLITAIRE.y.append(float(y))
                                buttons = mouse_SOLITAIRE.getPressed()
                                mouse_SOLITAIRE.leftButton.append(buttons[0])
                                mouse_SOLITAIRE.midButton.append(buttons[1])
                                mouse_SOLITAIRE.rightButton.append(buttons[2])
                                mouse_SOLITAIRE.time.append(mouse_SOLITAIRE.mouseClock.getTime())
                                if gotValidClick:
                                    continueRoutine = False  # end routine on response
                    
                    # *sound_16* updates
                    
                    # if sound_16 is starting this frame...
                    if sound_16.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                        # keep track of start time/frame for later
                        sound_16.frameNStart = frameN  # exact frame index
                        sound_16.tStart = t  # local t and not account for scr refresh
                        sound_16.tStartRefresh = tThisFlipGlobal  # on global time
                        # update status
                        sound_16.status = STARTED
                        sound_16.play(when=win)  # sync with win flip
                    
                    # if sound_16 is stopping this frame...
                    if sound_16.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > sound_16.tStartRefresh + 10-frameTolerance or sound_16.isFinished:
                            # keep track of stop time/frame for later
                            sound_16.tStop = t  # not accounting for scr refresh
                            sound_16.tStopRefresh = tThisFlipGlobal  # on global time
                            sound_16.frameNStop = frameN  # exact frame index
                            # update status
                            sound_16.status = FINISHED
                            sound_16.stop()
                    
                    # *key_resp_9* updates
                    
                    # if key_resp_9 is starting this frame...
                    if key_resp_9.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_9.frameNStart = frameN  # exact frame index
                        key_resp_9.tStart = t  # local t and not account for scr refresh
                        key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_9.status = STARTED
                        # keyboard checking is just starting
                        key_resp_9.clock.reset()  # now t=0
                    
                    # if key_resp_9 is stopping this frame...
                    if key_resp_9.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_resp_9.tStartRefresh + 15-frameTolerance:
                            # keep track of stop time/frame for later
                            key_resp_9.tStop = t  # not accounting for scr refresh
                            key_resp_9.tStopRefresh = tThisFlipGlobal  # on global time
                            key_resp_9.frameNStop = frameN  # exact frame index
                            # update status
                            key_resp_9.status = FINISHED
                            key_resp_9.status = FINISHED
                    if key_resp_9.status == STARTED:
                        theseKeys = key_resp_9.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_9_allKeys.extend(theseKeys)
                        if len(_key_resp_9_allKeys):
                            key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                            key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                            key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=TEST_SOLITAIRE,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        TEST_SOLITAIRE.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if TEST_SOLITAIRE.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in TEST_SOLITAIRE.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "TEST_SOLITAIRE" ---
                for thisComponent in TEST_SOLITAIRE.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for TEST_SOLITAIRE
                TEST_SOLITAIRE.tStop = globalClock.getTime(format='float')
                TEST_SOLITAIRE.tStopRefresh = tThisFlipGlobal
                thisExp.addData('TEST_SOLITAIRE.stopped', TEST_SOLITAIRE.tStop)
                # store data for SOLITAIRE_rep (TrialHandler)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.x', mouse_SOLITAIRE.x)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.y', mouse_SOLITAIRE.y)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.leftButton', mouse_SOLITAIRE.leftButton)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.midButton', mouse_SOLITAIRE.midButton)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.rightButton', mouse_SOLITAIRE.rightButton)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.time', mouse_SOLITAIRE.time)
                SOLITAIRE_rep.addData('mouse_SOLITAIRE.clicked_name', mouse_SOLITAIRE.clicked_name)
                sound_16.pause()  # ensure sound has stopped at end of Routine
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if TEST_SOLITAIRE.maxDurationReached:
                    routineTimer.addTime(-TEST_SOLITAIRE.maxDuration)
                elif TEST_SOLITAIRE.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-15.100000)
                
                # --- Prepare to start Routine "pressspace" ---
                # create an object to store info about Routine pressspace
                pressspace = data.Routine(
                    name='pressspace',
                    components=[Next_text, Next_polygon, Next_mouse, key_resp_2],
                )
                pressspace.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                Next_text.reset()
                # setup some python lists for storing info about the Next_mouse
                Next_mouse.clicked_name = []
                gotValidClick = False  # until a click is received
                # create starting attributes for key_resp_2
                key_resp_2.keys = []
                key_resp_2.rt = []
                _key_resp_2_allKeys = []
                # store start times for pressspace
                pressspace.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                pressspace.tStart = globalClock.getTime(format='float')
                pressspace.status = STARTED
                thisExp.addData('pressspace.started', pressspace.tStart)
                pressspace.maxDuration = None
                # keep track of which components have finished
                pressspaceComponents = pressspace.components
                for thisComponent in pressspace.components:
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
                
                # --- Run Routine "pressspace" ---
                thisExp.currentRoutine = pressspace
                pressspace.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisSOLITAIRE_rep, 'status') and thisSOLITAIRE_rep.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *Next_text* updates
                    
                    # if Next_text is starting this frame...
                    if Next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_text.frameNStart = frameN  # exact frame index
                        Next_text.tStart = t  # local t and not account for scr refresh
                        Next_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_text.status = STARTED
                        Next_text.setAutoDraw(True)
                    
                    # if Next_text is active this frame...
                    if Next_text.status == STARTED:
                        # update params
                        pass
                    
                    # *Next_polygon* updates
                    
                    # if Next_polygon is starting this frame...
                    if Next_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_polygon.frameNStart = frameN  # exact frame index
                        Next_polygon.tStart = t  # local t and not account for scr refresh
                        Next_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_polygon, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        Next_polygon.status = STARTED
                        Next_polygon.setAutoDraw(True)
                    
                    # if Next_polygon is active this frame...
                    if Next_polygon.status == STARTED:
                        # update params
                        pass
                    # *Next_mouse* updates
                    
                    # if Next_mouse is starting this frame...
                    if Next_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Next_mouse.frameNStart = frameN  # exact frame index
                        Next_mouse.tStart = t  # local t and not account for scr refresh
                        Next_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Next_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('Next_mouse.started', t)
                        # update status
                        Next_mouse.status = STARTED
                        prevButtonState = Next_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if Next_mouse.status == STARTED:  # only update if started and not finished!
                        buttons = Next_mouse.getPressed()
                        if buttons != prevButtonState:  # button state changed?
                            prevButtonState = buttons
                            if sum(buttons) > 0:  # state changed to a new click
                                # check if the mouse was inside our 'clickable' objects
                                gotValidClick = False
                                clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                                for obj in clickableList:
                                    # is this object clicked on?
                                    if obj.contains(Next_mouse):
                                        gotValidClick = True
                                        Next_mouse.clicked_name.append(obj.name)
                                if not gotValidClick:
                                    Next_mouse.clicked_name.append(None)
                                if gotValidClick:  
                                    continueRoutine = False  # end routine on response
                    
                    # *key_resp_2* updates
                    
                    # if key_resp_2 is starting this frame...
                    if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_2.frameNStart = frameN  # exact frame index
                        key_resp_2.tStart = t  # local t and not account for scr refresh
                        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        key_resp_2.status = STARTED
                        # keyboard checking is just starting
                        key_resp_2.clock.reset()  # now t=0
                    if key_resp_2.status == STARTED:
                        theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_2_allKeys.extend(theseKeys)
                        if len(_key_resp_2_allKeys):
                            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                            key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=pressspace,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        pressspace.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if pressspace.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in pressspace.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pressspace" ---
                for thisComponent in pressspace.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for pressspace
                pressspace.tStop = globalClock.getTime(format='float')
                pressspace.tStopRefresh = tThisFlipGlobal
                thisExp.addData('pressspace.stopped', pressspace.tStop)
                # store data for SOLITAIRE_rep (TrialHandler)
                x, y = Next_mouse.getPos()
                buttons = Next_mouse.getPressed()
                if sum(buttons):
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(Next_polygon, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(Next_mouse):
                            gotValidClick = True
                            Next_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        Next_mouse.clicked_name.append(None)
                SOLITAIRE_rep.addData('Next_mouse.x', x)
                SOLITAIRE_rep.addData('Next_mouse.y', y)
                SOLITAIRE_rep.addData('Next_mouse.leftButton', buttons[0])
                SOLITAIRE_rep.addData('Next_mouse.midButton', buttons[1])
                SOLITAIRE_rep.addData('Next_mouse.rightButton', buttons[2])
                if len(Next_mouse.clicked_name):
                    SOLITAIRE_rep.addData('Next_mouse.clicked_name', Next_mouse.clicked_name[0])
                # the Routine "pressspace" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisSOLITAIRE_rep as finished
                if hasattr(thisSOLITAIRE_rep, 'status'):
                    thisSOLITAIRE_rep.status = FINISHED
                # if awaiting a pause, pause now
                if SOLITAIRE_rep.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    SOLITAIRE_rep.status = STARTED
                thisExp.nextEntry()
                
            # completed 1.0 repeats of 'SOLITAIRE_rep'
            SOLITAIRE_rep.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # get names of stimulus parameters
            if SOLITAIRE_rep.trialList in ([], [None], None):
                params = []
            else:
                params = SOLITAIRE_rep.trialList[0].keys()
            # save data for this loop
            SOLITAIRE_rep.saveAsExcel(filename + '.xlsx', sheetName='SOLITAIRE_rep',
                stimOut=params,
                dataOut=['n','all_mean','all_std', 'all_raw'])
            # mark thisH as finished
            if hasattr(thisH, 'status'):
                thisH.status = FINISHED
            # if awaiting a pause, pause now
            if H.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                H.status = STARTED
            thisExp.nextEntry()
            
        # completed solitaire_repeat repeats of 'H'
        H.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if H.trialList in ([], [None], None):
            params = []
        else:
            params = H.trialList[0].keys()
        # save data for this loop
        H.saveAsExcel(filename + '.xlsx', sheetName='H',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    
    # --- Prepare to start Routine "bye" ---
    # create an object to store info about Routine bye
    bye = data.Routine(
        name='bye',
        components=[text_2, mouse],
    )
    bye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse
    gotValidClick = False  # until a click is received
    # store start times for bye
    bye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    bye.tStart = globalClock.getTime(format='float')
    bye.status = STARTED
    thisExp.addData('bye.started', bye.tStart)
    bye.maxDuration = None
    # keep track of which components have finished
    byeComponents = bye.components
    for thisComponent in bye.components:
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
    
    # --- Run Routine "bye" ---
    thisExp.currentRoutine = bye
    bye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse.started', t)
            # update status
            mouse.status = STARTED
            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    continueRoutine = False  # end routine on response        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=bye,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            bye.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if bye.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in bye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "bye" ---
    for thisComponent in bye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for bye
    bye.tStop = globalClock.getTime(format='float')
    bye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('bye.stopped', bye.tStop)
    # store data for thisExp (ExperimentHandler)
    x, y = mouse.getPos()
    buttons = mouse.getPressed()
    thisExp.addData('mouse.x', x)
    thisExp.addData('mouse.y', y)
    thisExp.addData('mouse.leftButton', buttons[0])
    thisExp.addData('mouse.midButton', buttons[1])
    thisExp.addData('mouse.rightButton', buttons[2])
    thisExp.nextEntry()
    # the Routine "bye" was not non-slip safe, so reset the non-slip timer
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
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
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
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
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
