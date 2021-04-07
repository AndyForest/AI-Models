# ConveRT dependencies
# tensorflow 2.0


print(f"Begin color_includes_inference.py")
# Import the required dependencies

import tensorflow_text 

import tensorflow_hub as tfhub

import tensorflow as tf
import wget
from tensorflow import keras
import numpy as np
import math
import scipy.special
import pandas as pd
import random
import pickle
import gzip
import os
import sys
import json
import urllib
import datetime
import time
import re
from IPython.core.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Button, Layout
import random 
import pathlib


if colabENV:
  filePath = r'/content/palettes/'
else:
  filePath = r'G:/My Drive/Colab Data/palettes/'
  

dataFileDir_done = f'{filePath}models/'
includeFileDir = f'{filePath}includes/'

def loadColourData():
    # Note: This function is used for training the model, not for inference
    
    print(f'{datetime.datetime.now()} Begin Loading data')
    global inputData, targetData, colourData_df, namedColours_df
    global colourColumns, colourPaletteColumns, colourDataByWidthsColumns

    # Load data
    try:
        # Check if the data is already loaded
        foo = len(targetData)
    except NameError:
        print(f'{datetime.datetime.now()} Loading data files')
        
        while True == False:
            # Commented out
            # inputData is numpy array with just the ConveRT vectors

            ## File missing TODO ANDY
            # Andy note: You shouldn't need this file, I used it for training the model only. It's 2GB in size, so cannot be loaded with wget from google drive
            inputData = pickle.load( open(f"{dataFileDir_done}palette_inputData_done.pkl", "rb" ) )

            # targetData is a numpy array with just the palettes in it. eg: 
            # [0.6 0.2 0.2 0.2 0.4 0.2 0.6 0.2 0.2 0.4 0.6 0.2 0.2 0.6 0.6 0.2 0.  0.4  0.4 0.2]
            # File Missing Andy TODO
            # Andy Note: Also shouldn't need this file, it was only used for training the model.
            targetData = pickle.load( open(f"{dataFileDir_done}palette_targetData_done.pkl", "rb" ) )

            # colourData_df has all the 5-palette data in it
            saveFilename = 'colourDataGood_5_df.csv'
            colourData_df = pd.read_csv(f'{dataFileDir_done}{saveFilename}', delimiter = ',')
            # Access individual title at row 390: 
            #   colourData_df.iloc[390]["title"]
            # Get a list of multiple items:
            #   colourData_df.iloc[127:129]["title"].tolist()
            #
            # Also works to access one title, but cumbersom: colourData_df[390:391]['title'].tolist()[0]

            # Load the list of colours with names
            namedColours_df = pd.read_csv(f'{includeFileDir}CSS_Basic_Colours.csv', delimiter = ',' , encoding='latin-1')

            print(f'{datetime.datetime.now()} Done loading data files')

    colourColumns = ['id','title','userName','numViews','numVotes','numComments','numHearts','rank','dateCreated','colors','colorWidths']
    colourPaletteColumns = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]
    colourDataByWidthsColumns = ['id','title', 'combinedVotes', 'rank', 'r0', 'g0', 'b0', 'w0', 'r1', 'g1', 'b1', 'w1', 'r2', 'g2', 'b2', 'w2', 'r3', 'g3', 'b3', 'w3', 'r4', 'g4', 'b4', 'w4']

# Inference utility functions

defaultColourChoices = 16

def findTop_n (fullList, n):
  top5_values = []
  top5_pos = []
  for i in range(n):
    top5_values.append(0)
    top5_pos.append(0)

  for i in range(len(fullList)):
    # Find the lowest of the top n values
    low_value = top5_values[0]
    low_pos = 0
    for f in range(1, n):
      if top5_values[f] <= low_value:
        low_value = top5_values[f]
        low_pos = f
    
    # Check if the current value is higher than the lowest
    if fullList[i] >= low_value:
      top5_values[low_pos] = fullList[i]
      top5_pos[low_pos] = i
  
  return top5_pos, top5_values

def floatToRGB(r, g, b):
  r = int(r * 255)
  g = int(g * 255)
  b = int(b * 255)
  return (r, g, b)

def oneHotToRGB(oneHot, colourChoices=defaultColourChoices):
  r, g, b = oneHotToFloat(oneHot, colourChoices)
  r = int(r * 255)
  g = int(g * 255)
  b = int(b * 255)
  return (r, g, b)

def oneHotToFloat(oneHot, colourChoices=defaultColourChoices):
  colours = colourChoices * 1.0
  r = (math.floor(oneHot / colours / colours)) / colours
  g = (math.floor((oneHot - r * colours * colours * colours) / colours)) / colours
  b = (oneHot - r * colours * colours * colours - g * colours * colours) / colours
  return (r, g, b)

def paletteHTML(paletteList, displayHTML=True, convertFloatToRGB=False):
  # paletteList = [[r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b]]
  # RGB values are normally 0-255
  # Call this if they're 0-1:
  # 
  # OR
  # paletteList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
  # RGB values are normally 0.0 to 1.0

  if len(paletteList) == 20:
    newPaletteList = []
    for i in range(0,20,4):
      r, g, b = floatToRGB(paletteList[i], paletteList[i+1], paletteList[i+2])
      thisPalette = [r, g, b]
      newPaletteList.append(thisPalette)
    paletteList = newPaletteList

  HTMLString = ""
  HTMLString = HTMLString + f'<div style="border-width: 1; border-style: solid; width:750px; display: grid; grid-template-columns:'
  for i in range(0,len(paletteList)):
    HTMLString = HTMLString + f' 150px'
  HTMLString = HTMLString + f' ;">'
  
  for palette in paletteList:
    hexColour = ""
    if convertFloatToRGB:
      palette = floatToRGB(palette[0], palette[1], palette[2])
    for colour in palette:
      # Convert to hex for HTML and chop off the leading "0x", and make sure it has a leading zero if one digit
      hexColour = hexColour + (str(hex(colour))[2:]).zfill(2)
    HTMLString = HTMLString + f'<div style="height:80px; width:150px; font-size: 30px; text-align: center; background-color: #{hexColour};">#{hexColour}</div>'
  HTMLString = HTMLString + f'</div>'
  
  if displayHTML:
    display(HTML(HTMLString))
  return HTMLString

def oneHotPaletteHTML(paletteOneHotList, colourChoices=defaultColourChoices):
  paletteList = []
  for paletteOneHot in paletteOneHotList:
    r, g, b = oneHotToRGB(paletteOneHot)
    paletteList.append([r, g, b])
  return paletteHTML(paletteList)

defaultColourChoices = 16

def webSafeColour(r, g, b, colourChoices=defaultColourChoices):
    if isinstance(r, int):
        # convert to float
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    return int(round(r*(colourChoices-1)) * colourChoices*colourChoices + round(g*(colourChoices-1)) * colourChoices + round(b*(colourChoices-1)))

def webSafeColourOneHot(colourList, colourChoices=defaultColourChoices):
    oneHot = np.full((colourChoices*colourChoices*colourChoices), False)
    for i in range(0, math.ceil(len(colourList)/3) *3, 3):
        try:
            thisColour = webSafeColour(colourList[i], colourList[i+1], colourList[i+2])
            oneHot[thisColour] = True
        except:
            print(f"error at i={i} thisColour={thisColour} \ncolourList = \n{colourList}")
            raise
    
    return oneHot

# Convert RGB data to HSV
# From https://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
# R, G, B values are [0, 255]. H value is [0, 360]. S, V values are [0, 1].

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
    
def rgb2hsv(r, g, b, rgbInput=True):
  # Expects 0-255 values for each. If data is already 
  if rgbInput:
    r, g, b = r/255.0, g/255.0, b/255.0
  
  mx = max(r, g, b)
  mn = min(r, g, b)
  df = mx-mn
  if mx == mn:
      h = 0
  elif mx == r:
      h = (60 * ((g-b)/df) + 360) % 360
  elif mx == g:
      h = (60 * ((b-r)/df) + 120) % 360
  elif mx == b:
      h = (60 * ((r-g)/df) + 240) % 360
  if mx == 0:
      s = 0
  else:
      s = df/mx
  v = mx
  return h, s, v



def paletteColour(paletteNumber, colourNum):
  # Return the colours from a palette in colourData_df
  return [colourData_df.iloc[paletteNumber][f'r{colourNum}'], colourData_df.iloc[paletteNumber][f'g{colourNum}'], colourData_df.iloc[paletteNumber][f'b{colourNum}']]

def palette(paletteNumber):
  # Return the whole palette
  thisPalette = []
  for i in range(0,5):
    thisPalette.append(paletteColour(paletteNumber, i))
  return thisPalette

try:
    if conveRT_initialized:
        print(f'{datetime.datetime.now()} ConveRT already initialized')
except NameError:
    conveRT_initialized = False

def initConveRT():
    global sess, module, text_placeholder, context_encoding_tensor, response_encoding_tensor, encoding_dim, conveRT_initialized
    
    # Initialize only once
    if conveRT_initialized == False:
        print(f'{datetime.datetime.now()} Initializing ConveRT')
        print(pathlib.Path().absolute())

        try:
            if sess is not None:
                sess.close()
        except NameError:
            sess = None

        sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

        # Poly-ai no longer hosting this file
        # module = tfhub.Module("http://models.poly-ai.com/convert/v1/model.tar.gz")
        
        # https://drive.google.com/uc?export=download&id=1Nelaj75b05eaHIfiUjeBFS1aEtEWw1AI
        #TODO
        # !wget "https://drive.google.com/uc?export=download&id=1Nelaj75b05eaHIfiUjeBFS1aEtEWw1AI" -O "model.tar.gz"
        # #module = tfhub.Module("https://drive.google.com/uc?export=download&id=1Nelaj75b05eaHIfiUjeBFS1aEtEWw1AI")
        # module = tfhub.Module("model.tar.gz")
        module = tfhub.Module("https://github.com/AndyForest/PolyAI-model/raw/master/models/model.tar.gz")
        #module = tfhub.load('https://drive.google.com/drive/folders/1q1pHTtKkfuuo8lbHUfMmYdvDMtOzt3TW?usp=sharing')
        # url = 'https://drive.google.com/uc?export=download&id=1q1pHTtKkfuuo8lbHUfMmYdvDMtOzt3TW'
        # wget "https://drive.google.com/uc?export=download&id=1Nelaj75b05eaHIfiUjeBFS1aEtEWw1AI" -O "model.tar.gz"
        # model = wget.download(url)

        text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
        context_encoding_tensor = module(text_placeholder, signature="encode_context")
        response_encoding_tensor = module(text_placeholder, signature="encode_response")

        encoding_dim = int(context_encoding_tensor.shape[1])
        print(f"{datetime.datetime.now()} ConveRT encodes contexts & responses to {encoding_dim}-dimensional vectors")

        sess.run(tf.compat.v1.tables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        conveRT_initialized = True
        print(f'{datetime.datetime.now()} Done Initializing ConveRT')



def encode_contexts(texts):
    initConveRT()
    return sess.run(context_encoding_tensor, feed_dict={text_placeholder: texts})

def encode_responses(texts):
    initConveRT()
    return sess.run(response_encoding_tensor, feed_dict={text_placeholder: texts})


def encodeResponseVectorList(responses, batch_size=1):
  
  total_responses = len(responses)
  response_encodings = []
  #for i in range(0, batch_size, batch_size):
  for i in range(0, total_responses, batch_size):
    if i + batch_size >= total_responses:
      # Last batch
      batch = responses[i:]
    else:
      batch = responses[i:i + batch_size]
    
    #if batch_size == 1:
    #  print(f'Processing: {i} {batch}')
    
    newEncodings = encode_responses(batch)
    #test_np = np.array(newEncodings)
    #print(f'Shape: {test_np.shape}')

    response_encodings.extend(newEncodings)
    #print(f'Total encodings: {len(response_encodings)}')
  return response_encodings


def findResponse(text, response_encodings, softmaxChoices=10, returnEncodingCount=1):
  # Find response using ConveRT
  '''
  softmaxChoices is how many responses to randomly choose between
  returnEncodingCount if larger than 1, return a list of this many possible encodings
  '''
  context_encoding = encode_contexts([text])
  scores = np.dot(response_encodings, context_encoding.T)
  top_index = np.argmax(scores)
  top_score = float(scores[top_index])
  
  # Top 1 response
  # print(f"[{top_score:.3f}] {responses[top_index]}")
  #return(top_index, top_score)

  # Find top softmaxChoices responses, and randomly pick one. Use score squared to prefer the higher ranked responses
  # TODO: add in the palette's score to the mix
  # 'combinedVotes'
  # colourData_df[response_index:response_index+1]['combinedVotes'].tolist()[0]

  # Note: scores is an array of single item arrays. Not sure why.
  scoresArray = np.asarray(scores).reshape(-1)
  #scoresArray = np.reshape(scores, -1)
  top_scores = np.argsort(scoresArray)
    
  top_n_scores = top_scores[-softmaxChoices:]
  
  #Softmax has squaring the probabilities built in
  top_n_scores_p = scipy.special.softmax([(scoresArray[item]) for item in top_n_scores])

  #print(f"top_n_scores_p = {top_n_scores_p}")

  if returnEncodingCount == 1:
    # []

    thisResponseChoice = np.random.choice(top_scores[-softmaxChoices:], p=top_n_scores_p)
    responseChoice = [thisResponseChoice]
    top_score = [scoresArray[thisResponseChoice]]
  elif returnEncodingCount >= softmaxChoices:
    # Return the whole list
    responseChoice = top_n_scores
    top_score = top_n_scores_p
  else:
    # return a random sample returnEncodingCount choices of the list
    # TODO: implement this. Right now, it just returns the softmaxChoices
    responseChoice = top_n_scores
    top_score = top_n_scores_p

  return(responseChoice, top_score)