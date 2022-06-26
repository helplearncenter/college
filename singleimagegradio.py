import cv2
""" Basic Utils """

""" Data Analysis and Visualization """
import numpy as np
""" Audio Analysis """
import librosa.display

""" Model Evaluation """
import librosa

""" Keras Functional API """

from tensorflow import keras
model = keras.models.load_model(r'mymodel')

AUDIO_DURATION = 2
SAMPLE_RATE = 22050
import tensorflow as tf
import requests
import gradio as gr

def predict_disease(filepath):

  def load_audio_test(fname, offset=0, duration=None):
      """loads the audio"""
      try:
          y, sr = librosa.load(fname, sr=SAMPLE_RATE, offset=offset, duration=duration)
      except IOError:
          y, sr = librosa.load(fname, sr=SAMPLE_RATE, offset=offset, duration=duration)
      except NameError:
          print("{0} does not exist".format(fname))
      return y


  def new_extract_mfccs_test(y):
      return(np.mean(librosa.feature.mfcc(y, sr=SAMPLE_RATE, n_mfcc=13).T,axis=0))
  
  
  newmfcc=new_extract_mfccs_test(load_audio_test(filepath) / np.std(new_extract_mfccs_test(load_audio_test(filepath)), axis=0))
  imgs = []
  xx=np.expand_dims(np.expand_dims(newmfcc,axis=1),axis=2)

  im = cv2.cvtColor(xx, cv2.COLOR_GRAY2RGB)
  imgs=[]
  imgs.append(im)
  imgs=np.array(imgs)
  
  labelsDict = {'artifact': 0, 'extrahls':1, 'murmur':2, 'normal/extrastole':3}
  prediction_array=model.predict(imgs)

  j=0
  for i,v in labelsDict.items():
    labelsDict[i]=float(prediction_array[0][j])
    j+=1
  return labelsDict
# filepath=r"C:\Users\sr\Pictures\heartbeatdataset\set_b\extrastole__130_1306347376079_D.wav"
# labelsDict=predict_disease(filepath)
# labelsDict


gr.Interface(fn=predict_disease, 
             inputs=gr.inputs.Audio(type='filepath'),
             outputs=gr.outputs.Label(num_top_classes=1)).launch(debug=True,share=True)