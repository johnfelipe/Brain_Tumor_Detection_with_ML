import gradio as gr
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

model = pickle.load(open("catboost.pkl", "rb"))

def classify_image(image):
  image = Image.fromarray(image)
  labels = ['Brain Tumor Present', 'No Brain Tumor']
  image = image.resize((120, 120))
  image = ImageOps.grayscale(image)
  image = np.array(image).reshape((1, -1))
  res = {labels[0]:float(model.predict_proba(image)[0][1]), labels[1]: float(model.predict_proba(image)[0][0])}
  text = "Are you sure this is a brain MRI picture? if yes, then following are the results obtained"
  if model.predict_proba(image)[0][0] < 0.5:
      pred = "The MRI image contains a Brain Tumor"
      symptoms = "Possible Symptoms : New or increasingly strong headaches, blurred vision, loss of balance, confusion and seizures (In some cases, there may be no symptoms as well)"
  
  else:
      pred = "The MRI image does not have a Brain Tumor"
      symptoms = "Possible Symptoms : None"
  return text, pred, res, symptoms
  
label1 = gr.outputs.Label(label="")  
label2 = gr.outputs.Label(label="Prediction")
label3 = gr.outputs.Label(label="Confidence Score")
label4 = gr.outputs.Label(label="Symptoms")
image = gr.inputs.Image()

interface = gr.Interface(title = "Brain Tumor Classifier",
             fn=classify_image,
             article="This an Online tool representing AI for a good cause, this online AI powered web application is built by Rauhan Ahmed Siddiqui, using this tool, one could know whether his/her brain MRI report contains a tumor or not with great accuracy, no matter how difficult it is to see that from a human eye.",
             inputs=image,
             outputs=[label1, label2, label3, label4],
             examples=[["images/Y3.jpg"],["images/21no.jpg"],["images/Y6.jpg"],["images/N17.jpg"],["images/Y6.jpg"]],
             interpretation=None,
             theme='dark-grass')
 
interface.launch()
