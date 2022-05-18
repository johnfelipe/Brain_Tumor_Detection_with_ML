import gradio as gr
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

model = pickle.load(open("catboost.pkl", "rb"))

def classify_image(image):
  image = Image.fromarray(image)
  labels = ['Tumor Cerebral Presente', 'Sin Tumor Cerebral']
  image = image.resize((120, 120))
  image = ImageOps.grayscale(image)
  image = np.array(image).reshape((1, -1))
  res = {labels[0]:float(model.predict_proba(image)[0][1]), labels[1]: float(model.predict_proba(image)[0][0])}
  text = "¿Está seguro de que se trata de una imagen de resonancia magnética cerebral? en caso afirmativo, a continuación se obtienen los resultados"
  if model.predict_proba(image)[0][0] < 0.5:
      pred = "La imagen de RMN contiene un tumor cerebral"
      symptoms = "Posibles síntomas: dolores de cabeza nuevos o cada vez más fuertes, visión borrosa, pérdida de equilibrio, confusión y convulsiones (En algunos casos, puede que no haya síntomas también)"
  
  else:
      pred = "La imagen de RMN no tiene un tumor cerebral"
      symptoms = "Posibles Síntomas: Ninguno"
  return text, pred, res, symptoms
  
label1 = gr.outputs.Label(label="")  
label2 = gr.outputs.Label(label="Predicción")
label3 = gr.outputs.Label(label="Puntuación de confianza")
label4 = gr.outputs.Label(label="Síntomas")
image = gr.inputs.Image()

interface = gr.Interface(title = "Clasificador de tumor cerebral",
             fn=classify_image,
             article="Esta herramienta en línea que representa a la IA por una buena causa, esta aplicación web basada en IA en línea es construida por el Mgtr Jhon Felipe Urrego Mejia, usando esta herramienta, uno podría saber si su informe de resonancia magnética cerebral contiene un tumor o no con gran precisión, no importa lo difícil que es ver eso de un ojo humano.",
             inputs=image,
             outputs=[label1, label2, label3, label4],
             examples=[["images/Y3.jpg"],["images/21no.jpg"],["images/Y6.jpg"],["images/N17.jpg"],["images/Y6.jpg"]],
             interpretation=None,
             theme='dark-grass')
 
interface.launch(server_name="0.0.0.0")
