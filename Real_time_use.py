import pickle
import json
from utility import *
import pandas as pd

def load_model():
    filename = "CRF_MODEL.sav"
    model = pickle.load(open(filename, 'rb'))
    return model 


model = load_model() # loading the CRF model

data = "" # pass the string here 

X = data_processing(data) # applying data processing

pred,scores = model_prediction(model,X) # making predictions and getting scores

result = postprocessing(data,pred,scores)  ## result breaker has been added in postprocessing

print(result)