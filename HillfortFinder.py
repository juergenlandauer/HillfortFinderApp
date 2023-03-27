#from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

mypath = Path(".")

import streamlit as st

class Predict:
    def __init__(self, filename):
        st.write(list(mypath.glob('**/*')))
        st.write(str(mypath/filename))
        self.learn_inference = load_learner(mypath/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.prepare_output()
            self.get_prediction()
            self.process_output()
    
    @staticmethod
    def get_image_from_upload():
        # gets images
        uploaded_file = st.file_uploader("Upload Files",type=['tif','jpeg', 'jpg'])
        if uploaded_file is not None:
            # TODO store locally!!!
            return PILImage.create((uploaded_file))
        return None

    def prepare_output(self):
        # shows uploaded image
        # TODO use to show progress bar, instructions, download file, etc.
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):
        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

def process_output(self):
    # shows uploaded image
    # TODO use to show progress bar, instructions, download file, etc.
    st.write(f'process output')            
            
if __name__=='__main__':
    st.set_page_config(page_title="Hillfort Finder App")
    st.header("Hillfort Finder App")


    modelfile = 'm_L100_1.pkl'

    predictor = Predict(modelfile)

