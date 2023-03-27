#from fastai.vision.widgets import *
from fastai.vision.all import *
import gc
import streamlit as st
from pathlib import Path
from io import BytesIO
import rasterio as rio
from rasterio.windows import Window

# range normalizer
NORMALIZER = "SCALE"
from normalizer import normalizeImg

### parameters
# path
VERSION = 100
learner_path = '.'
nnet = "m_L" + str(VERSION) +"_1.pkl"
use_CUDA = False
use_TTA = False

# grid parameters
bboxSize = 512 # in metres
offset = 0.33

## auxiliaries :-)
# create a testset and predict 
def predictAndStore(predImgList, coords, fileList=None, use_TTA=False):
    print ("number of images to predict:", len(predImgList))
    firstTime = True

    ensemble_list = [learner_path+"/"+nnet]
    #st.write("ensemble", ensemble_list)
    if len(ensemble_list) == 0:
        print ("no models!!! STOPPING...")
        assert(False)
        
    for fold in ensemble_list:
        print ("fold: ---", fold)
        learn=load_learner(fold)
    
        if use_CUDA:# put it all to GPU
            dl_test = learn.dls.test_dl(predImgList).to('cuda')
            learn.model = learn.model.cuda()
            learn.dls.to('cuda')
        else:
            dl_test = learn.dls.test_dl(predImgList)

        if not use_TTA: predictions = learn.get_preds(dl=dl_test)  
        else:           predictions = learn.tta(dl=dl_test)
        
        # free some memory
        learn.to('cpu')
        del learn, dl_test; gc.collect(); torch.cuda.empty_cache()
    
        # this is the ensembling, we compute the mean iteratively
        if firstTime:
            firstTime=False
            # initialization:
            preds = predictions[0]/len(ensemble_list)
        else:
            # subsequently we add the other predictions and divide by the number of models
            # this results in computing the mean
            preds += predictions[0]/len(ensemble_list)
       
        del predictions # free some memory.
    
    # extract coordinates
    x, y = zip(*coords)
    #store results: put it all into a dataframe and export it to CSV 
    df = pd.DataFrame(
        {'prob': preds.numpy()[:, 0],
         'x': x, 'y': y})
    return df

# yield a single snippet (iterator)
def get_snippet(tileList, bboxSize, offset=1.):
    stepsize = int(bboxSize * offset)
    for t_fn in tileList: # for all tiles in list
        with rio.open(t_fn) as tile:
            progress_text = "Operation in progress. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            # for all coordinates in this tile
            ##st.write ("width, stepsize, bboxSize, offset=", tile.width, stepsize, bboxSize, offset)
            for x in range(0, tile.width, stepsize): # we could probably subtract bboxSize, never mind
                my_bar.progress(min(int (x/tile.width*100),100), text=progress_text)
                ##st.write("snippet x=", x/tile.width)
                for y in range(0, tile.height, stepsize):
                    win=Window(x, y, bboxSize, bboxSize)
                    snippet = tile.read(1, window=win) # band 1, windowed
                    if snippet.shape != (bboxSize, bboxSize): continue # incomplete snippet => drop it
                    NaN_percentage = (snippet==tile.nodata).sum()
                    if NaN_percentage > 0: #ignore this snippet
                        st.write("NaN patch encountered, ignored.")
                        continue
                    if snippet.sum() == 0:
                        #st.write("black: ignore")
                        continue # ignore

                    normalized_snippet = normalizeImg(snippet, NORMALIZER).astype(np.uint8)
                    win_transform = tile.window_transform(win)
                    centerPoint = tile.xy(y+ snippet.shape[0]//2, x+ snippet.shape[1]//2)
                    filename = ''
                    # return image, filename, and center
                    yield (PILImage.create(normalized_snippet), filename, centerPoint, win_transform)
    yield (None, None, None, None) # end of list reached

MAX_IMAGES = 512 # how many should we batch process at once in memory?

# predict an entire tile list
def predict_tileList(bboxSize, offset, tileList):
    snippetList = []; coordList = []; fileList = []; transList = []; i = 0
    ret = pd.DataFrame(columns = ['prob','x','y'])

    for img, fn, coord, trans in get_snippet(tileList, bboxSize, offset):
        if img is not None: # end not yet reached
            snippetList.append(img)
            fileList.append(fn) # only for reference, file may be not created!!!
            coordList.append(coord) # center point
            transList.append(trans) # center point
            i += 1

        if i >= MAX_IMAGES or img==None: # process batch if enough images or end reached
            st.write ("new batch")
            if snippetList: # list not empty
                chunk_ret = predictAndStore(snippetList, coordList, fileList, use_TTA=use_TTA)
                ret = pd.concat([ret, chunk_ret]) ## add it to result
                snippetList = []; coordList = []; fileList = []; transList = []; i = 0
    st.write("AI processed ", len(ret), "patches.")
    return ret

class Predict:
    def __init__(self):
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.prepare_output()
            self.get_prediction()
            self.process_output()
    
    @staticmethod
    def get_image_from_upload():
        # gets images
        uploaded_file = st.file_uploader("Upload Files",type=['tif'])
        if uploaded_file is not None:
            with open('./mytile.tif', 'wb') as f: f.write(uploaded_file.getbuffer())
        return uploaded_file
    
    def prepare_output(self):
        st.write ("File loaded...now predicting...")

    def get_prediction(self):
        tileList =["./mytile.tif"]
        self.recfile = "HillfortFinderResults.csv"
        self.ret = predict_tileList(bboxSize, offset, tileList)
        #self.csv = self.ret.to_csv(self.recfile, index=False, encoding='utf-8')
    
    ##def convert_df(df): return df.to_csv().encode('utf-8')

    def process_output(self):
        # shows uploaded image
        # TODO use to show progress bar, instructions, download file, etc.
        st.write('PROCESSING COMPLETE...')      
        csv = self.ret.to_csv(index=False).encode('utf-8')##convert_df(self.ret)
        st.download_button(label="Download data as CSV", data=csv, file_name=self.recfile, mime='text/csv')
        
if __name__=='__main__':
    st.set_page_config(page_title="Hillfort Finder App")
    st.header("Hillfort Finder App")
    predictor = Predict()

