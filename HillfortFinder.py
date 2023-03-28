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
            progress_text = "LiDAR tile uploaded, now predicting hillforts. This could take several minutes..."
            my_bar = st.progress(0, text=progress_text)
            # for all coordinates in this tile
            ##st.write ("width, stepsize, bboxSize, offset=", tile.width, stepsize, bboxSize, offset)
            for x in range(0, tile.width, stepsize): # we could probably subtract bboxSize, never mind
                my_bar.progress(min(int (x/tile.width*100),100), text=progress_text)
                ##st.write("snippet x=", x/tile.width)
                for y in range(0, tile.height, stepsize):
                    win=Window(x, y, bboxSize, bboxSize)
                    snippet = tile.read(1, window=win) # band 1 only, windowed
                    if snippet.shape != (bboxSize, bboxSize): continue # incomplete snippet => drop it
                    NaN_percentage = (snippet==tile.nodata).sum()
                    if NaN_percentage > 0: #ignore this snippet
                        #st.write("NaN patch encountered, ignored.")
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

MAX_IMAGES = 256 # how many should we batch process at once in memory?

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
            print("new batch")
            if snippetList: # list not empty
                chunk_ret = predictAndStore(snippetList, coordList, fileList, use_TTA=use_TTA)
                ret = pd.concat([ret, chunk_ret]) ## add it to result
                snippetList = []; coordList = []; fileList = []; transList = []; i = 0
    st.write("COMPLETE! ", len(ret), "patches from your tile were processed")
    return ret

def spatial_resolution(raster):
    t = raster.transform
    x = t[0]; y =-t[4]
    return x, y

def get_image_from_upload():
    st.write('Please upload your LiDAR tile here.')
    st.write('Note that the "raw" DEM (Digital Elevation Model) produces best results. Visualisations such as hillshade are not tested.')
    uploaded_file = st.file_uploader("Upload your LiDAR tile:",type=['tif'])
    if uploaded_file is not None:
        with open('mytile.tif', 'wb') as f: f.write(uploaded_file.getbuffer())
        # analyse data
        dataset = rio.open('mytile.tif')
        count = dataset.count
        st.write('Now analysing your tile:')
        st.write('* number of bands = ', count)
        if count > 1: st.write(' ==> Your tile has more than one LiDAR band, we use only band #1')
        if dataset.crs is None: 
            st.write('==> Your tile does not have a coordinate reference system (CRS). Aborting...')
            assert(False) # hard abort :-)
        else:
            st.write('* CRS = ', dataset.crs.to_epsg())
            res = spatial_resolution(dataset)
            st.write('* Resolution (m) = ', res[0])
            if res[0] < 1. or res[0] > 2.: 
                st.write('!!! This App has only been tested *with 1m and 2m resolution only*, use at own risk !!!')
    return uploaded_file

def get_prediction():
    if st.session_state.IMG is not None:
        ret = predict_tileList(bboxSize, offset, tileList)
        return ret
    else: st.write("IMG is None!")

import base64
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Click here to download results</a>'
    return href

def process_output(ret):
    href = get_table_download_link(ret)
    st.markdown(href, unsafe_allow_html=True)
    st.write("How to use the output? Find instructions here:")
    st.markdown(f'<a href="https://github.com/juergenlandauer/HillfortFinderApp/raw/main/HillfortFinderApp%20usage%20manual.pdf">HillfortFinderApp usage manual</a>', unsafe_allow_html=True)

tileList =["./mytile.tif"]
recfile = "HillfortFinderResults.csv"

if __name__=='__main__':
    st.session_state.IMG = None
    st.set_page_config(page_title="Hillfort Finder App")
    st.header("Hillfort Finder App")
    st.session_state.IMG = get_image_from_upload()
    if st.session_state.IMG is not None:
        ret = get_prediction()
        process_output(ret)
