# ## Hillfort Predictor 

# This computes the predictions for a list of areas (in batch mode)

# This just loads some libraries
import glob
import os

from shutil import copyfile
import rasterio as rio
from rasterio.windows import Window
from fastai.vision.all import *

# range normalizer
NORMALIZER = "SCALE"
from normalizer import normalizeImg
#VERSION = 'L100'
VERSION = 'L101'
SAVE_THRESH = .75 # 85%

# should we ignore snippets with the wrong size?
IGNORE_INCOMPLETE_SNIPPETS = True
#IGNORE_INCOMPLETE_SNIPPETS = False

# should we use Nvidia CUDA?
##use_CUDA = True
use_CUDA = False # too slow here

# path to CNN model
learner_path = "./"
# file name of neural network *.pkl file (weights)
nnet = "m_" + str(VERSION) +"_*.pkl"

# use TTA (test time augmentation)? It always never helped so far though...
use_TTA = False 

# grid parameters
bboxSize = 512 # in metres
offset = 0.33

# create a testset and predict 
def predictAndStore(predImgList, coords, fileList=None, use_TTA=False):
    print ("number of images to predict:", len(predImgList))
    firstTime = True

    ensemble_list = list(learner_path+"/"+nnet)
    if len(ensemble_list) == 0:
        print ("no models!!! STOPPING...")
        assert(False)
        
    # we iterate over the ensembled models:
    for fold in ensemble_list:
        learn=load_learner(fold)
    
        dl_test = learn.dls.test_dl(predImgList)
        predictions = learn.get_preds(dl=dl_test)  
        del learn, dl_test; gc.collect(); torch.cuda.empty_cache()
        if firstTime:
            preds = predictions[0]/len(ensemble_list)
        del predictions # free some memory.

    # extract coordinates
    x, y = zip(*coords)
    df = pd.DataFrame(
        {'prob': preds.numpy()[:, 0],
         'x': x,
         'y': y,
         'data': fileList
        })
    
    return df

# get coordinates
def getCoordsFromGeoTiffList(imagelist):
    coordList = []

    for im in imagelist:
        with rio.open(im) as imgrio:
            center_imgrio = imgrio.xy(imgrio.width // 2, imgrio.height // 2) # center coordinate
            #center_imgrio = imgrio.xy(0,0) # corner coordinate
            coordList.append(center_imgrio)
            #print (center_imgrio, imgrio.bounds, imgrio.crs)
    
    return coordList

# yield a single snippet (iterator)
def get_snippet(tileList, bboxSize, offset=1.):
    stepsize = int(bboxSize * offset)

    for t_fn in tileList: # for all tiles in list
        with rio.open(t_fn) as tile:
            # for all coordinates in this tile
            for x in range(0, tile.width, stepsize): # we could probably subtract bboxSize, never mind
                #print (t_fn, x)
                for y in range(0, tile.height, stepsize):
                    win=Window(x, y, bboxSize, bboxSize)
                    snippet = tile.read(1, window=win) # band 1, windowed
                    if snippet.shape != (bboxSize, bboxSize): 
                        if IGNORE_INCOMPLETE_SNIPPETS:
                            #print ("ignore>", (x,y), snippet.shape)
                            continue # incomplete snippet => drop it
                
                    NaN_percentage = (snippet==tile.nodata).sum()
                    if NaN_percentage > 0: #ignore this snippet
                        continue
                        
                    if snippet.sum() == 0: continue # ignore

                    normalized_snippet = normalizeImg(snippet, NORMALIZER).astype(np.uint8)
                    win_transform = tile.window_transform(win)
                    centerPoint = tile.xy(y+ snippet.shape[0]//2, x+ snippet.shape[1]//2)
                    #print ("from xy", centerPoint, x,y)
                    filename = SNIPPETS_PATH+t_fn.name+"_"+str(centerPoint)+"_"+\
                               str((snippet.shape[1],snippet.shape[0]))+".tif"
                    #print (filename)

                    if False and SAVE_SNIPPETS: # Store images for prediction
                        out_meta = tile.meta.copy()
                        out_meta.update({
                            "height":    snippet.shape[0],
                            "width":     snippet.shape[1],
                            "transform": win_transform,
                            #"count":     1,
                            "dtype":     rio.ubyte, # 8 bit
                            "nodata":    0,
                            "compression": "lzw"
                        })
                        with rio.open(filename, 'w', **out_meta) as dst: dst.write(normalized_snippet, indexes=1)
    
                    # return image, filename, and center
                    yield (PILImage.create(normalized_snippet), filename, centerPoint, win_transform)
                # end for
        # end with
        #break # DEBUG: 1st tile only
    yield (None, None, None, None) # end of list reached - NEEDED?

MAX_IMAGES = 1024 # how many should we batch process at once in memory?

# predict an entire tile list
def predict_tileList(bboxSize, offset, tileList):
    snippetList = []; coordList = []; fileList = []; transList = []; i = 0
    ret = pd.DataFrame(columns = ['prob','x','y','data'])

    for img, fn, coord, trans in get_snippet(tileList, bboxSize, offset):
        if img is not None: # end not yet reached
            snippetList.append(img)
            fileList.append(fn) # only for reference, file may be not created!!!
            coordList.append(coord) # center point
            transList.append(trans) # center point
            i += 1
    
        if i >= MAX_IMAGES or img==None: # process batch if enough images or end reached
            if snippetList: # list not empty
                chunk_ret = predictAndStore(snippetList, coordList, fileList, use_TTA=use_TTA)
                ## add it to result
                ret = pd.concat([ret, chunk_ret])
                 
                if SAVE_SNIPPETS: # Store images for prediction
                    chunk_ret = ret[ret['prob']>=SAVE_THRESH]
                    with rio.open(tileList[0]) as tile: #dummy
                        for index, row in chunk_ret.iterrows():
                            normalizedSnippet = normalizeImg(snippetList[index], 'SkyViewFactor')###.astype(np.uint8)
                            normalizedSnippet = np.squeeze(normalizedSnippet)
                            print (np.min(normalizedSnippet), np.max(normalizedSnippet), normalizedSnippet.shape)
                            out_meta = tile.meta.copy()
                            out_meta.update({
                                "height":    snippetList[index].shape[0],
                                "width":     snippetList[index].shape[1],
                                "transform": transList[index],
                                "count":     1,
                                "dtype":     rio.ubyte, # 8 bit
                                "nodata":    0,
                                "compression": "lzw"
                            })
                            # remove ending 4-char ".tif"
                            fn = fileList[index][:-4]+"_"+f"{row['prob']:.2f}"+'.tif'
                            print (fn)
                            with rio.open(fn, 'w', **out_meta) as dst: 
                                dst.write(normalizedSnippet, indexes=1)

                snippetList = []; coordList = []; fileList = []; transList = []; i = 0
    return ret

def predictTilesChilterns():
    tileList =get_image_files("../Data/LiDAR_tiles/Chilterns/")
    recfile = "./hits/record_allTestareas_Chilterns_"+str(VERSION)+"_"+str(offset)+"_"+str(patchSampleRate)+".csv"
    predictTiles(tileList, recfile)

#NUM_WORKERS = 1

def predictTiles(tileList, recfile):

    ret = predict_tileList(bboxSize, offset, tileList)
    ret.to_csv(recfile, index=False)
predictTilesChilterns()

