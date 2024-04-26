import os
import leafmap
import torch
from samgeo.text_sam import LangSAM
import glob
import argparse
import time


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


if __name__ =='__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("--image_path", type=str, required=True) 
  parser.add_argument("--text_prompt", type=str, required=True)
  parser.add_argument("--folder_name", type=str, required=True)

  args = parser.parse_args()

  IMAGEPATH = args.image_path
  TEXTPROMPT = args.text_prompt
  FOLDERNAME = args.folder_name

  #list_text_values = {0.5}
  #list_box_values = {0.5}

  list_text_values = {0.2, 0.4, 0.6, 0.8}
  
  # Values of 0 and 1 were discarted because they contained zero to all bounding boxes
  list_box_values = {0.2, 0.3, 0.4, 0.5, 0.6, 0.8}


  IMAGEPATH=IMAGEPATH[:-1]
  print(IMAGEPATH)

  pathObj = os.path.split(IMAGEPATH)
  patchName = pathObj[1][:-4]
  resultPath = pathObj[0]
  rootFolder = os.path.abspath(os.path.join(resultPath, '..'))
  
  if torch.cuda.is_available():
    print("GPU is found")
  else:
    print("GPU is not found")

  resultsTiffMaskPath = "{}/Results/{}/TiffMask".format(rootFolder, FOLDERNAME)   
  resultsGDBoxPath = "{}/Results/{}/GDBox".format(rootFolder, FOLDERNAME)   
  resultsTiff2SHPPath = "{}/Results/{}/Tiff2SHP".format(rootFolder, FOLDERNAME)   
  resultsGDBoxSHPPath = "{}/Results/{}/GDBoxSHP".format(rootFolder, FOLDERNAME)   

  if not os.path.exists(resultsTiffMaskPath):
    os.makedirs(resultsTiffMaskPath)
    print("Path:", resultsTiffMaskPath)

  if not os.path.exists(resultsGDBoxPath):
    os.makedirs(resultsGDBoxPath)

  if not os.path.exists(resultsTiff2SHPPath):
    os.makedirs(resultsTiff2SHPPath)

  if not os.path.exists(resultsGDBoxSHPPath):
    os.makedirs(resultsGDBoxSHPPath)

  sam = LangSAM()

  for text in list_text_values:
    for box in list_box_values:

      text_str=str(int(text*100))
      box_str=str(int(box*100))

      maskPath = "{}/{}_{}_{}_{}.tif".format(resultsTiffMaskPath, patchName, TEXTPROMPT, text_str, box_str)     
      boxDinoPath = "{}/{}_{}_{}_{}_box.tif".format(resultsGDBoxPath, patchName, TEXTPROMPT, text_str, box_str)
      shpNamePath = "{}/{}_{}_{}_{}.shp".format(resultsTiff2SHPPath, patchName, TEXTPROMPT, text_str, box_str)
      boxNamePath = "{}/{}_{}_{}_{}.shp".format(resultsGDBoxSHPPath, patchName, TEXTPROMPT, text_str, box_str)

      print(patchName, " text:", text_str, " box:", box_str)

      sam.masks = None

      start_time = time.time()  # <-- Start the timer
      sam.predict(IMAGEPATH, TEXTPROMPT, box_threshold=box, text_threshold=text, output=maskPath)
      duration = time.time() - start_time  # <-- Calculate the duration

      print(f"Prediction took {duration:.2f} seconds.")  # <-- Print the duration


      if (sam.masks != None):
        print("Save box")
        sam.show_anns(
          cmap='Blues',
          box_color='red',
          blend=True,
          output=boxDinoPath
        )
        
        sam.raster_to_vector(maskPath, shpNamePath)
        sam.save_boxes(boxNamePath)

  torch.cuda.empty_cache()
