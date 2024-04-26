import torch
from samgeo import SamGeo
import glob
import geopandas as gpd
import os
import argparse
import time

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

class sites:
  def __init__(self, pathTIF, imagesPath, labelsPath, outputPath):
    self.pathTIF = pathTIF
    self.imagesPath = imagesPath
    self.labelsPath = labelsPath
    self.outputPath = outputPath

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path_tif", type=str, required=True) 
    parser.add_argument("--folder_path", type=str, required=True) 
    parser.add_argument("--folder_shp", type=str, required=True) 
    parser.add_argument("--folder_point_seg", type=str, required=True) 

    args = parser.parse_args()

    tif_field = args.folder_path_tif
    path_field = args.folder_path
    shp_field = args.folder_shp
    point_seg_field = args.folder_point_seg

    siteDB = sites(tif_field, path_field, shp_field, point_seg_field)

    sam = SamGeo(
        model_type="vit_h",
        checkpoint="sam_vit_h_4b8939.pth",
        automatic=False,
        sam_kwargs=None,
    )

    if not os.path.exists(point_seg_field):
        os.makedirs(point_seg_field)
        print("Path:", point_seg_field)

    for patchPath in os.listdir(siteDB.imagesPath):

        patchPathFile = "{}/{}".format(siteDB.imagesPath, patchPath)
        patchName = patchPath[:-4]
        shpNameWater = "{}/{}_{}.shp".format(siteDB.labelsPath, patchName, 'water')
        shpNameBuildings = "{}/{}_{}.shp".format(siteDB.labelsPath, patchName, 'buildings')

        gdf_water = gpd.read_file(shpNameWater)
        gdf_buildings = gpd.read_file(shpNameBuildings)

        coord_list_water = [(x,y) for x,y in zip(gdf_water['geometry'].x , gdf_water['geometry'].y)]
        coord_list_buildings = [(x,y) for x,y in zip(gdf_buildings['geometry'].x , gdf_buildings['geometry'].y)]

        #If shapefile has information
        if coord_list_water or coord_list_buildings:

            sam.set_image(patchPathFile)

            if coord_list_water:
                outputTIFNameWater = "{}/{}_{}_gs.tif".format(siteDB.outputPath, patchName, 'water')
                outputSHPNameWater = "{}/{}_{}_gs.shp".format(siteDB.outputPath, patchName, 'water')
                sam.predict(coord_list_water, point_labels=1, point_crs="EPSG:4326", output=outputTIFNameWater)
                sam.tiff_to_vector(outputTIFNameWater, outputSHPNameWater)

            if coord_list_buildings:
                outputTIFNameBuildings = "{}/{}_{}_gs.tif".format(siteDB.outputPath, patchName, 'buildings')
                outputSHPNameBuildings = "{}/{}_{}_gs.shp".format(siteDB.outputPath, patchName, 'buildings')
                sam.predict(coord_list_buildings, point_labels=1, point_crs="EPSG:4326", output=outputTIFNameBuildings)
                sam.tiff_to_vector(outputTIFNameBuildings, outputSHPNameBuildings)

        else:
            print("Empty list")

    torch.cuda.empty_cache()
