'''
Functions for extract probabilities of LULC types from Google Dynamyc World
averaged across defined period of time and area of interest

Brown, C.F., Brumby, S.P., Guzder-Williams, B. et al. Dynamic World, Near real-time global 10 m land use land cover mapping. Sci Data 9, 251 (2022). https://doi.org/10.1038/s41597-022-01307-4

Account in Google Earth Engine is required for using the module.

Authentification in Google Earth Engine is required before using the functions, i.e.:
ee.Authenticate() 
'''

import math

import numpy as np
import ee
import rasterio.features
import shapely
from shapely.affinity import translate, scale
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj

# Dataset located at Google Earth Engine
ee.Initialize()

def get_scale(width: float, height: float, resolution: int=10, max_element=1e5) -> int:
    ''' Define optimal scale for reducing number of elements returned from GEE

    Parameters
    ----------
    width : float
        Width of AOI (area of interest) in units of crs (m)
    height : float
        Height of AOI in units of crs (m)
    resolution : int, default 10
        Resolution of LULC map. It is 10 m in GDW
    max_element : int, default 1e5
        Maximum limit elements (pixels) returned from GEE

    Returns
    -------
    int
        Optymal scale factor (m / px)
    '''

    area = height/resolution * width/resolution
    if area < max_element:
        return resolution
    scale = math.ceil(area / max_element)
    return scale * resolution

def convert_polygon(shapely_polygon: Polygon) -> ee.Geometry.Polygon:
    ''' Convert polygon from shapely into Googl Earth Engine format

    Parameters
    ----------
    shapely_polygon : shapely.geometry.Polygon
        polygon in shapely format

    Returns
    -------
    ee.Geometry.Polygon
        poligon in GEE format

    '''

    if isinstance(shapely_polygon, shapely.geometry.MultiPolygon):
        polygons = list(exterior)
        coords = [np.dstack((p.exterior.coords.xy)).tolist() for p in polygons]
        ee_polygon = ee.Geometry.MultiPolygon(coords)
        return ee_polygon

    x, y = shapely_polygon.exterior.coords.xy
    ee_polygon = ee.Geometry.Polygon(np.dstack((x,y)).tolist())
    return ee_polygon

def get_band(tile: ee.image.Image, aoi: ee.Geometry.Polygon, band: str) -> np.array:
    ''' Extracts rectangle from one of the band from Google Earth Engine image

    Parameters
    ----------
    tile : ee.image.Image
        GEE image that contains required band
    aoi : ee.Geometry.Polygon
        Area of interest which bounds used for the rectangle extraction
    band : str
        Name of the image band

    Returns
    -------
    np.array
        2d array with values in AOI from the band
    '''

    array = np.array(tile.sampleRectangle(aoi).get(band).getInfo())
    return array.T

def transform_crs(geometry: Polygon, source_crs: str, destination_crs: str) -> Polygon:
    ''' Transforms polygon from source coordinate reference system into
        destination reference system

    Parameters
    ----------
    geometry: shapely.geometry.Polygon
        Polygon in a source reference system
    source_crs: str
        String that represents source CRS, like 'EPSG:4328'
    destination_crs: str
        String that represents destination CRS

    Returns
    -------
    shapely.geometry.Polygon
        Polygon in a destination reference system
    '''

    reproject = pyproj.Transformer.from_proj(
        pyproj.Proj(source_crs),
        pyproj.Proj(destination_crs))
    new_geometry = transform(reproject.transform, geometry)
    return new_geometry

def gdw_get_mean_probabilities(polygon: Polygon,
                               crs: str,
                               startDate:str,
                               endDate: str) -> dict:
    ''' Calculates mean probabilities of LULC types from Google Dynamyc World.
        The probabilitieas are averaged across defined period of time and area of interest
        The LULC bands in Google Dynamic World: water, trees, grass, flooded_vegetation,
        crops, shrub_and_scrub, built, bare, snow_and_ice

    Parameters
    ----------
    polygon : Polygon
        Area of interest
    crs : str
        Coordinate reference system of the area of interest
    startDate : str
        Start of a period for averaging probability of LULC in a format 'YYYY-MM-DD'
    endDate : str
        End of a period for averaging probability of LULC in a format 'YYYY-MM-DD'

    Returns
    -------
    dict
        Dictionary with LULC categories as keys, 
        and their mean probabilities in AOI for defined perion as values
    '''

    PROBABILITY_BANDS = [
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub',
        'built', 'bare', 'snow_and_ice'
    ]
    mean_probabilities = {}

    aoi = convert_polygon(polygon)
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterBounds(aoi)
          .filterDate(startDate, endDate)
          .filter(ee.Filter.contains('.geo', aoi)))
    crs_dw = dw.first().projection().crs().getInfo()

    mask = transform_crs(polygon, crs, crs_dw)
    # mask = formeragro_df_map.to_crs(crs_dw).iloc[index]['geometry']
    width = mask.bounds[2] - mask.bounds[0]
    height = mask.bounds[3] - mask.bounds[1]
    mask = translate(mask, -mask.bounds[0], -mask.bounds[1])
    scale_dw = get_scale(width=width, height=height)

    projection = ee.Projection(crs_dw).atScale(scale_dw)
    probabilityCol = dw.select(PROBABILITY_BANDS)
    meanProbability = probabilityCol.reduce(ee.Reducer.mean())
    meanProbability = meanProbability.setDefaultProjection(projection)

    for iter, band in enumerate(PROBABILITY_BANDS):
        prob_array = get_band(tile=meanProbability,
                              aoi=aoi,
                              band=band+'_mean')
        if iter == 0:
            x_scale = prob_array.shape[0]/mask.bounds[2]
            y_scale = prob_array.shape[1]/mask.bounds[3]
            mask = scale(geom=mask, xfact=x_scale, yfact=y_scale, origin=(0, 0))
            mask = rasterio.features.rasterize([mask.buffer(-1)],
                                               prob_array.T.shape).T.astype(bool)

        prob_array = np.ma.masked_array(prob_array, mask=~mask)
        mean_probabilities[band] = prob_array.mean()

    return mean_probabilities
