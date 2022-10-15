''' Functions for extract probabilities of LULC types from Google Dynamyc World
averaged across defined period of time and area of interest

Brown, C.F., Brumby, S.P., Guzder-Williams, B. et al. Dynamic World,
Near real-time global 10 m land use land cover mapping.
Sci Data 9, 251 (2022). https://doi.org/10.1038/s41597-022-01307-4

Account in Google Earth Engine is required for using the module.

Authentification in Google Earth Engine is required before using the functions,
i.e.: ee.Authenticate()

Author: Aleksander Nikitin
email: nikitinale@gmail.com
'''

import math
import time

import numpy as np
import ee
import rasterio.features
import shapely
from shapely.affinity import translate, scale
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj

# Dataset located at Google Earth Engine
# EE functions cannot not be used without initialization
ee.Initialize()

# All LULC types in Dynamic World Land Use Land Cover classification taxonomy
BANDS = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub',
        'built', 'bare', 'snow_and_ice']

def get_scale(width: float, height: float, resolution: int = 10, max_element=1e5) -> int:
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


def convert_polygon(shapely_polygon: Polygon,
                    crs: str = 'EPSG:4326') -> ee.Geometry.Polygon:
    ''' Convert polygon or multipoligon from shapely
        into Googl Earth Engine format

    Parameters
    ----------
    shapely_polygon : shapely.geometry.Polygon or MultiPolygon
        polygon in shapely format

    Returns
    -------
    ee.Geometry.Polygon or MultiPolygon
        poligon in GEE format

    '''

    projection = ee.Projection(crs)
    # if geometry not single polygon but multipolygon
    if isinstance(shapely_polygon, shapely.geometry.MultiPolygon):
        polygons = list(shapely_polygon)
        coords = [np.dstack((p.exterior.coords.xy)).tolist() for p in polygons]
        ee_polygon = ee.Geometry.MultiPolygon(coords, proj=projection)
        return ee_polygon

    x, y = shapely_polygon.exterior.coords.xy
    ee_polygon = ee.Geometry.Polygon(
        np.dstack((x, y)).tolist(), proj=projection)
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


def transform_crs(geometry: Polygon,
                  source_crs: str,
                  destination_crs: str) -> Polygon:
    ''' Transforms polygon or multypoligon from source coordinate 
        reference system into destination coordinate reference system

    Parameters
    ----------
    geometry: shapely.geometry.Polygon
        Polygon in a source reference system
    source_crs: str
        String that represents source CRS, like 'EPSG:4326'
    destination_crs: str
        String that represents destination CRS

    Returns
    -------
    shapely.geometry.Polygon
        Polygon in a destination reference system
    '''

    reproject = pyproj.Transformer.from_proj(
        pyproj.Proj(source_crs),
        pyproj.Proj(destination_crs),
        always_xy=True)
    new_geometry = transform(reproject.transform, geometry)
    return new_geometry

def mode(array: np.array) -> any:
    ''' Calculate mode in the <array> -- the most frequent value.

    Parameters:
    -----------
    array: np.ma.array (any dtype)
        input array

    Returns: any (according to the dtype of the array)
    --------
        The most frequent value in the array
    '''

    freq = np.unique(array, return_counts=True)
    idx = np.argmax(freq[1])
    return freq[0][idx]


def none_fun(array: np.array) -> any:
    ''' Return unmasked flatten array

    Parameters:
    -----------
    array: np.ma.array (any dtype)
        input array

    Returns: np.array
    --------
        Unmasked flatten array        
    '''

    data = array[~array.mask]
    return data.data


def gdw_get_mean_probabilities(polygon: Polygon,
                               crs: str,
                               startDate: str,
                               endDate: str,
                               place_id: any=None,
                               bands: list=BANDS,
                               reducer_time: str='mean',
                               reducer_spatial: str='mean',
                               sequence: bool=True) -> dict:
    ''' Calculates mean probabilities of LULC types from Google Dynamyc World.
        The probabilitieas are averaged across defined period of time and
        defined area of interest.

    Parameters
    ----------
    polygon : Polygon
        Area of interest
    crs : str
        Coordinate reference system of the area of interest
    startDate : str
        Start of a period for averaging probability of LULC in a format
        'YYYY-MM-DD'
    endDate : str
        End of a period for averaging probability of LULC in a format
        'YYYY-MM-DD'
    bands: list of str, default ['water', 'trees', 'grass',
        'flooded_vegetation', 'crops', 'shrub_and_scrub',
        'built', 'bare', 'snow_and_ice']
        Bands of Dynamic World for aggregation. Possible options:
        'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
        'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
    place_id : str or int, default None
        ID for the record with averaged probabilities of LULC
    sequence : Bool, default True
        If data are retrieved in a long sequence the 1 second pause
        enchance stability of the process
    reducer_time : str, default 'mean'
        Function for aggregation probabilities of LULC types (bands) across
        time dimension. Possible options: 'mean', 'max', 'min', 'median',
        'mode', 'std'. Mode aggregation makes sense for assessing the most probable
        type of LULC.
    reducer_spatial : str, default 'mean'
        Function for aggregation probabilities of LULC types (bands) across
        region of interest. Possible options: 'mean', 'max', 'min', 'median',
        'mode', 'std', 'none'. Mode aggregation makes sense for assessing the most
        probable type of LULC. 'none' means no spatial aggregation,
        numpy.arrays with pixel values will returned for each band. Elements
        of the arrays with same index represent the same pixels.

    Returns
    -------
    dict
        Dictionary with LULC categories as keys, 
        and their mean probabilities in AOI for defined perion as values

    Notes:
    ------
        The LULC bands in Google Dynamic World: water, trees, grass,
        flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice.
        Band 'label' not included by default.
    '''

    if sequence:
        time.sleep(1)
    if place_id:
        mean_probabilities = {'id': place_id}
    else:
        mean_probabilities = {}

    # creates AOI in ee.Polygon format with crs='EPSG:4326'
    if not ('epsg:4326' in crs.to_string().lower()):
        _polygon = transform_crs(polygon, crs, 'EPSG:4326')
        aoi = convert_polygon(_polygon)
    else:
        aoi = convert_polygon(polygon)

    # search all DW products that contains time perion and AOI with buffer
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterBounds(aoi)
          .filterDate(startDate, endDate)
          .filter(ee.Filter.contains('.geo', aoi.buffer(5e2))))
    crs_dw = dw.first().projection().crs().getInfo()

    # converts AOI mask in crs of the GDW products
    mask = transform_crs(polygon, crs, crs_dw)
    width = mask.bounds[2] - mask.bounds[0]
    height = mask.bounds[3] - mask.bounds[1]
    mask = translate(mask, -mask.bounds[0], -mask.bounds[1])
    # appropriate scale optimize number of elements returned from GEE
    scale_dw = get_scale(width=width, height=height)

    projection = ee.Projection(crs_dw).atScale(scale_dw)
    probabilityCol = dw.select(bands)
    treducer_fun = {'mean': ee.Reducer.mean(),
                    'max': ee.Reducer.max(),
                    'min': ee.Reducer.min(),
                    'median': ee.Reducer.median(),
                    'mode': ee.Reducer.mode(),
                    'std': ee.Reducer.stdDev()}[reducer_time]
    meanProbability = probabilityCol.reduce(treducer_fun)
    meanProbability = meanProbability.setDefaultProjection(projection)

    sreducer_fun = {'mean': np.mean,
                   'max': np.max,
                   'min': np.min,
                   'median': np.median,
                   'mode': mode,
                   'std': np.std,
                   'none': none_fun}[reducer_spatial]
    x_scale = 0
    for band in bands:
        try:
            prob_array = get_band(tile=meanProbability,
                                  aoi=aoi,
                                  band=band+'_'+reducer_time)
            if x_scale == 0:
                x_scale = prob_array.shape[0]/mask.bounds[2]
                y_scale = prob_array.shape[1]/mask.bounds[3]
                mask = scale(geom=mask, xfact=x_scale,
                             yfact=y_scale, origin=(0, 0))
                mask = rasterio.features.rasterize([mask.buffer(-1)],
                                                   prob_array.T.shape).T.astype(bool)

            prob_array = np.ma.masked_array(prob_array, mask=~mask)
            mean_probabilities[band] = sreducer_fun(prob_array[~prob_array.mask])
        except ee.EEException as e:
            print('Error in samplig probabilities values', e)
            mean_probabilities[band] = None
        except ValueError as e:
            print('ValueError: ', e, mask)
            mean_probabilities[band] = None
        except TypeError as e:
            print('TypeError: ', e, mask)
            mean_probabilities[band] = None
        except:
            print('Unknown error...')
            mean_probabilities[band] = None
    return mean_probabilities
