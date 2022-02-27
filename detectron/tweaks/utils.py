import numpy as np
import json,copy
import glob
import os,csv, sys, cv2
import time, ast, fnmatch

import base64
import matplotlib.pyplot as plt
import random
import pandas as pd

from detectron2.structures import BoxMode
from copy import deepcopy
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image
from collections import Counter
from math import (
    degrees, radians,
    sin, cos, asin, tan, atan, atan2, pi,
    sqrt, exp, log, fabs, log10, pow
)
from multiprocessing import Pool
from functools import partial
# from constants import (
#     EARTH_MEAN_RADIUS,
#     EARTH_MEAN_DIAMETER,
#     EARTH_EQUATORIAL_RADIUS,
#     EARTH_EQUATORIAL_METERS_PER_DEGREE,
#     I_EARTH_EQUATORIAL_METERS_PER_DEGREE,
#     HALF_PI,
#     QUARTER_PI,
# )
csv.field_size_limit(sys.maxsize)
ZOOMS = {
            0:18,
            1:18,
            2:15,
            3:17,
            4:15,
            5:15,
            6:15
        }
GOLD_SIZES = {
            0:25,
            1:25,
            2:3,
            3:12,
            4:3,
            5:3,
            6:3
        }
#  centers in lat , lon
CENTRES = {
            0:[37.73755663692416, -122.19795016945281],
            1:[32.58577585559755, -117.09164085240766],
            2:[32.61748188924153, -117.14119088106783],
            3:[32.60760476678458, -117.08442647549721],
            4:[37.694753719037756, -122.19294177307802],
            5:[37.71336706451458, -122.19060472858666],
            6:[32.59795016014067, -117.11036626803674]
        }


# note the different order between marios FIELDITEMS and LXMERT FIELDNAMES
# FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
#               "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def multiAnnotations(anno, centre, img_dim,classes,sn_id):
            if 'points_x' in anno.keys():
                px = anno["points_x"]
                py = anno["points_y"]
                tmp_ob = copy.deepcopy(anno)
                # input(tmp_ob)
                # tmp_ob['coordinates'] = [newGPS[1], newGPS[0]]
                tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],img_dim)
                # drawMapPixels([],filename,[anno])
                if tmp_pixs:
                    px = tmp_pixs["points_x"]
                    py = tmp_pixs["points_y"]
                # else:
                #     print(anno['name'])
                #     print("????!!!!!!!!!!")
            else:
                # generate points x and y
                # new_scenario = []
                # for item in scenario:
                #     new_item = copy.deepcopy(item)
                # print(height, width)
                # print(sn_id, )
                # print(img_dim)
                newGPS = getPointLatLng(anno['pixels_bb'][0]+10, anno['pixels_bb'][1]+10,CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id],img_dim[1],img_dim[0])
                # print(newGPS)
                tmp_ob = copy.deepcopy(anno)
                # print(tmp_ob['coordinates'],[newGPS[1], newGPS[0]])
                tmp_ob['coordinates'] = [newGPS[1], newGPS[0]]
                tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],img_dim)
                # drawMapPixels([],filename,[anno])
                if tmp_pixs:
                    px = tmp_pixs["points_x"]
                    py = tmp_pixs["points_y"]
                else:
                    # print(anno)
                    # print(anno['name'])
                    return {}

            try:
                categ_ = classes.index(anno['keywords'].split(' ')[0])
            except Exception as exc:
                # print(exc)
                categ_ = 0
            poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            assert np.min(px) > 0 and  np.min(py) > 0
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": categ_,
                "iscrowd":0,
                "name":anno['name']

            }
            return obj

def get_rosmi_dicts(rosmi_path,dtype):
    img_dir = os.path.join(rosmi_path,'images')
    json_file = os.path.join(rosmi_path, f"{dtype}.json")

    with open(json_file) as f:
        imgs_anns = json.load(f)
    with open('ROSMI/cleaned.json') as cln_f:
        classes = json.load(cln_f)
    regions_ = {}
    for idx in range(7):
        print(f"Reading scenario {idx}")
        with open(os.path.join(rosmi_path,f'scenario{idx}.json')) as s_f:
            regions_[f'scenario{idx}.json'] = json.load(s_f)

    dataset_dicts = []
    for scenario in tqdm(imgs_anns):
        record = {}

        for ke, va in scenario.items():
            record[ke] = va
        filename = os.path.join(img_dir, scenario["image_filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = scenario['img_id']
        record["scenario"] = scenario['scenario_items']
        record["height"] = height
        record["width"] = width


        sn_id = int(scenario['scenario_items'].split('rio')[1].split('.j')[0])
        centre = calculateTiles(CENTRES[sn_id],ZOOMS[sn_id])

        try:
            img = Image.open(filename)
        except Exception as e:
            print(e)
            continue

        img_dim = [img.getbbox()[2],img.getbbox()[3]]
        # print(height, width)
        # img_dim[0] is width
        # print(img_dim[1], img_dim[0])
        # input("?")

        annos = regions_[scenario['scenario_items']] + scenario['dynamo_obj']
        objs = []
        # multiAnnotations(anno, centre, img_dim,classes)
        # with Pool() as pool:
        #     objAnnot=partial(multiAnnotations, centre=centre, img_dim=img_dim,classes=classes,sn_id=sn_id) # prod_x has only one argument x (y is fixed to 10)
        #     objs = pool.map(objAnnot, annos)
        # while {} in objs:
        #     print("popping")
        #     objs.pop(objs.index({}))
            # objs = pool.map(sqrt, numbers)
        for anno in annos:
            if 'points_x' in anno.keys():
                px = anno["points_x"]
                py = anno["points_y"]
                # tmp_ob = copy.deepcopy(anno)
                # # input(tmp_ob)
                # # tmp_ob['coordinates'] = [newGPS[1], newGPS[0]]
                # tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],img_dim)
                # # drawMapPixels([],filename,[anno])
                # if tmp_pixs:
                #     px = tmp_pixs["points_x"]
                #     py = tmp_pixs["points_y"]
            else:
                # generate points x and y
                # new_scenario = []
                # for item in scenario:
                #     new_item = copy.deepcopy(item)
                # print(height, width)
                # print(sn_id, )
                # input(img_dim)
                newGPS = getPointLatLng(anno['pixels_bb'][0]+10, anno['pixels_bb'][1]+10,CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id],height,width)
                # print(newGPS)
                tmp_ob = copy.deepcopy(anno)
                tmp_ob['coordinates'] = [newGPS[1], newGPS[0]]
                tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],img_dim)
                if tmp_pixs:
                    px = tmp_pixs["points_x"]
                    py = tmp_pixs["points_y"]
                else:
                    print(anno['name'])
                    # drawItem([anno['name']],filename,pixels_bb=[anno['pixels_bb']])
                    # input("??")
                    continue
            try:
                categ_ = classes.index(anno['keywords'].split(' ')[0])
            except:
                categ_ = 0
            poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            assert np.min(px) > 0 and  np.min(py) > 0
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": categ_,
                "iscrowd":0,
                "name":anno['name']

            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def get_blocks_dicts(img_dir,type):
    # blocks
    classes_id = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','blank']
    listOfFiles = os.listdir(img_dir)
    pattern = f"{type}*"
    imgs_anns = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            print(entry)
            json_file = os.path.join(img_dir, entry)
            with open(json_file) as f:
                imgs_anns += json.load(f)
    # print(imgs_anns)
    # input(len(imgs_anns))
    dataset_dicts = []
    id_dicts = {}
    for k, example in enumerate(imgs_anns):
        record = {}
        try:
            filename = os.path.join(img_dir, example["image_path"])

            height, width = cv2.imread(filename).shape[:2]
        except:
            continue
        record["file_name"] = filename
        record["image_id"] = k
        record["height"] = height
        record["width"] = width

        # annos = v["regions"]
        objs = []
        # input(example)
        for k, anno in enumerate(ast.literal_eval(example['bounding_boxes'])):
            # print(anno)
            # assert not anno["region_attributes"]
            # anno = anno["shape_attributes"]
            # if anno['category_id'] in id_dicts.keys():
            #     id_dicts[anno['category_id']] += 1
            # else:
            #     id_dicts[anno['category_id']] = 1
            name = 'blank'
            if example['decoration'] == 'digit':
                name = str(k+1)
            # input(name)
            # px = anno["points_x"]
            # py = anno["points_y"]
            # poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            # poly = [p for x in poly for p in x]
            # assert np.min(px) > 0 and  np.min(py) > 0
            # input(anno)
            obj = {
                "bbox": [anno[3], anno[0], anno[1], anno[2]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes_id.index(name),
                "name":name

            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        # input(dataset_dicts)
        # with open('ROSMI/statistics/classes_statistics.json', 'w') as outfile:
        #     json.dump(id_dicts, outfile)
    return dataset_dicts


def get_osm_dicts(img_dir):
    json_file = os.path.join(img_dir, "OSMsegm_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    id_dicts = {}
    for k, v in imgs_anns.items():
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = k
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for a_key, anno in annos.items():
            # assert not anno["region_attributes"]
            # anno = anno["shape_attributes"]
            if anno['category_id'] in id_dicts.keys():
                id_dicts[anno['category_id']] += 1
            else:
                id_dicts[anno['category_id']] = 1

            px = anno["points_x"]
            py = anno["points_y"]
            poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            assert np.min(px) > 0 and  np.min(py) > 0
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": anno['category_id'],
                "iscrowd":0,
                "name":anno['name']

            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        with open('ROSMI/statistics/classes_statistics.json', 'w') as outfile:
            json.dump(id_dicts, outfile)
    return dataset_dicts



def readOSM(js_path,categs,category_ids):
    osm_map = []
    # categories = ["aerialway", "aeroway", "amenity", "barrier", "boundary", "building", "craft", "emergency", "geological", "highway", "historic", "landuse", "leisure", "man_made", "military", "natural", "office", "place", "power", "public_transport", "railway", "route", "shop", "sport", "telecom", "tourism", "waterway"]
    with open(js_path) as json_file:
        san_diego = json.load(json_file)

    for f in san_diego['features']:

        # input(f)
        tmp_ = {}
        try:
            if 'id' in f.keys():
                tmp_['id'] = f['id'].split('/')[1]
            else:
                tmp_['id'] = str(random.randint(0,50000000))

        except:
            tmp_['id'] = str(random.randint(0,50000000))

        tmp_['keywords'] = ""
        if f['geometry']['type'] == 'MultiLineString' or f['geometry']['type'] == 'MultiPolygon':
            continue

        class_ = [x for x in categs if x in list(f['properties'].keys())]
        if len(class_) >= 1:
            class_ = class_[0]
            if f['properties'][class_] not in category_ids:
                category_ids.append(f['properties'][class_])
            # print(class_)
            # input(f['properties'][class_])

        else:
            # print("no category:")
            # print(f['properties'])
            continue

        tmp_['category_id'] = category_ids.index(f['properties'][class_])
        try:
            tmp_['name'] = f['properties']['name']
        except:
            continue
        try:

            tmp_['keywords'] += f['properties'][class_]
            tmp_['keywords'] += ' '+class_
            tmp_['keywords'] += ' '+f['properties']['type']
        except:
            pass
        try:
            tmp_['keywords'] += ' '+f['properties'][tmp_['p_type']]
        except:
            pass
        try:
            tmp_['keywords'] = ' '+f['properties']['gnis:feature_type']
        except:
            pass
        # if len(tmp_['keywords'].split(' ')) < 3:
        #     print(f['properties'])
        #     print(tmp_['keywords'])
        tmp_['g_type'] = f['geometry']['type']
        tmp_['coordinates'] = f['geometry']['coordinates']
        osm_map.append(tmp_)
        # input(tmp_)

    with open('classes.json','w') as cl:
        json.dump(category_ids,cl)
    print("Found {} classes!".format(len(category_ids)))

    return osm_map



def loadROSMI():
    with open('/home/marios/experiments/gps_prediction/attend_landmarks/OSPAM_dataset/ROSMI_dataset/train.json') as f:
        train_ = json.load(train_, f)
    with open('/home/marios/experiments/gps_prediction/attend_landmarks/OSPAM_dataset/ROSMI_dataset/val.json') as f:
        val_ = json.load(val_, f)

    # with open('/home/marios/experiments/gps_prediction/attend_landmarks/OSPAM_dataset/ROSMI_dataset/test.json', 'w') as f:
    #     json.dump(test_, f)
    # with open('ROSMI_dataset/ROSMI.json') as json_file:
    #     scenarios = json.load(json_file)


    return rosmi
def gen_chunks(reader, chunksize=100):
    """
    Chunk generator. Take a CSV `reader` and yield
    `chunksize` sized slices.
    """
    chunk = []
    for index, line in enumerate(tqdm(reader)):
        if (index % chunksize == 0 and index > 0):
            yield chunk
            del chunk[:]
        chunk.append(line)
    yield chunk

def load_det_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, FIELDITEMS, delimiter="\t")
        for it in gen_chunks(reader,  chunksize=1000):
            # print(len(item[0]))
            # input(len(item))
            for i, item in enumerate(it):
                for key in ['img_h', 'img_w', 'num_boxes']:
                    item[key] = int(item[key])

                boxes = item['num_boxes']
                decode_config = [
                    ('boxes', (boxes, 4), np.float64),
                    ('features', (boxes, -1), np.float64),
                    ('names', (boxes, -1), np.dtype('<U100'))
                ]
                for key, shape, dtype in decode_config:
                    # print(key)
                    # print(item[key])
                    item[key] = np.frombuffer(base64.b64decode(ast.literal_eval(item[key])), dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    item[key].setflags(write=False)

                data.append(item)
                if topk is not None and len(data) == topk:
                    break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def draw_and_save(name,xy,size,type):
    mask = Image.new('L', size, color = 0)
    draw=ImageDraw.Draw(mask)
    highlighted_area = xy
    if type == 'Point':
        draw.rectangle(highlighted_area, fill = 255)
    elif type == 'LineString':
        draw.line(highlighted_area,fill=255)
    else:
        draw.polygon(highlighted_area,fill=255)
    mask.save('OSPAM_masks/'+name+'.png')

# can draw an item or list of items with names
def drawItem(name,image_path,points=None,pixels_bb=None):

    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(e)
        return

    img_dim = [img.getbbox()[2],img.getbbox()[3]]

    if type(name) != list:
        name = [name]
        if pixels_bb is not None and type(pixels_bb) != list:
            pixels_bb = [pixels_bb]
        if points is not None and type(points) != list:
            points = [points]

    draw = ImageDraw.Draw(img)
    for id in range(len(name)):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        if points:
            draw.text(tuple((points[id]['points_x'][0],points[id]['points_y'][0])),name[id],fill=color,font=font)
            draw.line([tuple((x,y)) for x,y in zip(points[id]['points_x'],points[id]['points_y'])],fill=color)
        if pixels_bb:
            # input(static)
            draw.text(tuple((pixels_bb[id][0],pixels_bb[id][1])),name[id],fill=color,font=font)
            draw.rectangle(pixels_bb[id],outline=color)
        else:
            print("No pixels given. Give points x and y [points] or bounding box [pixels_bb]")
    img.show()




def drawMapPixels(scn,image_path,annots=None,cls=None):

    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(e)
        return

    img_dim = [img.getbbox()[2],img.getbbox()[3]]
    if not annots:
        if type(scn)==dict:
            scenes = scn.values()
        else:
            scenes = scn

        for scene in scenes:
            input(scene)
            # for static in scene:
            draw = ImageDraw.Draw(img)
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            # input(static)
            draw.text(tuple((scene['points_x'][0],scene['points_y'][0])),scene['name'],fill=color,font=font)
            draw.line([tuple((x,y)) for x,y in zip(scene['points_x'],scene['points_y'])],fill=color)
    else:
        for obj in annots:
            # for static in scene:
            draw = ImageDraw.Draw(img)
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            # input(static)
            draw.text(tuple((obj['bbox'][0],obj['bbox'][1])),cls[obj['category_id']],fill=color,font=font)
            draw.rectangle(obj['bbox'],outline=color)
        # draw = ImageDraw.Draw(img)
    img.show()
    # img.save('test.png')



def drawMapItems(scenes,centres_,zooms_,image_path):


    sent_id = -1
    for s_key,scene in scenes.items():
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", zooms_[s_key])

        # input(len(scene))

        try:
            img = Image.open(os.path.join(image_path,str(s_key)+'.png'))
        except Exception as e:
            print(e)
            continue

        img_dim = [img.getbbox()[2],img.getbbox()[3]]
        # print(img_dim)
        # # get the index of the centre of this scenario
        # id_of_scenario = centres_.index(output['prev_output']['scenario'][0])
        # data_iter['scenario_items'] = 'scenario{}.json'.format(id_of_scenario)
        #
        #
        # testing scenarios by drawing all items TEMPORARY
        centre = calculateTiles(centres_[s_key],zooms_[s_key])
        for static in scene:
            draw = ImageDraw.Draw(img)
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            # input(static)
            if static['g_type'] == 'Point':
                # input("inside point")
                pix_center = convertGeoToPixel(centre,[static['coordinates'][1],static['coordinates'][0]],zooms_[s_key],img_dim)
                draw.text(pix_center,static['name'],fill=color,font=font)
                # print(static['name'])

                # print(pix_center)
                # dist_ = haversine(c[1],c[0],map['coordinates'][0],map['coordinates'][1])
            elif static['g_type'] == 'LineString':
                pix_center = []
                for pnt in static['coordinates']:
                    # input(pnt)
                    pix_center.append(tuple(convertGeoToPixel(centre,[pnt[1],pnt[0]],zooms_[s_key],img_dim)))
                draw.line(pix_center,fill=color)
                # print(static['name'])
                # dist_ = haversine(c[1],c[0],map['coordinates'][0][0],map['coordinates'][0][1])
            else:
                pix_center = []
                for side in static['coordinates']:
                    for pnt in side:
                        pix_center.append(tuple(convertGeoToPixel(centre,[pnt[1],pnt[0]],zooms_[s_key],img_dim)))

                        # pix_center = convertGeoToPixel(centre,[pnt[1],pnt[0]],output['prev_output']['scenario'][1],img_dim)
                draw.polygon(pix_center,fill=color)

            draw = ImageDraw.Draw(img)
        img.save('scenario{}.png'.format(s_key))
        # input("Scenario: {}".format(s_key))

        draw = ImageDraw.Draw(img)


# returns dictionary with the name 'points_x' and 'points_y'
def generatePixel(scene_object,centre,zoom,img_dim, size=10):
    pix_pos = {}
    # depending on the geometry type generate pixels from the
    # raw GPS and save them. These are used for prediction and
    # for the generation of the attention masks
    if scene_object['g_type'] == 'Point':
        # print(img_dim)
        # print(zoom)
        # print(centre)
        # print([scene_object['coordinates'][1],scene_object['coordinates'][0]])
        tmppx = convertGeoToPixel(centre,[scene_object['coordinates'][1],scene_object['coordinates'][0]],zoom,img_dim)
        # input(tmppx)
        if tmppx[0] < 0 or tmppx[0] >= img_dim[0]  or tmppx[1] < 0 or tmppx[1] >= img_dim[1]:
            return None
        x_,y_ = midPointCircleDraw(tmppx[0],tmppx[1],size,img_dim)
        pix_pos['points_x'] = x_
        pix_pos['points_y'] = y_
        # tmp_land['points_x'] = tmppx[0]
        # tmp_land['points_y'] = tmppx[1]
        # tmp_land['all_att_pixels'] = [tmppx[0]-10,tmppx[1]-10,tmppx[0]+10,tmppx[1]+10]
    elif scene_object['g_type'] == 'LineString':
        tmppx = []
        for pnt in scene_object['coordinates']:
            tmp_px = convertGeoToPixel(centre,[pnt[1],pnt[0]],zoom,img_dim)

            if tmp_px[0] < 0 or tmp_px[0] >= img_dim[0]  or tmp_px[1] < 0 or tmp_px[1] >= img_dim[1]:
                continue
            tmppx.append(tmp_px)

        mask = Image.new('L', img_dim, color = 0)
        mask_draw=ImageDraw.Draw(mask)
        # highlighted_area = xy
        # if type == 'Point':
        new_tmppx = []
        if len(tmppx) >=2:
            mask_draw.line([(x_[0],x_[1]) for x_ in tmppx], fill = 255)
            width, height = mask.size
            pix2 = mask.load()
            for x in range(1,width):
                for y in range(1,height):
                    if pix2[x,y] == 255:
                        # input(pix2[x,y])
                        new_tmppx.append([float(x),float(y)])
            # mask.show()
        # input("??")
        pix_pos['points_x'] = [i[0] for i in new_tmppx]
        pix_pos['points_y'] = [i[1] for i in new_tmppx]

        # pix_pos['points_x'] = [i[0] for i in tmppx]
        # pix_pos['points_y'] = [i[1] for i in tmppx]
        # tmp_land['all_att_pixels'] = tmppx
        # draw.line(tmppx,fill=color)
        # dist_ = haversine(c[1],c[0],map['coordinates'][0][0],map['coordinates'][0][1])
    else:
        tmppx = []
        for side in scene_object['coordinates']:
            for pnt in side:
                tmp_px = convertGeoToPixel(centre,[pnt[1],pnt[0]],zoom,img_dim)
                # if tmp_px[0] < 0:
                #     tmp_px[0] = 0
                # if tmp_px[0] > img_dim[0]:
                #     tmp_px[0] = img_dim[0]
                # if tmp_px[1] < 0:
                #     tmp_px[1] = 0
                # if tmp_px[1] > img_dim[1]:
                #     tmp_px[1] = img_dim[1]
                if tmp_px[0] < 0 or tmp_px[0] >= img_dim[0]  or tmp_px[1] < 0 or tmp_px[1] >= img_dim[1]:
                    # print(tmp_px)
                    continue
                tmppx.append(tmp_px)
        mask = Image.new('L', img_dim, color = 0)
        mask_draw=ImageDraw.Draw(mask)
        # highlighted_area = xy
        # if type == 'Point':
        new_tmppx = []
        if len(tmppx) >=2:
            mask_draw.polygon([(x_[0],x_[1]) for x_ in tmppx], fill = 255)
            width, height = mask.size
            pix2 = mask.load()
            for x in range(1,width):
                for y in range(1,height):
                    if pix2[x,y] == 255:
                        # input(pix2[x,y])
                        new_tmppx.append([float(x),float(y)])
            # mask.show()
        # input("??")
        pix_pos['points_x'] = [i[0] for i in new_tmppx]
        pix_pos['points_y'] = [i[1] for i in new_tmppx]

    if len(pix_pos['points_x']) < 6 or len(pix_pos['points_y']) < 6:
        # print("less than 5 points")
        # print(pix_pos)
        # print(scene_object['name'])
        return None
    # tmp_regions[tmposm['category_id']] = tmp_land
    return pix_pos


def generateRandLatLng(center):
    offet1 = 0.0003;



    if np.random.random() < 0.2:
       pros1 = 1;
       pros2 = -1;
    elif np.random.random() < 0.45:
       pros1 = -1;
       pros2 = 1;
    elif np.random.random() < 0.75:
       pros1 = -1;
       pros2 = -1;
    else:
       pros1 = 1;
       pros2 = 1;

    return [pros1*(np.random.random()*offet1+offet1/2)+center[0],pros2*(np.random.random()*offet1+offet1/2)+center[1]]


# Implementing Mid-PoCircle Drawing
# Algorithm
def midPointCircleDraw(x_centre,y_centre, r,im_dim):
    x_points = []
    y_points = []
    x = r
    y = 0

    # Printing the initial poon the
    # axes after translation
    # print("(", x + x_centre, ", ",
    #            y + y_centre, ")",
    #            sep = "", end = "")
    if (x + x_centre) > 0 and (x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 0 and (y + y_centre) < (im_dim[1]-1):
        x_points.append(x + x_centre)
        y_points.append(y + y_centre)
    # When radius is zero only a single
    # powill be printed
    if (r > 0) :
        # print("(", x + x_centre, ", ",
        #           -y + y_centre, ")",
        #           sep = "", end = "")
        if (x + x_centre) > 0 and (x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 0 and (-y + y_centre) < (im_dim[1]-1):
            x_points.append(x + x_centre)
            y_points.append(-y + y_centre)

        if (y + x_centre) > 0 and (y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 0 and (x + y_centre) < (im_dim[1]-1):
            x_points.append(y + x_centre)
            y_points.append(x + y_centre)

        if (-y + x_centre) > 0 and (-y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 0 and (x + y_centre) < (im_dim[1]-1):
            x_points.append(-y + x_centre)
            y_points.append(x + y_centre)
        #
        # print("(", -y + x_centre, ", ",
        #             x + y_centre, ")", sep = "")
    # Initialising the value of P
    P = 1 - r
    while (x > y) :

        y += 1

        # Mid-pois inside or on the
        # perimeter
        if (P <= 0):
            P = P + 2 * y + 1

        # Mid-pois outside the perimeter
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1

        # All the perimeter points have
        # already been printed
        if (x < y):
            break

        # Printing the generated poand its reflection
        # in the other octants after translation

        if (x + x_centre) > 1 and (x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 1 and (y + y_centre) < (im_dim[1]-1):
            x_points.append(x + x_centre)
            y_points.append(y + y_centre)
        # print("(", x + x_centre, ", ", y + y_centre,
        #                     ")", sep = "", end = "")

        if (-x + x_centre) > 1 and (-x + x_centre) < (im_dim[0]-1)  and (y + y_centre) > 1 and (y + y_centre) < (im_dim[1]- 1):
            x_points.append(-x + x_centre)
            y_points.append(y + y_centre)
        # print("(", -x + x_centre, ", ", y + y_centre,
        #                      ")", sep = "", end = "")

        if (x + x_centre) > 1 and (x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 1 and (-y + y_centre) < (im_dim[1]- 1):
            x_points.append(x + x_centre)
            y_points.append(-y + y_centre)
        # print("(", x + x_centre, ", ", -y + y_centre,
        #                      ")", sep = "", end = "")
        # print("(", -x + x_centre, ", ", -y + y_centre,
        #                                 ")", sep = "")

        if (-x + x_centre) > 1 and (-x + x_centre) < (im_dim[0]-1)  and (-y + y_centre) > 1 and (-y + y_centre) < (im_dim[1]- 1):
            x_points.append(-x + x_centre)
            y_points.append(-y + y_centre)
        # If the generated pois on the line x = y then
        # the perimeter points have already been printed
        if (x != y) :

            if (y + x_centre) > 1 and (y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 1 and (x + y_centre) < (im_dim[1]- 1):
                x_points.append(y + x_centre)
                y_points.append(x + y_centre)
            # print("(", y + x_centre, ", ", x + y_centre,
            #                     ")", sep = "", end = "")

            if (-y + x_centre) > 1 and (-y + x_centre) < (im_dim[0]-1)  and (x + y_centre) > 1 and (x + y_centre) < (im_dim[1]- 1):
                x_points.append(-y + x_centre)
                y_points.append(x + y_centre)
            # print("(", -y + x_centre, ", ", x + y_centre,
            #                      ")", sep = "", end = "")

            if (y + x_centre) > 1 and (y + x_centre) < (im_dim[0]-1)  and (-x + y_centre) > 1 and (-x + y_centre) < (im_dim[1]- 1):
                x_points.append(y + x_centre)
                y_points.append(-x + y_centre)
            # print("(", y + x_centre, ", ", -x + y_centre,
            #                      ")", sep = "", end = "")

            if (-y + x_centre) > 1 and (-y + x_centre) < (im_dim[0]-1)  and (-x + y_centre) > 1 and (-x + y_centre) < (im_dim[1]- 1):
                x_points.append(-y + x_centre)
                y_points.append(-x + y_centre)
            # print("(", -y + x_centre, ", ", -x + y_centre,
            #                                 ")", sep = "")
    return x_points,y_points


# -122.44, 34.44 etc.
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def destination(point, distance, bearing):

    # print(point)
    lon1,lat1 = (radians(float(coord)) for coord in point)

    # print(distance," ",bearing)
    radians_bearing = radians(float(bearing))
    # print(radians_bearing)

    delta = float(distance) / EARTH_MEAN_RADIUS
    lat2 = asin(
        sin(lat1)*cos(delta) +
        cos(lat1)*sin(delta)*cos(radians_bearing)
    )
    numerator = sin(radians_bearing) * sin(delta) * cos(lat1)
    denominator = cos(delta) - sin(lat1) * sin(lat2)

    lon2 = lon1 + atan2(numerator, denominator)
    # print(type(lon2))
    lon2_deg = (degrees(lon2) + 540) % 360 - 180
    lat2_deg = degrees(lat2)

    return [lon2_deg,lat2_deg]

def getPointLatLng(x, y,centre_lon,centre_lat,_zoom,height,width):
    parallelMultiplier = cos(centre_lat * pi / 180)
    degreesPerPixelX = 360 / pow(2, _zoom + 8)
    degreesPerPixelY = 360 / pow(2, _zoom + 8) * parallelMultiplier
    pointLat = centre_lat - degreesPerPixelY * ( y - height / 2)
    pointLng = centre_lon + degreesPerPixelX * ( x  - width / 2)

    return (pointLat, pointLng)
# Calculating tile needed for converting GPS to pixels
# expects latlon like [37.0000, -122.2222]
def calculateTiles(latlon,zoom):

    lon_rad = radians(latlon[1]);
    lat_rad = radians(latlon[0]);
    n = pow(2.0, zoom);

    tileX = ((latlon[1] + 180) / 360) * n;
    tileY = (1 - (log(tan(lat_rad) + 1.0/cos(lat_rad)) / pi)) * n / 2.0;
    # print(" X: {}, Y: {}".format(tileX,tileY))
    return [tileX,tileY]

# Convert GPS to pixels. Need center of the map in GPS, lat/lon GPS, zoom level,
# dimension of the image.
def convertGeoToPixel(centre_, latlon, zoom, imgDim, adjust=False):
    # 	mapWidth = imgDim[0];
    # 	mapHeight = imgDim[1]
    #
    # double lon = lon_centre
    # double lat = lat_centre
    # double zoom = 6; # 6.5156731549786215 would be possible too
    if len(centre_) == 0:
        # print("New centre")
        centre_ = calculateTiles(latlon,zoom)
    point = calculateTiles(latlon,zoom)
    # print(centre_[0] - point[0])
    # print(imgDim)
    # print(zoom)
    if adjust:
        pix_x = imgDim[0]/2.13 - (centre_[0] - point[0])*256
        pix_y = imgDim[1]/2.5 - (centre_[1] - point[1])*256
    else:
        pix_x = imgDim[0]/2 - (centre_[0] - point[0])*256
        pix_y = imgDim[1]/2 - (centre_[1] - point[1])*256

    return [pix_x,pix_y]




def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            # gt_boxes_img = gt_boxes[img_id]
            gt_boxes_img = gt_boxes[img_id]['boxes']
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}


def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax


###############################################################################
######### helper functions for getting natural language statistics ############
###############################################################################

def safe_divide(numerator, denominator):
    if denominator == 0 or denominator == 0.0:
        index = 0
    else: index = numerator/denominator
    return index

# Simple TTR
def ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes,ntokens)

# Root TTR
def root_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes, sqrt(ntokens))

# Log TTR
def log_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(log10(ntypes), log10(ntokens))


# Mass TTR
def maas_ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide((log10(ntokens)-log10(ntypes)), pow(log10(ntokens),2))

# Moving average TTR (MATTR)
# By default, the window size is 50 words. However, this can be customized using the window_length argument.
def mattr(text, window_length = 50):

    if len(text) < (window_length + 1):
        ma_ttr = safe_divide(len(set(text)),len(text))

    else:
        sum_ttr = 0
        denom = 0
        for x in range(len(text)):
            small_text = text[x:(x + window_length)]
            if len(small_text) < window_length:
                break
            denom += 1
            sum_ttr+= safe_divide(len(set(small_text)),float(window_length))
        ma_ttr = safe_divide(sum_ttr,denom)

    return ma_ttr



# Mean segmental TTR (MSTTR)
# By default, the segment size is 50 words. However, this can be customized using the window_length argument.
def msttr(text, window_length = 50):

    if len(text) < (window_length + 1):
        ms_ttr = safe_divide(len(set(text)),len(text))

    else:
        sum_ttr = 0
        denom = 0

        n_segments = int(safe_divide(len(text),window_length))
        seed = 0
        for x in range(n_segments):
            sub_text = text[seed:seed+window_length]
            #print sub_text
            sum_ttr += safe_divide(len(set(sub_text)), len(sub_text))
            denom+=1
            seed+=window_length

        ms_ttr = safe_divide(sum_ttr, denom)

    return ms_ttr


# Hypergeometric distribution D (HDD)
# A more straightforward and reliable implementation of vocD (Malvern, Richards, Chipere, & Duran, 2004) as per McCarthy and Jarvis (2007, 2010).
def hdd(text):
    #requires Counter import
    def choose(n, k): #calculate binomial
        """
        A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1): #this was changed to "range" from "xrange" for py3
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    def hyper(successes, sample_size, population_size, freq): #calculate hypergeometric distribution
        #probability a word will occur at least once in a sample of a particular size
        try:
            prob_1 = 1.0 - (float((choose(freq, successes) * choose((population_size - freq),(sample_size - successes)))) / float(choose(population_size, sample_size)))
            prob_1 = prob_1 * (1/sample_size)
        except ZeroDivisionError:
            prob_1 = 0

        return prob_1

    prob_sum = 0.0
    ntokens = len(text)
    types_list = list(set(text))
    frequency_dict = Counter(text)

    for items in types_list:
        prob = hyper(0,42,ntokens,frequency_dict[items]) #random sample is 42 items in length
        prob_sum += prob

    return prob_sum

# Measure of lexical textual diversity (MTLD)
# Calculates MTLD based on McCarthy and Jarvis (2010).
def mtld(input, min = 10): #original MTLD described in Jarvis & McCarthy
    def mtlder(text):
        factor = 0
        factor_lengths = 0
        start = 0
        for x in range(len(text)):
            factor_text = text[start:x+1]
            if x+1 == len(text):
                factor += safe_divide((1 - ttr(factor_text)),(1 - .72))
                factor_lengths += len(factor_text)
            else:
                if ttr(factor_text) < .720 and len(factor_text) >= min:
                    factor += 1
                    factor_lengths += len(factor_text)
                    start = x+1
                else:
                    continue

        mtld = safe_divide(factor_lengths,factor)
        return mtld
    input_reversed = list(reversed(input))
    mtld_full = safe_divide((mtlder(input)+mtlder(input_reversed)),2)
    return mtld_full
