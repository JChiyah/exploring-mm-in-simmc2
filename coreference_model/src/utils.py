# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from urllib.request import urlopen

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
# FIELDITEMS = ["img_id", "img_h", "img_w", "num_boxes","t_num_boxes", "boxes", "features","t_boxes","box_order", 't_simmc2_obj_indexes', 'category', 't_category', 'category_scores']
FIELDITEMS = ["img_id", "img_h", "img_w", "num_boxes","t_num_boxes", "boxes", "features","names","t_boxes","t_names","box_order"]\
             + ['t_simmc2_obj_indexes', 'category', 't_category', 'category_scores']


def load_obj_tsv(fname, topk=None, hide_images=False):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    is_detectron_output = 'detectron' in fname
    print("Start to load Faster-RCNN detected objects from %s (from detectron? %s)" % (fname, is_detectron_output))
    with open(fname) as f:
        fields = FIELDITEMS if is_detectron_output else FIELDNAMES
        reader = csv.DictReader(f, fieldnames=fields, delimiter="\t")
        for i, item in enumerate(reader):

            # print(item.keys())
            # print(item['img_id'])
            int_fields = ['img_h', 'img_w', 'num_boxes']
            if is_detectron_output:
                int_fields.append('t_num_boxes')

            for key in int_fields:
                if hide_images and key == 'num_boxes':
                    item[key] = 0
                item[key] = int(item[key])

                # print(f"{key}: {item[key]}")

            if is_detectron_output:
                boxes = item['t_num_boxes']  # if not hide_images else 0
                decode_config = [
                    # ('objects_id', (boxes, ), np.int64),
                    # ('objects_conf', (boxes, ), np.float32),
                    # ('attrs_id', (boxes, ), np.int64),
                    # ('attrs_conf', (boxes, ), np.float32),
                    ('boxes', (boxes, 4), np.float64),
                    ('t_boxes', (boxes, 4), np.float64),
                    ('features', (boxes, 2048), np.float64),
                    ('category', (boxes, ), '<U100'),
                    ('t_category', (boxes, ), '<U100'),
                    # ('category_scores', (boxes, ), np.float64),
                    ('t_simmc2_obj_indexes', (boxes, ), int)
                ]
            else:
                boxes = item['num_boxes']  # if not hide_images else 0
                decode_config = [
                    ('objects_id', (boxes, ), np.int64),
                    ('objects_conf', (boxes, ), np.float32),
                    ('attrs_id', (boxes, ), np.int64),
                    ('attrs_conf', (boxes, ), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]

            for key, shape, dtype in decode_config:
                # if key == 't_simmc2_obj_indexes':
                #     print(f"decoding {key}")
                #     print(len(base64.b64decode(_fix_encoding(item[key], is_detectron_output))))
                #     print(base64.b64decode(_fix_encoding(item[key], is_detectron_output)))
                #     print(np.frombuffer(base64.b64decode(_fix_encoding(item[key], is_detectron_output)), dtype=dtype))
                #     # print(np.frombuffer(base64.b64decode(item[key]), dtype=dtype)[0])
                #     exit(1)
                item[key]: np.ndarray = np.frombuffer(base64.b64decode(_fix_encoding(item[key], is_detectron_output)), dtype=dtype)
                # print(key)
                # print(item[key])
                item[key] = item[key].reshape(shape).copy()
                # we copy the array since otherwise it is a view from the data and cannot write it
                # if (hide_images and 't_' not in key) or key != 'features':
                # never hide the ground truth values (starting with t_)
                # item[key] = np.zeros(item[key].shape, dtype=dtype)
                # item[key].setflags(write=False)

            data.append(item)

            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    if hide_images:
        print('Images are not used (empty array given)')
    return data


def _fix_encoding(string, is_detectron_output) -> bytes:
    if is_detectron_output:
        byte_str = bytes(string, 'utf-8', 'ignore')
        byte_str = byte_str[2:len(byte_str)-1]
        return byte_str
    else:
        return string


def simmc2_draw_bboxes(bboxes: List[dict], user_utterance: str, load_path, save_path, verbose=False):
    """Draw bounding boxes given the screenpath and list of bboxes.

    Args:
        bboxes: List of all bounding boxes, in XYWH format
        load_path: Path to load the screenshot
        save_path: Path to save the screenshot with bounding boxes
        verbose: Print out status statements
    """
    # Read images and draw rectangles.
    if 'm_' in load_path:
        load_path = load_path.replace('m_', '')
    base = Image.open(load_path)
    base = base.convert("RGBA")

    # make a blank image for the transparent bbox overlays
    image = Image.new("RGBA", base.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(image)
    # Get a font.
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 23)    # ImageFont.load_default()
    offset = 2
    opacity = 0.25
    for index, bbox_datum in enumerate(bboxes):
        object_index = bbox_datum.get("name", index)
        x0, y0, width, height = bbox_datum["bbox"]
        x1, y1 = x0 + width, y0 + height

        fill = bbox_datum.get('fill', None)
        if fill:
            fill = fill + (int(255 * opacity),)

        draw.rectangle(
            [(x0, y0), (x1, y1)],
            outline=bbox_datum.get('outline', None), fill=fill
        )

        # Draw text with black background.
        text = str(object_index)
        text_width, text_height = font.getsize(text)
        draw.rectangle(
            (
                x0 + offset,
                y1 - offset,
                x0 + 2 * offset + text_width,
                y1 - 2 * offset - text_height,
            ),
            fill=bbox_datum.get('fill', (0, 0, 0)),
        )
        draw.text(
            (x0 + offset, y1 - 2 * offset - text_height),
            text,
            fill=(0, 0, 0),
            font=font,
        )

    # draw user_utterance at top of image
    text_width, text_height = font.getsize(user_utterance)
    draw.rectangle(
        (
            0,
            0,
            0 + 2 * offset + text_width,
            0 + 2 * offset + text_height,
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (0, 0),
        user_utterance,
        fill=(255, 255, 255),
        font=font,
    )

    image = Image.alpha_composite(base, image)

    # Save the image with bbox drawn.
    if verbose:
        print("Saving: {}".format(save_path))
    image.save(save_path, "PNG")


# def simmc2_get_bbox

# for some reason the following does not work with the image,
# perhaps something is wrong with the original imgs
# url = "https://i.ytimg.com/vi/W4qijIdAPZA/maxresdefault.jpg"
# file = BytesIO(urlopen(url).read())
# img = Image.open(file)
# draw = ImageDraw.Draw(img, "RGBA")
# draw.rectangle(((280, 10), (1010, 706)), fill=(200, 100, 0, 127))
# draw.rectangle(((280, 10), (1010, 706)), outline=(0, 0, 0, 127), width=3)
# img.save(save_path + 'orange-cat.png')
