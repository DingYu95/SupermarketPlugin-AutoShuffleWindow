import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from unrealcv.util import read_npy, read_png
from unrealcv import client


while not client.isconnected():
    client.connect()
    if client.isconnected():
        print('Connected to Unreal Engine4')
        break
    

img_file_id = '0'
output_folder = 'C:/Code/'
camera_id = '0'

target_name_list = ['Chair']
actor_name_list = client.request('vget /objects').encode().split()

# Store rgba tuple as value
actor_color_dict = {}
target_actor_dict = {}
target_rect_dict = {}

for actor_name in actor_name_list:
    tmp_color = client.request('vget /object/'+ actor_name + '/color').encode()[1:-1].split(',')
    rgba = tuple([int(color[2:]) for color in tmp_color])
    actor_color_dict[actor_name] = rgba
    for target_name in target_name_list:
        if actor_name.startswith(target_name):
            if target_name not in target_actor_dict:
                target_actor_dict[target_name] = [actor_name]
            else:
                target_actor_dict[target_name].append(actor_name)
            break

# save to png file
render_file_path = output_folder + 'render' + img_file_id + '.png'
mask_file_path = output_folder + 'mask' + img_file_id + '.png'
client.request('vget /camera/' + camera_id + '/lit ' + render_file_path)
mask_res = client.request('vget /camera/' + camera_id + '/object_mask png')
mask_array = read_png(mask_res)
cv2.imwrite(mask_file_path, mask_array)


for target in target_actor_dict:
    target_rect_dict[target] = []
    actor_list = target_actor_dict[target]
    for actor_name in actor_list:
        current_target = np.all((mask_array == actor_color_dict[actor_name]), axis=-1).astype(np.uint8)
        _, contours, _ = cv2.findContours(current_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # top_left x, y, width, height
        rect_info = cv2.boundingRect(contours[0])
        target_rect_dict[target].append(rect_info)

full_info_dict = {}
for target in target_actor_dict:
    full_info_dict[target] = {'render_img_path': render_file_path,
                              'mask_img_path': mask_file_path,
                              'object_name': target, 
                              'bbox': target_rect_dict[target],
                              'instance_name_list': target_actor_dict[target], 
                              'instance_num': len(target_actor_dict[target])}


with open(output_folder + 'label_info.json', 'w') as f:
    json.dump(full_info_dict, f, indent=4)
