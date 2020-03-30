import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import shutil
from tqdm import tqdm

def crop_data(viz=False, limit=False, save_img=False, write_json=False):
    
    # defined interest classes
    interest_classes = ['FFC', 'FFS', 'hemangioma', 'HCC', 'cyst']
    
    # duplicate original file
    shutil.copy('/home/liverusg/PycharmProjects/AI_USG_liver/experiments/retinanet_mix/annotations/cleaned_sampled_seed4.json','/home/liverusg/PycharmProjects/AI_USG_liver/experiments/retinanet_mix/annotations/cropped_sampled_seed4.json')

    # Load json file
    json_data = 'annotations/cropped_sampled_seed4.json'
    json_object = json.load(open(json_data, "r+", encoding='utf-8'))

    # Start cropped image => Plot bbox => save bounding box into new json files
    img_output_path = 'cropped_data/cropped_images/'
    anno_output_path = 'cropped_data/cropped_annotation/'
    
    # create limit sample
    c = 0
    
    if write_json:
        new_json = open(anno_output_path+'cropped_annotation.json', 'w')
    
    for k,v in tqdm(json_object.items()):
        if viz:
            fig = plt.figure(figsize=(20,10))
        
        file_path  = v['jpg_img_path']
        data_type = v['set'].split("_")[0].strip()
        img = cv2.imread(file_path, 0)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get window information
        left_w = int(v['window_min_x'])
        top_w = int(v['window_min_y'])
        right_w = int(v['window_max_x'])
        bot_w = int(v['window_max_y'])

        # Crop image
        crop_img = img[top_w:bot_w, left_w:right_w].copy()
        # Get new image shape
        height, width = crop_img.shape
        
        if write_json:
            v['jpg_img_path'] = img_output_path + data_type+"/"+k
        
        # Save cropped image according to the set of data
        if save_img:
            cv2.imwrite(img_output_path +data_type+"/"+ k, crop_img)

        # Get all regions of the image
        regions = v['regions']

        for l in regions:
            if 'structure' in l['region_attributes'].keys():
                if l['region_attributes']['structure'] in interest_classes:
                    lesion_type = l['region_attributes']['structure']
                    bbox = l['bbox']

                    if viz:
                        # draw box and text
                        cv2.rectangle(img, (bbox['x0'], bbox['y0']), (bbox['x1'], bbox['y1']), (0,0,255), 2) 
                        cv2.putText(img, lesion_type, (bbox['x0'],bbox['y0'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) 
                        cv2.putText(img, lesion_type, (bbox['x0'],bbox['y0'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 

            elif 'markings' in l['region_attributes'].keys():
                mark = l['region_attributes']['markings']
                mbbox = l['bbox']
                
                if viz:
                    # draw box and text
                    cv2.rectangle(img, (mbbox['x0'], mbbox['y0']), (mbbox['x1'], mbbox['y1']), (0,255,0), 2) 
                    cv2.putText(img, mark, (mbbox['x0'], mbbox['y0'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) 
                    cv2.putText(img, mark, (mbbox['x0'], mbbox['y0'] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
        
        if viz:
            plt.imshow(img, cmap='gray')
            plt.show()

        #################################################################################################################################

        # Begin crop the image
        if viz:
            fig = plt.figure(figsize=(20,10))

        for l in regions:
            if 'structure' in l['region_attributes'].keys():
                if l['region_attributes']['structure'] in interest_classes:
                    lesion_type = l['region_attributes']['structure']
                    bbox = l['bbox']

                    # draw new_bbox and text
                    new_x0 = bbox['x0'] - left_w
                    new_y0 = bbox['y0'] - top_w
                    new_x1 = bbox['x1'] - left_w
                    new_y1 = bbox['y1'] - top_w

                    if new_x0 < 0 or new_x0 > width or new_x1 < 0 or new_x1 > width:
                        print("Found lesion out of frame (x-value)", k)
                    elif new_y0 < 0 or new_y0 > height or new_y1 < 0 or new_y1 > height:
                        print("Found lesion out of frame (y-value)", k)

                    if write_json:
                        # overwrite lesion bbox
                        l['bbox']['x0'] = new_x0
                        l['bbox']['y0'] = new_y0
                        l['bbox']['x1'] = new_x1
                        l['bbox']['y1'] = new_y1

                    if viz:
                        cv2.rectangle(crop_img, (new_x0, new_y0), (new_x1, new_y1), (255,0,0), 2) 
                        cv2.putText(crop_img, lesion_type, (new_x0, new_y0 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) 
                        #cv2.putText(crop_img, lesion_type, (new_x0, new_y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 
                else:
                    del l

            elif 'markings' in l['region_attributes'].keys():
                mark = l['region_attributes']['markings']
                mbbox = l['bbox']

                # draw new_mbbox and text
                new_m_x0 = mbbox['x0'] - left_w
                new_m_y0 = mbbox['y0'] - top_w
                new_m_x1 = mbbox['x1'] - left_w
                new_m_y1 = mbbox['y1'] - top_w
                
                if new_m_x0 < 0 or new_m_x0 > width or new_m_x1 < 0 or new_m_x1 > width:
                    print("Found mark out of frame (x-value)", k)
                elif new_m_y0 < 0 or new_m_y0 > height or new_m_y1 < 0 or new_m_y1 > height:
                    print("Found mark out of frame (y-value)", k)
                
                if write_json:
                    # overwrite marking box
                    l['bbox']['x0'] = new_m_x0
                    l['bbox']['y0'] = new_m_y0
                    l['bbox']['x1'] = new_m_x1
                    l['bbox']['y1'] = new_m_y1
                
                if viz:
                    cv2.rectangle(crop_img, (new_m_x0, new_m_y0), (new_m_x1, new_m_y1), (0,255,255), 2) 
                    cv2.putText(crop_img, mark, (new_m_x0, new_m_y0 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2) 
                    #cv2.putText(crop_img, mark, (new_m_x0, new_m_y0 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) 

        if viz:
            plt.imshow(crop_img, cmap='gray')
            plt.show()
            
        # count file    
        #c+= 1
            
        #print("Finished",c,"files")
        
        if limit:
            if c > 10:
                break
                
    if write_json:
        json.dump(json_object, new_json)
                
                
crop_data(viz=False, limit=False, save_img=False, write_json=True)