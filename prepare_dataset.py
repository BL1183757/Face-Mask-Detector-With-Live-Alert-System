import os
import cv2
import xml.etree.ElementTree as ET

dataset_path="C:/Users/Administrator/Downloads/Face Mask Detection"

images_path=os.path.join(dataset_path,"images")
annotations_path=os.path.join(dataset_path,"annotations")

output_base_path='preprocesssed_dataset'

os.makedirs(output_base_path,exist_ok=True)
os.makedirs(os.path.join(output_base_path,'with_mask'),exist_ok=True)
os.makedirs(os.path.join(output_base_path,'without_mask'),exist_ok=True)

for filename in os.listdir(annotations_path):
    if filename.endswith(".xml"):
        xml_path=os.path.join(annotations_path, filename)
        image_name=filename.replace('.xml','.png')
        image_path=os.path.join(images_path,image_name)
            
        if not os.path.exists(image_path):
            print(f"Image file for {filename} not found, skipping.")
            continue
            
        tree=ET.parse(xml_path)
        root=tree.getroot()
        
        img=cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}, skipping. Image file is corrupted or unreadable.")
            continue
        
        for object_tag in root.findall('object'):
            name=object_tag.find('name').text
            bndbox=object_tag.find('bndbox')
            xmin=int(bndbox.find('xmin').text)
            ymin=int(bndbox.find('ymin').text)
            xmax=int(bndbox.find('xmax').text)
            ymax=int(bndbox.find('ymax').text)
            
            face_roi=img[ymin:ymax, xmin:xmax]
            
            if name=='with_mask':
                output_dir=os.path.join(output_base_path,'with_mask')
            elif name=='without_mask':
                output_dir=os.path.join(output_base_path,'without_mask')
            else:
                continue
            
            output_image_name=f"{os.path.basename(image_path).split('.')[0]}_{xmin}_{ymin}.jpg"
            cv2.imwrite(os.path.join(output_dir, output_image_name), face_roi)
            

print("[INFO] Data Preparation complete.")
print(f"Cropped faces saved to '{output_base_path}' directory.")           
            