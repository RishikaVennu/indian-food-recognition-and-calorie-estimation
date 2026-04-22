from google.colab import drive
drive.mount('/content/drive')
import os

PROJECT_PATH = "/content/drive/MyDrive/indian_food_ai"

os.makedirs(PROJECT_PATH, exist_ok=True)

%cd $PROJECT_PATH
!pip install ultralytics
from ultralytics import YOLO
import os
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="jxo3G7rn29gBhGwcQz")

# Dataset 1
project1 = rf.workspace("indianfoodnet").project("indianfoodnet")
dataset1 = project1.version(1).download("yolov8")

# Dataset 2
project2 = 
rf.workspace("south-indian-food-detection-and-classification").project("food-detection-nlusn")
dataset2 = project2.version(1).download("yolov8")

import yaml

def load_classes(data_yaml):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    return data["names"]

c1 = load_classes(dataset1.location + "/data.yaml")
c2 = load_classes(dataset2.location + "/data.yaml")

def normalize(name):
    name=name.lower()
    name=name.replace("briyani","biryani")
    name=name.replace("idly","idli")
    return name

c1=[normalize(c) for c in c1]
c2=[normalize(c) for c in c2]

all_classes=sorted(list(set(c1+c2)))

print("Total classes:",len(all_classes))

map1 = {c1.index(c): all_classes.index(c) for c in c1}
map2 = {c2.index(c): all_classes.index(c) for c in c2}

print(map1)
print(map2)

import glob

def fix_labels(folder,mapping):

    labels = glob.glob(folder+"/*.txt")

    for file in labels:

        new_lines=[]

        with open(file) as f:
            lines=f.readlines()

        for line in lines:

            parts=line.split()

            cls=int(parts[0])

            if cls in mapping:
                parts[0]=str(mapping[cls])

            new_lines.append(" ".join(parts))

        with open(file,"w") as f:
            f.write("\n".join(new_lines))

fix_labels(dataset1.location+"/train/labels",map1)
fix_labels(dataset1.location+"/valid/labels",map1)

fix_labels(dataset2.location+"/train/labels",map2)
fix_labels(dataset2.location+"/valid/labels",map2)

import shutil

merged="/content/merged"

folders=[
"train/images",
"train/labels",
"valid/images",
"valid/labels"
]

for f in folders:
    os.makedirs(merged+"/"+f,exist_ok=True)

def copy_files(src,dst):

    for f in os.listdir(src):
        shutil.copy(src+"/"+f,dst+"/"+f)

copy_files(dataset1.location+"/train/images",merged+"/train/images")
copy_files(dataset1.location+"/train/labels",merged+"/train/labels")

copy_files(dataset2.location+"/train/images",merged+"/train/images")
copy_files(dataset2.location+"/train/labels",merged+"/train/labels")

copy_files(dataset1.location+"/valid/images",merged+"/valid/images")
copy_files(dataset1.location+"/valid/labels",merged+"/valid/labels")

copy_files(dataset2.location+"/valid/images",merged+"/valid/images")
copy_files(dataset2.location+"/valid/labels",merged+"/valid/labels")

data_yaml=f"""
path: {merged}

train: train/images
val: valid/images

names: {all_classes}
"""

with open("data.yaml","w") as f:
    f.write(data_yaml)

print("data.yaml created")


model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project=PROJECT_PATH,
    name="food_detector"
)
