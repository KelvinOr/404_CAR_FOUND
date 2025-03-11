import numpy as np
import pandas as pd
import cv2
import os
import requests
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
from ultralytics import YOLO
import torch
from collections import defaultdict
from collections import Counter
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import helpers 
from datetime import timezone
from dotenv import load_dotenv

load_dotenv()



# remove all files in test_gov_traffic_data
for file in os.listdir('test_gov_traffic_data'):
    os.remove(os.path.join('test_gov_traffic_data', file))

# Create output directory if it doesn't exist
output_dir = 'test_gov_traffic_data'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv('Traffic_Camera_Locations_Sc.csv')

for i in range(len(df)):
    try:
        url = df.iloc[i]['url']
        key = df.iloc[i]['key']
        
        # Download image with requests instead of cv2.imread
        headers = {'User-Agent': 'Mozilla/5.0'}  # Some sites require user-agent
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Convert bytes to numpy array
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            
            # Decode image using OpenCV
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Create safe filename
                filename = f"{key}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                # Save image
                if cv2.imwrite(output_path, img):
                    print(f"Saved: {filename}")
                else:
                    print(f"Failed to save: {filename}")
            else:
                print(f"Failed to decode image from: {url}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
            
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")


dir = "test_gov_traffic_data"
model = YOLO("yolo11x.pt")

#detect device is cudo or mps 
device = "cpu"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

hong_kong_road_data = []
df = pd.read_csv('Traffic_Camera_Locations_Sc.csv')


current_iso_date = datetime.now(timezone.utc).isoformat()


for image in os.listdir(dir):
    result = model.predict(source = os.path.join(dir, image),imgsz=(416), device=device)
    image_file_name = image.split(".")[0]
    region = df[df['key'] == image_file_name]['region'].values[0]
    district = df[df['key'] == image_file_name]['district'].values[0]
    description = df[df['key'] == image_file_name]['description'].values[0]
    #find location by image_file_name
    location = df[df['key'] == image_file_name]
    print(location.keys())
    latitude = location['latitude'].values[0].item()
    longitude = location['longitude'].values[0].item()

    plot = result[0].plot()
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

    image_return_data = dict()

    boxes = result[0].boxes
    if boxes.cls is not None and len(boxes.cls) > 0:
        classes = boxes.cls.cpu().numpy().astype(int) 
        class_names = [model.names[cls] for cls in classes]
        counts = Counter(class_names)
        
        #print(f"Image: {image}")
        for name , count in sorted(counts.items()):
            name_replace = name.replace(" ", "_")
            image_return_data[name_replace] = count
        #print("-" * 30) 
    else:
        print(f"Image: {image} - No objects detected")
        
    hong_kong_road_data.append({"@timestamp": current_iso_date ,"Key": image_file_name, "region": region, "district": district, "detail_location": description, "image_item": image_return_data, "geolocation": {"lat": latitude, "lon": longitude} })

#print(hong_kong_road_data)

ES_URL = os.getenv("ES_URL")
ES_USER =  os.getenv("ES_USER")
ES_PW =  os.getenv("ES_PW")

es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PW),
    verify_certs=False
)

ES_URL = os.getenv("ES_URL")
ES_USER =  os.getenv("ES_USER")
ES_PW =  os.getenv("ES_PW")

es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PW),
    verify_certs=False
)

# check if ilm policy exists
try:
    es.ilm.get_lifecycle(name="traffic_data_ilm")
    print("ILM policy already exists.")
except:
    es.ilm.put_lifecycle(
        name="traffic_data_ilm",
        body={
            "policy": {
                "phases": {
                    "hot": {
                        "min_age": "0ms",
                        "actions": {
                            "set_priority": {
                                "priority": 100
                            }
                        }
                    },
                    "delete": {
                        "min_age": "7d",
                        "actions": {
                            "delete": {}
                        }
                    }
                }
            }
        }
    )

if "traffic_data" not in es.indices.get_alias(index="*"):
    es.indices.create(
        index="traffic_data",
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.lifecycle.name": "traffic_data_ilm",         
                "index.lifecycle.rollover_alias": "traffic_data_alias"
            }, 
            "mappings": {
                "dynamic_templates": [
                    {
                        "all_fields_in_object_as_integer": {
                        "path_match": "image_item.*",
                            "mapping": {
                                "type": "integer"
                            }
                        }
                    }
                ],
                "properties": {
                    "@timestamp": {
                        "type": "date"
                    },
                    "Key": {
                        "type": "keyword"
                    },
                    "image_item": {
                        "type": "object"
                    },
                    "geolocation": {
                        "type": "geo_point"
                    },
                    "region": {
                        "type": "keyword"
                    },
                    "district": {
                        "type": "keyword"
                    },
                    "detail_location": {
                        "type": "text" 
                    }
                }
            },
            "aliases": {
                "traffic_data_alias": {
                    "is_write_index": True  
                }
            }
        }
    )
    es.indices.put_alias(index="traffic_data", name="traffic_data_alias")


bulk_data = []
for document in hong_kong_road_data:
    bulk_data.append({
        "_op_type": "index",       
        "_index": "traffic_data",  
        "_source": document        
    })

helpers.bulk(es, bulk_data)