
import ultralytics 
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

results = model.train(data="config.yaml", epochs=20)
