from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--yolo","-y",type=str)
parser.add_argument("--imgsz","-i",type=int,default=320)
parser.add_argument("--batch","-b",type=int,default=2)
args = parser.parse_args()

model = YOLO(args.yolo)
model.export(
    format="onnx",
    dynamic=False,
    imgsz=args.imgsz,
    opset=12,
    simplify=True,
    batch=args.batch
)
print("Model exported")