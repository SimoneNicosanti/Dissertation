import cv2
import torch
import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def main() :
    fig = plt.figure(figsize = (6, 9))
    ax = plt.subplot()
    avgDropDict = {}

    img = cv2.imread('./crossing.jpg')
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Load an image
    timeList = []
    batteryList = []
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    for x in range(0, 100) :
        start = time.time()
        # Inference
        model(img_rgb)
        end = time.time()

        battery = psutil.sensors_battery()
        batteryList.append(battery.percent)
        timeList.append(end - start)

    batteryList = np.array(batteryList)
    batteryList = batteryList - batteryList[0] + 100
    
    ax.plot(batteryList, label = "Yolo M")

    batteryList = []
    for x in range(0, 100) :
        batteryList.append(psutil.sensors_battery().percent)
        time.sleep(timeList[x])
        print(f"Iteration >>> {x}")
    
    batteryList = np.array(batteryList)
    batteryList = batteryList - batteryList[0] + 100
    ax.plot(batteryList, label = "Idle")
        

    ax.legend()
    fig.suptitle("YOLO - Idle Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Battery %")
    fig.tight_layout()
    fig.savefig("./YOLO_IdleComparison.png")
    plt.clf()

if __name__ == "__main__" :
    main()