import cv2
import torch
import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main() :
    fig = plt.figure(figsize = (6, 9))
    ax = plt.subplot()
    avgDropDict = {}

    img = cv2.imread('./crossing.jpg')
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Load an image
    for extension in ["n", "s", "m", "l", "x"] :
        batteryList = []
        model = torch.hub.load('ultralytics/yolov5', 'yolov5' + extension, pretrained=True)
        for x in range(0, 100) :

            # Inference
            model(img_rgb)

            battery = psutil.sensors_battery()
            batteryList.append(battery.percent)

        batteryList = np.array(batteryList)
        batteryList = batteryList - batteryList[0] + 100
        
        ax.plot(batteryList, label = extension)

        avgDiff = np.diff(batteryList).mean()
        avgDropDict[extension] = avgDiff

    ax.legend()
    fig.suptitle("YOLO Battery Consumption")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Battery %")
    fig.tight_layout()
    fig.savefig("./BatteryConsumption.png")
    plt.clf()

if __name__ == "__main__" :
    main()