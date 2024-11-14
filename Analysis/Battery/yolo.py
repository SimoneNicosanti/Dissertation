import torch

def main() :
        
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained = True)
    print(model)
    

if __name__ == "__main__" :
    main()