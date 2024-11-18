import keras

def main() :
    model : keras.Model = keras.applications.mobilenet_v2.MobileNetV2()

    model.save("../../../models/server/mobile_net.keras")

if __name__ == "__main__" :
    main()