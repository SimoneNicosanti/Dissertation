import keras


def main():
    model_1: keras.Model = keras.applications.mobilenet_v2.MobileNetV2()

    model_1.save("../../../models/server/mobile_net.keras")

    model_2: keras.Model = keras.applications.MobileNetV3Large()
    model_2.save("../../../models/server/mobile_net_1.keras")


if __name__ == "__main__":
    main()
