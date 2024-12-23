## Conversione da Torch a Keras

Esiste una versione di YOLOv8 già offerta in keras (https://keras.io/api/keras_cv/models/tasks/yolo_v8_detector/).

### Analisi libreria onnx2tf

Oltre a questo la libreria _onnx2tf_ permette contestualmente di:

- Creare un saved_model con dentro diversi .tflite corrispondenti
- Ritornare un istanza di tf_keras.Model
  In questo caso non abbiamo propriamente un keras.Model, ma la versione di Model del modulo Keras di tensorflow.

Nel modello convertito da onnx alcuni parametri sono presi come _args_ altri come _kwargs_, ma nell'input al layer sono elencati solo quelli che sono passati come _args_, quindi i collegamenti tra layer non sono ricostruiti correttamente; per accedere all'elenco completo degli argomenti si può fare così:

```
for elem in currLayer._inbound_nodes :
	currLayer._inbound_nodes[0]._flat_arguments
```

> [!Error] Versioning
> Il problema che c'è nell'uso di onnx2tf è che ritorna un modello tf_keras che corrisponde alla versione 2 di keras, non alla versione 3.
> Quando si cerca quindi di fare il parsing (con l'implementazione già usata) le strutture sono completamente diverse

> [!Quote] Dalla documentazione di Keras
> Starting with TensorFlow 2.16, doing `pip install tensorflow` will install Keras 3. When you have TensorFlow >= 2.16 and Keras 3, then by default `from tensorflow import keras` ([`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)) will be Keras 3.
>
> Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf_keras` (or equivalently `tf-keras` – note that `-` and `_` are equivalent in PyPI package names). To use it, you can install it via `pip install tf_keras` then import it via `import tf_keras as keras`.

Il problema quindi sta nelle diverse versioni di Keras: si potrebbe valutare il downgrade alla versione precedente di Keras...

### Analisi Libreria nobuco

Anche lei usa la versione 2 di keras


## Analisi di Keras_CV

Ha anche YoloV8 pre implementato (https://keras.io/examples/vision/yolov8/)

Si basa su delle classi Backbone che rappresentano le strutture di altre reti.

Danno alcuni problemi con la suddivisione dei modelli: si tratta proprio di due modelli diversi, infatti hanno anche un numero diverso di layers.

| Numeri di Layers per BackBone e senza                 |
| ----------------------------------------------------- |
| ![[Numero Livelli Tra MobileNet KerasCV e Keras.png]] |

Nel modello che si crea a partire dalla backbone, la backbone viene inclusa come parte del nuovo modello come se fosse un solo layer; in figura sono stampati:
- Numero di livelli del modello finale e della backbone
	- Nel finale la backbone è considerata come un livello unico
- Il Numero di livelli della backbone
	- Stampato solo se un livello del modello principale è un modello a sua volta
- Risultato di layer.layers == backbone.layers
	- A riprova che il layer è proprio la backbone (il risultato del confronto è true)

| Output di Test sul Modello Completo        |
| ------------------------------------------ |
| ![[Schermata del 2024-11-29 15-47-25.png]] |
### Decoding
Il decoding in Yolo viene fatto solo nel caso in cui venga chiamata la `predict` e non se viene chiamata la semplice call.

Provare ad integrare il decoding direttamente nel modello solleva un errore relativo all'impossibilità di calcolare la dimensione di Output del modello. Infatti anche la stessa implementazione di Yolo non lo integra nel modello, ma lo chiama con una funzione a parte a valle della predizione dei valori non decodificati.
![[Schermata del 2024-12-23 16-55-44.png|Errore ricevuto all'integrazione del decoding]]

Una possibile alternativa in questo senso è quella di creare al pari del modello principale una funzione di Utility che fa la decodifica, copiandola dall'implementazione originaria.