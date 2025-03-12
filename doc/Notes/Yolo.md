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


# Pre/Post Processing in Onnx
A questo link ci sono una serie di esempi con classi di pre e post processing (citare la fonte)
https://github.com/levipereira/ultralytics/tree/main/examples

In generale ci sono delle piccole discrepanze; possono essere dovute:
- Tagli di precisione in fase di export del modello da torch ad Onnx
- Modifiche sulle soglie di confidenza e di IoU (Non ho idea di dove il modello originale tiri fuori uno skateboard, ma comunque...)

Modificate leggermente le classi in modo da creare un'interfaccia comune per entrambi i task.

## Detection

| Modello Originale                         | Onnx PPP                                  |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20250227105701.jpg\|325]] | ![[Pasted image 20250227105805.jpg\|325]] |


## Segmentation

| Modello Originale                         | Onnx PPP                                  |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20250227105920.jpg\|325]] | ![[Pasted image 20250227105937.jpg\|325]] |

# Quantizzazione dei Modelli
La quantizzazione dei modelli si può realizzare con Onnx fornendo un dataset di calibrazione.

Come dataset di calibrazione è stato usato Coco128 (https://www.kaggle.com/datasets/ultralytics/coco128/).

Il problema principale della quantizzazione fatta con Onnx è che quando i modelli vengono quantizzati, le predizioni sulle classi vengono portate tutte quante a zero, portando quindi a non avere predizioni di nessun tipo.
Questo problema è riportato anche in:
- https://github.com/microsoft/onnxruntime/issues/14233

Come riportato in:
- https://github.com/microsoft/onnxruntime/issues/14233
- https://github.com/microsoft/onnxruntime/issues/17410
- https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2
I livelli più bassi della rete tendono ad avere delle performance pessime perché il range dei valori dei tensori è molto grande. La soluzione è quella quindi di non quantizzare questi livelli. Tuttavia questi link fanno riferimento a Yolov8 o Yolov5 e non a versioni più recenti.

Ci sono due alternative:
- Fare export torch-->TFLite (Quantized Int) --> Onnx
	- Troppe conversioni
	- Vengono sollevati degli errori
- Escludere alcuni livelli dalle quantizzazioni
	- Alternative
		- Si possono escludere n livelli vicino all'output per attenuare questo effetto
		- Anche l'esclusione dell'ultimo livello di Concatenazione porta ad avere delle predizioni con delle prestazioni buone, quindi ci si potrebbe limitare a questo livello

