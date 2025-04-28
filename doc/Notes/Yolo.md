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


# Analisi dell'Input-Output

Gli output che vengono restituiti da un modello Yolo variano a seconda del task del modello: nello specifico l'output del task di detection è un sottoinsieme dell'output del task di segmentation.

## Pre Elaborazione
Data un'immagine bisogna fare in modo che la sua dimensione coincida con quella che il modello si aspetta di ricevere in input. Inoltre Yolo si aspetta di ricevere un input standardizzato.

I passi da fare per la pre-elaborazione quindi sono i seguenti:
1. Fare il ridimensionamento dell'immagine: questo ridimensionamento va fatto in maniera tale da mantenere le proporzioni corrette in modo da non creare problemi in fase di inferenza
2. Aggiungere il padding all'immagine: quando ridimensiono mantenendo il rapporto dell'immagine, non è detto che ottengo un'immagini con le dimensioni che mi servono. Per compensare questo aspetto quindi posso aggiungere del padding all'immagine in modo da ottenere l'input della dimensione che mi serve
3. Scambiare i canali: il modello yolo si aspetta un input in formato CHW, quindi se serve bisogna trasporre l'immagine mettendo prima il canale
4. Normalizzare, dividendo il tutto per 255

## Post Elaborazione
Il modello di detection ha un singolo output, in formato (batch_size, 4 + num_classes, 8400); in questo output abbiamo che:
- il 4 sono le informazioni sul bounding box in formato (x_centre, y_centre, width, height)
- I num_classes valori sono gli score per ogni classe per quella specifica anchor
- 8400 sono le anchor totali

Il modello di segmentazione ha due output:
- output0
	- Questo output ha formato (batch_size, 4 + num_classes + mask_info_size, 8400)
		- Come prima i 4 valori e i num_classes valori sono riferiti rispettivamente alle info delle bounding box e agli score per classe
		- I mask_info valori invece sono valori che servono, combinati con l'altro output a trovare la mask di segmentazione
- output1
	- Ha formato (batch_size, mask_info_size, mask_width, mask_height)
	- Questo output viene chiamato "prototype" e combinato con le mask info dell'altro output ci permette di trovare la mask di segmentazione

La fase di post-elaborazione della parte di bounding boxes quindi è uguale in entrambi i casi:
1. Si estraggono dall'output le parti relative alle bounding boxes
2. Per ogni anchor si trova la classe a score maggiore e il valore corrispondente di score, ottenendo un tensore (batch_size, final_size, 6)
3. Si applica la non-max-suppression prendendo solo le anchor con score e con iou maggiori di certa soglia.
4. A questo punto si ridimensionano le bounding box in modo da adattarle alla dimensione dell'immagine originale

La fase di post-elaborazione di masks e prototipi invece vale solo in caso di segmentazione; in questo caso:
1. Si estrae da output0 la parte di masks, ottenendo un tensore di size (batch_size, masks_info_size, 8400)
2. Per ogni elemento del batch, si fa il prodotto tra (8400, masks_info_size) con (masks_info_size, mask_width, mask_height)
	1. In sostanza quello che si sta facendo in questa fase è pesare le masks ottenute con i prototipi
	2. Otteniamo quindi un tensore di size (8400, mask_width, mask_height)
3. Si fa l'upscale del tensore così ottenuto in modo da farlo combaciare con la dimensione dell'immagine originale

La fase di moltiplicazione potrebbe essere onerosa in termini computazionali; c'è da dire però che questa non deve essere fatta su tutte le anchor, ma è sufficiente farla solo su quelle che passano la non maximum suppression, quindi il numero di calcoli da fare si riduce notevolmente.

Usando la libreria *supervision*, fornendo le info in formato richiesto si possono fare varie cose, tra cui disegnare l'output delle predizioni e le maskere. Supervision può essere utile perché permette di fare anche analisi di video in modo abbastanza trasparente.