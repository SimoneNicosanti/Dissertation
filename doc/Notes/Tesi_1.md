
## Analisi del Modello MobileNetV3Large
Analizzando l'attributo _config_ del modello (che restituisce un oggetto strutturato come un json che descrive il modello) sono riuscito a capire il problema relativo a quei livelli di Add e Multiply che, come mostrato in Netron, prendevano in input da un solo livello, cosa che risultava strana trattandosi di livelli aggreganti; il problema è che questi livelli in realtà prendono in input l'output di un livello precedente per poi eseguire un'operazione element-wise (appunto una somma o un prodotto) con una costante (nello specifico, per la somma la costante è sempre 3.0 mentre per la moltiplicazione è sempre 0.16 con 6 periodico). Quei livelli che venivano mostrati in netron in sostanza non sono affatto dei livelli (cioè istanziati in fase di costruzione del modello com[[Colloquio_1 (2024-11-19)]]e dei keras.Layer), ma sono il risultato di operazioni del seguente tipo:  
```
x1 = prevLayer(prevPrevOutput)  
x2 = x1 + 3  
nextLayer(x2)  
```

Quando viene fatta l'operazione di somma, questa non viene mappata su un oggetto di tipo layer all'atto della compilazione del modello, ma su un oggetto keras diverso, cioè un'istanza di una classe del modulo _keras.src.ops.numpy_ che mappano delle operazioni element-wise in keras.

Premettendo che non mi è molto chiaro il motivo per cui siano necessarie operazioni di questo tipo, in quanto riterrei più sensato imparare dei pesi del modello che fanno direttamente in modo, prendendo per riferimento l'esempio precedente, che x1 sia direttamente uguale al suo valore sommato 3 piuttosto che fare questo. Comunque, facendo il parsing del _config_, sono riuscito ad estrarre la classe di appartenenza di questi oggetti e farne il wrap all'interno di oggetti che si comportassero in modo simile a dei livelli, in modo da poterli distribuire al pari di livelli "classici".

> [!Warning] Validità e generalizzazione della soluzione
> A questo punto il dubbio che mi sorge è relativo alla validità di una soluzione di questo tipo in un contesto più generale: potrebbero esserci altri oggetti con cui keras fa cose simili di cui potrei non venire a sapere l'esistenza e che potrebbero dare problemi simili.


### Conferma di questo aspetto
In MobileNetV3 questo comportamento è dovuto alla definizione della *hard_sigmoid* (https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/applications/mobilenet_v3.py#L538), definita come:
```python
def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)
```
Il motivo per cui si usa questa funzione è che approssima bene la funzione *swish*. Questa funzione *swish*, che si vuole usare come funzione di attivazione nel modello, è definita come:
$$
swish(x)=x * \sigma(x)
$$
e risulta pertanto abbastanza costosa dal punto di vista computazionale (https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa). La *hard_sigmoid* permette di approssimarla (grafico) con un costo computazionale minore, che in questo tipo di modello è importante.

| Funzione H-Sigmoid e sua Approssimazione |
| ---------------------------------------- |
| ![[h-swish.png]]                         |
Il motivo per cui non si fa imparare alla rete direttamente i pesi che permettono di ottenere quel risultato è dovuto al fatto che in questo caso stiamo approssimando la funzione di attivazione (per quanto l'obiezione rimane valida in parte).

## Divisione di Modello Generico (Focus su MobileNetV3)
Riferimento al link https://gist.github.com/martinsbruveris/1ce43d4fe36f40e29e1f69fd036f1626.
```python
model = build_model()

cut_layer = model.get_layer(name="fc2")
model_a = Model(model.inputs, cut_layer.output)
model_b = Model(cut_layer.output, model.outputs)
```
Prendendo il layer tramite il suo nome e passando gli input e gli output del layer nelle divisioni del modello, si ottengono dei sotto modelli del modello originale che fanno la computazione solo di quel pezzo.
Considerazioni :
- Gestione dei nodi di fork e join
	- Se abbiamo delle pipeline parallele all'interno del modello, bisogna fare attenzione a dove vengono messi i nodi di fork e di join: se il subModel comincia con il nodo di fork e finisce con il nodo di join, tutto ciò che è intermedio viene inserito nel sotto modello.
	- Si potrebbe in realtà riuscire a fare una divisione abbastanza tranquilla in ogni punto della rete: supposto che i nodi di fork e join siano in server diversi, quando il fork esegue invia il suo input al join e alla parte di esecuzione dell'altro; quando il join riceve, deve aspettare prima la seconda parte di esecuzione, quindi ritorna null al nodo di fork; quando il join riceve la seconda parte di input allora emette il risultato e terminata l'esecuzione totale, il fork riceverà l'output dal secondo ramo.
		- NOOOO. Questa cosa non funziona (vedere [[Split_On_Join.png]]): per arrivare all'output di *multiply* c'è bisogno dell'output di entrambi i rami, quindi se parto da *Conv2D*, keras mette in automatico anche tutto l'altro ramo che serve per arrivare all'iput per quello specifico livello
		- Questa cosa funziona se si specificano come input del modello quelli provenienti da tutti i nodi di input e che non sono nel sotto modello (vedere [[Multi_Input_Variant.png]] e [[Multi_Input_Variant_Graph.png]]), ad esempio specifico come input sia l'output della relu (che diventa input della multiply) sia l'output dell'AvgPooling (che diventa inpur di Conv2D). 
		- Allo stesso modo funziona se il livello di partenza è un livello di Join (vedere [[Starting_With_Join.png]]) quindi ad esempio parto dalla Multiply e vado alla BatchNormalization.
- La gestione dei nodi che non sono layer (come per operazioni numpy) non dovrebbe più dare problemi: se si fa la divisione del modello usando SOLO i layers restituiti da *model.layers* quei nodi del grafo NON sono annoverato tra i layer quindi:
	- lo split lì non si può fare. 
	- Allo stesso tempo saranno considerati come parte della computazione per arrivare al nodo di output, quindi verranno presi in considerazione in automatico.

| Fork & Join             | Split on Join senza input   |
| ----------------------- | --------------------------- |
| ![[Fork-Join.png\|300]] | ![[Split_On_Join.png\|300]] |

```python
subModel = keras.Model(
	inputs=[
		model.get_layer("expanded_conv_11_squeeze_excite_conv").input,
		model.get_layer("activation_12").output],
	outputs=model.get_layer("expanded_conv_11_squeeze_excite_mul").output,
)
```

| Inizio con Nodo di Join     | Split on Join con Input aggiuntivi                     |
| --------------------------- | ------------------------------------------------------ |
| ![[Starting_With_Join.png]] | ![Variante Multi Input](Multi_Input_Variant_Graph.png) |

Posto di fare molta attenzione a dove e come il modello viene tagliato e quali e quanti sono gli input che si aspetta di ricevere. 
In linea di principio:
- Se il sotto modello è sequenziale non ho problemi
- Se il sotto modello contiene il fork ma non il join non ho problemi: devo inviare l'output del sotto modello a tutti i successori (che comunque devo di trovare)
- Se il sotto modello contiene il join, ma non il fork devo fare attenzione a mettere nell'input del sotto modello gli output di tutti i rami che non sono compresi nel sotto modello (che comunque devo trovare)
- Se il sotto modello contiene sia il fork sia il join corrispondente non ho problemi

Devo comunque fare attenzione a ricostruire da quali livelli mi aspetto di ricevere l'input per poter recuperare le informazioni sui loro tensori, in particolare se il sotto modello contiene dei join.

Pseudocodice
```python
for layer in subModel:
	layerInputs <- get input layers
	for inputLayer in layerInputs:
		if inputLayer in subModel:
			continue
		else:
			add inputLayer.output as input of the submodel
```

Dato un nodo devo trovare i primi layer validi che lo precedono e che lo succedono.
```python
### Cerco i nodi predecessori e successori
for layer in model:
	inputOps <- get layer input ops
	prevLayers = []
	for op in inputOp:
		if op is a layer:
			prevLayers.append(op)
		else :
			opPrevLayers <- get op prev layers
			prevLayers.append(opPrevLayers)
		## Updating prev layers
		layer.prevLayers = prevLayers
		## Updating next layers
		for otherLayer in prevLayers :
			otherLayer.nextLayers.append(layer)
	
```

La divisione del modello fatta in questo modo funziona!!!

Bisogna un attimo ricostruire però il rapporto tra nomi dei tensori di input e nomi dei tensori di output. Il codice di seguito crea un raccordo tra nome del livello precedente e nomi dei tensori in input al sotto modello da quegli stessi livelli.
```python
subModelInput += [
	keras.Input(
		shape=prevLayerOut[i].shape[1:],
		tensor=prevLayerOut[i],
		name=prevLayer,
		) 
	for i in range(0, len(prevLayerOut))
	]
```

| Output Sotto-Modello Precedente               | Input Sotto-Modello Successivo    |
| --------------------------------------------- | --------------------------------- |
| ![[Output Sotto Modello Precedente.png\|280]] | ![[Input Sotto Modello.png\|307]] |
Funziona ma parzialmente: bisogna fare in modo che gli input e gli output dei modelli siano dei tensori con nomi, in modo da poter raccordare l'output di un modello con l'input del modello successivo.
RISOLTO: Creati dei dict per rappresentare input ed output e assegnati i nomi corrispondenti che servono.


> [!Bug] Non era risolto in realtà...
> > [!Done] Risolto
> > Risolto creando prima il nuovo oggetto di tipo keras.Input partendo dal tensore output del livello precedente e poi modificandone il nome con *obj.name*.
> > > [!Warning] Warning ricevuto nel Parsing
> > > Durante il parsing del modello e la ricostruzione dei dizionari di input e di output viene ricevuto un Warning. In sostanza c'è una mancata corrispondenza tra i nomi dei tensori di input nel dizionario e le chiavi del dizionario stesso; comunque Keras sembra gestire la cosa abbastanza bene, ricostruendo il tutto correttamente.
> > > La cosa strana è che se cerco di risolvere il problema impostando un nuovo tensore di input ho errore di operazione ripetuta.
> > > 
> > > Non so se potrebbe dare problemi in altri punti dell'esecuzione o nel parsing di modelli diversi.
> > > Comunque sì: c'era un problema nel nome dell'input quando il sotto modello era esportato come saved_model.
> > 
> > Con questa soluzione sembra funzionare sia per formato .keras sia per formato saved_model
> 
> Quando si crea una nuova lista con i diversi modelli veniva dato un errore al salvataggio del modello. Probabilmente era dovuto al fatto che quel tensore veniva trasformato in un placeholder.

```
### Warining
/opt/conda/lib/python3.11/site-packages/keras/src/models/functional.py:106: UserWarning: When providing `inputs` as a dict, all keys in the dict must match the names of the corresponding tensors. Received key 'activation_18' mapping to value <KerasTensor shape=(None, None, None, 960), dtype=float32, sparse=False, name=keras_tensor_188> which has name 'keras_tensor_188'. Change the tensor name to 'activation_18' (via `Input(..., name='activation_18')`)

#############

ValueError: The name "expanded_conv_3_expand" is used 2 times in the model. All operation names should be unique.

```
Il secondo errore qui era dovuto al fatto che in realtà il modello non era diviso correttamente, ma ripartiva da capo tutte le volte, portando quindi a ripetizione delle operazioni.

> [!Bug] In realtà non risolveva il problema
> Codice che risolve il problema
> ```python
> if len(prevOpsDict[layer.name]) == 0:
> 	## Input Layer
> 	newInput = keras.Input(
> 		shape=layer.output.shape[1:],
> 		tensor=layer.output,
> 		name=layer.name,
> 	)
> 	newInput.name = layer.name
> 	subModelInput[layer.name] = newInput
> else:
> 	for prevLayerName in prevOpsDict[layer.name]:
> 		prevLayerOut = model.get_layer(prevLayerName).output
> 		if prevLayerName not in subLayersNames:
> 			newInput = keras.Input(
> 				shape=prevLayerOut.shape[1:],
> 				tensor=prevLayerOut,
> 				name=prevLayerName,
> 			)
> 			newInput.name = prevLayerName
> 		subModelInput[prevLayerName] = newInput
> ```

Per adesso risolto così, ma il warning rimane, anche se non dovrebbe dare problemi.
```python
subModelInput = {}
for layer in subLayers:
	if len(prevOpsDict[layer.name]) == 0:
		subModelInput["input"] = layer.output
	else:
		for prevLayerName in prevOpsDict[layer.name]:
			prevLayer = model.get_layer(prevLayerName)
			prevLayerOut = (
				prevLayer.output
				if isinstance(prevLayer.output, list)
				else [prevLayer.output]
			)
			if prevLayerName not in subLayersNames:
				for _, out in enumerate(prevLayerOut):
				# newInput = keras.Input(tensor=prevLayerOut)
					subModelInput[out.name] = out
return subModelInput
```

> [!NOTE] Distribuzione per Livello Singolo
> Notare che questa stessa distribuzione si può fare nello stesso modo se si vuole dividere il modello per livelli singoli: è sufficiente mettere 1 come size massima della sotto lista di layers.

> [!Done] Nuova Divisione del Modello
> Piuttosto che fare la divisione partendo dal *config* del modello (in caso di modelli strani / complessi il config viene esportato in modo strano quindi non funziona), uso direttamente gli attributi degli oggetti keras!!
> Posso trovare i livelli validi predecessori del modello direttamente partendo dai kerasTensor e dalle loro History
>> ```python
>> kerasTensor._keras_history ## Gets its operation history
>> kerasTensor._keras_history.operation ## Gets list of previous operations
>> model.output._keras_history.operation.input ## Gets operation input (it is a keras tensor!!)
>> ```
>
>Funziona decisamente meglio e non dovrebbe dare problemi di parsing del config qualora ci fossero modelli con config non standard


> [!bug]
> Per modelli più complessi l'output di un layer potrebbe essere una lista o un dizionario: in questo caso bisogna gestire vedendo quale output di questo livello viene ricevuto come input del livello corrente. Potrebbero esserci altri problemi dovuti ai nomi di questi input però...
> Vedere esempio fatto con KerasCV e YoloV8.
> 
> C'è differenza tra un layer che dà il suo output a più di un livello successivo e un livello che ha proprio più input. In questo caso il Parsing fatto non funziona perché assume un unico output

| Errore in caso di livello con più Keras Tensors di output |
| --------------------------------------------------------- |
| ![[Problema Livello Con Multi Output.png]]                |
Modificato il parsing in modo che supporti delle liste, ma per adesso se un layer ha più di un output non posso recuperare quale output di quel layer sia usato come input del layer corrente.


NOTA!!! AGGIORNARE LA MODIFICA FATTA SU Conversions In Parsing

## Protocollo di Serializzazione usato da RPyC
Formato usato *Brine*.

> [!Note] Brine
> _Brine_ is a simple, fast and secure object serializer for **immutable** objects. The following types are supported: `int`, `bool`, `str`, `float`, `unicode`, `bytes`, `slice`, `complex`, `tuple` (of simple types), `frozenset` (of simple types) as well as the following singletons: `None`, `NotImplemented`, and `Ellipsis`.
> https://rpyc.readthedocs.io/en/latest/api.html#serialization

Potrebbe non essere adatto al trasferimento di dati complessi perché c'è il doppio passaggio, uno per serializzare e altro per convertire in Brine.

## Implementazione distribuzione con gRPC
### Distribuzione per Livello Singolo
Controllare l'articolo per vedere le differenze nell'uso del JSON e di protocol buffer
https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62

> [!Warning] Massima dimensione messaggio gRPC
> Problema nell'uso di gRPC: la massima dimensione del messaggio è 4 MB, quindi o si aumenta il limite oppure si passa ad un invio in Stream in caso di batch o input particolarmente grandi

> [!Warning] Problema con il numero di thread gestibili
> In gRPC si deve specificare il numero di thread che possono eseguire un certo servizio. Se sono meno del numero di layer gestiti da un'istanza di servizio si rischia il blocco. Questo aspetto deve essere gestito in qualche modo.


Per risolvere il secondo problema si potrebbero usare delle chiamate non bloccanti, oppure si fa tornare l'output parziale al front-end che poi lo invia al prossimo layer (troppo passaggio dati e troppa latenza forse...). Altrimenti si potrebbe fare implementando delle call asincrone implementazione di call asincrone

Contesto di esecuzione: 
* Se il livello successivo è gestito dallo stesso server viene fatta direttamente la chiamata locale
* Non ci sono più richieste e quindi non ci sono nemmeno dei meccanismi di sincronizzazione sulle strutture dati condivise del server (anche se comunque vengono accedute in lettura)

Nel complesso i tempi di esecuzione sembrano migliori rispetto ad un'esecuzione con RPyC.

| Dati su 50 Run | Niente  | gRPC    | RPyC    |
| -------------- | ------- | ------- | ------- |
| Mean           | 0.54863 | 1.42036 | 2.47241 |
| Std Dev        | 0.04415 | 0.12214 | 0.61252 |
| x / Locale     | 1       | 2.58892 | 4.5065  |

### Distribuzione per Sotto Modelli usando gRPC
Implementazione del servizio con gRPC e per sotto modelli (invece che per singolo livello).

Riuscito abbastanza tranquillamente. Aspetti significativi dell'implementazione:
- Interazione Registry-Server
	- Il server si registra sul registry; il registry risponde con un indice che rappresenta la porzione di modello che quel server deve prendere in carico
	- Il server analizza l'input del suo sotto modello e invia al registry quali sono i livelli di input del suo sotto modello
	- Il registry registra questi livelli associandoli alla coppia (indirizzo IP - Porta) del server in questione
- Interazione Client-Server
	- Il Client genera il suo input
	- Chiede al registry dove si trova l'input layer
	- Invia l'input al modello
	- Riceve dizionario di input con i diversi output del modello
- Elaborazione del Server
	- Il server riceve un messaggio di input
		- Per il requestId ricevuto si salva quell'input in un dizionario
		- Se ho ricevuto tutti gli input per il mio sotto modello per una specifica richiesta
			- Elaboro l'input e ricevo un dizionario di output
			- Per ogni livello di output
				- Chiedo al registry dove si trova il livello di input corrispondente
				- Invio l'output
		- Se mi manca qualche input
			- Ritorno un output vuoto


Contesto di esecuzione:
- 100 Run
- 5 Server, ognuno con una porzione del modello diversa
- Unico client (senza richieste concorrenti)
- Niente GPU
- Flusso completamente delegato ai server (le risposte non tornano indietro al client ma sono i server che si passano il risultato intermedio della computazione)

| Dati su 100 Run | Niente  | gRPC    |
| --------------- | ------- | ------- |
| Mean            | 0.52443 | 0.60116 |
| Std Dev         | 0.02925 | 0.02863 |
| x / Locale      | 1       | 1.14631 |
L'impatto della distribuzione è nel complesso molto minore rispetto all'implementazione precedente: nel caso precedente l'input veniva passato per ogni layer e quindi lo stack di chiamate si accresceva appesantendo l'esecuzione. In questo caso ad una chiamata corrisponde l'esecuzione dell'intero sotto modello, quindi 

> [!Bug] Ultimo Layer
> Bisogna fare attenzione a come identificare l'ultimo layer. Non posso identificarlo come quel layer che non ha successori, perché il registry potrebbe rispondermi picche anche per nodi gestori dei successivi che sono caduti nel frattempo.
> 
> Si potrebbe gestirlo con un nome fisso (come nel caso dell'input-layer), ma perderei flessibilità per modelli a più output.
> 
> Lo potrei aggiungere in dei metadati del modello che viene preso in carico da un certo server.


> [!Warning] Da aggiungere
> Aggiungere il fatto che il registry gestisce una delle liste per quanto riguarda gli input attesi. Dato un input di un sotto modello possono esserci più sotto modelli che lo aspettano.



### Distribuzione per Sotto Modelli con TF Serving
Se tutto ha senso come dovrebbe, si potrebbe fare anche la divisione e far gestire ogni sotto modello a dei TF Server.
Aspetti da considerare:
- Distribuzione dell'input
	- Si potrebbe creare una componente di ricezione dell'input che aspetta di ricevere l'input completo del sotto modello per poi chiamare TF Serving che lo esegue
- L'output del sotto modello non viene inviato subito al sotto modello successivo, ma torna al chiamante
	- A meno di non gestire in queste componenti di arrivo anche l'aspetto di invio dell'output ai sotto modelli successivi

Comando per lanciare il TF Serving
```shell
tensorflow_model_serving --port={x gRPC} --rest_api_port={} --model_name={} --model_base_path={}

```


## Conversione da Torch a Keras
Esiste una versione di YOLOv8 già offerta in keras (https://keras.io/api/keras_cv/models/tasks/yolo_v8_detector/).

### Analisi libreria onnx2tf
Oltre a questo la libreria *onnx2tf* permette contestualmente di:
- Creare un saved_model con dentro diversi .tflite corrispondenti 
- Ritornare un istanza di tf_keras.Model
In questo caso non abbiamo propriamente un keras.Model, ma la versione di Model del modulo Keras di tensorflow.
BISOGNA VEDERE SE SI RIESCE A CONVERTIRE QUESTA IN un 

Nel modello convertito da onnx alcuni parametri sono presi come *args* altri come *kwargs*, ma nell'input al layer sono elencati solo quelli che sono passati come *args*, quindi i collegamenti tra layer non sono ricostruiti correttamente; per accedere all'elenco completo degli argomenti si può fare così:
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



## Calcolo dei FLOPS
Riferimento al link (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md)


Il calcolo dei FLOPS per modello si può fare con il seguente:
```python
input_signature =[
				  tf.TensorSpec(shape=(1, 32, 32, 3), dtype=params.dtype, name=params.name)
	
	for params in model.inputs
]

forward_graph = tf.function(model, input_signature).get_concrete_function().graph
options = option_builder.ProfileOptionBuilder.float_operation()

graph_info: GraphNodeProto = model_analyzer.profile(forward_graph,
													options=options)
flops = graph_info.total_float_ops

print(f"TOTAL MODEL FLOPS >>> {flops}")
```
IMPORTANTE!! Aggiungere le dimensioni del tensore, altrimenti si ottiene un valore diverso che non so quanto senso abbia.

Nel report che viene stampato, vengono stampate informazioni relative ad ogni livello e alle operazioni singole che sono fatte in quel livello: ad esempio la convoluzione è considerata come operazione unica, mentre la Activation e la Batch Normalization vengono viste come operazioni composte

| Convoluzione Singola    | ![[Schermata del 2024-11-26 18-42-37.png\|500]] |
| ----------------------- | ----------------------------------------------- |
| **Attivazione Singola** | ![[FLOPS_Activation.png]]    |
| **Batch Normalization** | ![[FLOPS_Expanded_Conv.png]]      |
|                         |                                                 |
|                         |                                                 |

> [!Done]
> In realtà non è un problema vero e proprio... la re_lu non fa delle vere e proprie operazioni aritmetiche, ma controlla che un elemento sia $\geq 0$, quindi non si tratta di prodotti o somme o operazioni che ha senso contare come operazioni floating points. 
> > [!Warning] Assenza Re_Lu
> > Nel caso di MobileNetV3Large le Re_Lu non sono considerate


*graph_info* è un oggetto .proto con un campo repeated di di nome children; accedendo a questi children, iterando e prendendo la seconda parte della stringa (splittata sullo /) si ottengono i flops per operazione; si somma sul totale del livello.
```python
opsDict = {}
## graph_info.children è un repeated di oggetto Proto
for child in graph_info.children:
	childName: str = child.name
	childNameParts: list = childName.split("/") ## First Part is model name
	levelName: str = childNameParts[1]
	if levelName not in opsDict:
		opsDict[levelName] = 0
		opsDict[levelName] += child.total_float_ops
```

> [!Warning] Attenzione
> Queste sono le operazioni in virgola mobile, non i FLOPS: per ottenere i FLOPS si dovrebbe dividere per il tempo di esecuzione.
> Fatto in questo modo quindi posso recuperare i FLOPS per il modello ma non per il singolo livello/operazione che viene eseguita.


In alternativa, si può fare il calcolo dei FLOPS usando il parsing del modello: per si crea un sotto modello per ogni livello del modello originale, si calcola il suo tempo di esecuzione e si fa il monitoring sulle sue operazioni floating point. In questo caso escono fuori delle ReLU che hanno delle operazioni floating point, ma in realtà si tratta di quelle ReLU che vengono accorpate con i livelli fantasma di Add e Multiply, quindi le operazioni contate sono quelle relative a questi livelli.

| ReLU con Add       | ![[ReLU_FLOPS.png\|400]] |
| ------------------ | ----------------------------------------------- |
| **ReLU senza Add** | ![[ReLU_No_FLOPS.png\|400]] |
A questo punto credo che l'opzione migliore e più comoda sia l'altra: anche se questa permette di fare tutto in un modo, è un po' troppo macchinosa.

## Analisi di Keras_CV
Ha anche YoloV8 pre implementato (https://keras.io/examples/vision/yolov8/)

Si basa su delle classi Backbone che rappresentano le strutture di altre reti.

Danno alcuni problemi con la suddivisione dei modelli: si tratta proprio di due modelli diversi, infatti hanno anche un numero diverso di layers.

| Numeri di Layers per BackBone e senza      |
| ------------------------------------------ |
| ![[Numero Livelli Tra MobileNet KerasCV e Keras.png]] |

