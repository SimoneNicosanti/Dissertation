
## Analisi del Modello MobileNetV3Large
Analizzando l'attributo _config_ del modello (che restituisce un oggetto strutturato come un json che descrive il modello) sono riuscito a capire il problema relativo a quei livelli di Add e Multiply che, come mostrato in Netron, prendevano in input da un solo livello, cosa che risultava strana trattandosi di livelli aggreganti; il problema è che questi livelli in realtà prendono in input l'output di un livello precedente per poi eseguire un'operazione element-wise (appunto una somma o un prodotto) con una costante (nello specifico, per la somma la costante è sempre 3.0 mentre per la moltiplicazione è sempre 0.16 con 6 periodico). Quei livelli che venivano mostrati in netron in sostanza non sono affatto dei livelli (cioè istanziati in fase di costruzione del modello come dei keras.Layer), ma sono il risultato di operazioni del seguente tipo:  
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


> [!Warning] Warning ricevuto nel Parsing
> Durante il parsing del modello e la ricostruzione dei dizionari di input e di output viene ricevuto un Warning. In sostanza c'è una mancata corrispondenza tra i nomi dei tensori di input nel dizionario e le chiavi del dizionario stesso; comunque Keras sembra gestire la cosa abbastanza bene, ricostruendo il tutto correttamente.
> La cosa strana è che se cerco di risolvere il problema impostando un nuovo tensore di input ho errore di operazione ripetuta.
> 
> Non so se potrebbe dare problemi in altri punti dell'esecuzione o nel parsing di modelli diversi.
```
### Warining
/opt/conda/lib/python3.11/site-packages/keras/src/models/functional.py:106: UserWarning: When providing `inputs` as a dict, all keys in the dict must match the names of the corresponding tensors. Received key 'activation_18' mapping to value <KerasTensor shape=(None, None, None, 960), dtype=float32, sparse=False, name=keras_tensor_188> which has name 'keras_tensor_188'. Change the tensor name to 'activation_18' (via `Input(..., name='activation_18')`)

#############

ValueError: The name "expanded_conv_3_expand" is used 2 times in the model. All operation names should be unique.

```


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

Nel complesso i tempi di esecuzione sembrano migliori rispetto ad un'esecuzione con RPyC: tempi di esecuzione per 50 esecuzioni.

|            | Locale     | gRPC       | RPyC       |
| ---------- | ---------- | ---------- | ---------- |
| Mean       | 0.54862999 | 1.42035601 | 2.47241231 |
| Std Dev    | 0.04414955 | 0.12214391 | 0.61251661 |
| x / Locale | 1          | 2.6296     | 4.5065     |

### Distribuzione per Sotto Modelli
Implementazione del servizio con gRPC e per sotto modelli (invece che per singolo livello).

