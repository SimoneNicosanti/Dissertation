## Con Monitoring + Callback

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

Nel report che viene stampato, vengono stampate informazioni relative ad ogni livello e alle operazioni singole che sono fatte in quel livello: ad esempio la convoluzione è considerata come operazione unica, mentre la Activation e la Batch Normalization vengono viste come operazioni composte.

| Convoluzione Singola    | ![[Schermata del 2024-11-26 18-42-37.png\|500]] |
| ----------------------- | ----------------------------------------------- |
| **Attivazione Singola** | ![[FLOPS_Activation.png]]                       |
| **Batch Normalization** | ![[FLOPS_Expanded_Conv.png]]                    |

> [!Done]
> In realtà non è un problema vero e proprio... la re_lu non fa delle vere e proprie operazioni aritmetiche, ma controlla che un elemento sia $\geq 0$, quindi non si tratta di prodotti o somme o operazioni che ha senso contare come operazioni floating points.
>
> > [!Warning] Assenza Re_Lu
> > Nel caso di MobileNetV3Large le Re_Lu non sono considerate

_graph_info_ è un oggetto .proto con un campo repeated di di nome children; accedendo a questi children, iterando e prendendo la seconda parte della stringa (splittata sullo /) si ottengono i flops per operazione; si somma sul totale del livello.

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

Per il calcolo dei tempi di esecuzione si può inserire una callback nell'inferenza; la callback fa il wrap della funzione di inference del livello in una funzione che prende il tempo di esecuzione del livello. Quando viene chiamato il livello viene registrato il suo tempo di esecuzione; allo stesso modo si può registrare il tempo di esecuzione del modello complessivo (anche se c'è un pelo più di tempo dovuto alla registrazione dei tempi).
```python
def wrap_layer(self, layer):
	original_call = layer.call
	
	def timed_call(inputs, *args, **kwargs):
		start_time = time.time_ns()
		result = original_call(inputs, *args, **kwargs)
		layer_name = layer.name
		duration = time.time_ns() - start_time
		if layer_name not in self.layer_times:
			self.layer_times[layer_name] = []
		self.layer_times[layer_name].append(duration)
	
		return result

	layer.call = timed_call

def set_model(self, model):
	super().set_model(model)
	for layer in model.layers:
		self.wrap_layer(layer)
```

### Con Parsing del Modello
In alternativa, si può fare il calcolo dei FLOPS usando il parsing del modello: per si crea un sotto modello per ogni livello del modello originale, si calcola il suo tempo di esecuzione e si fa il monitoring sulle sue operazioni floating point. In questo caso escono fuori delle ReLU che hanno delle operazioni floating point, ma in realtà si tratta di quelle ReLU che vengono accorpate con i livelli fantasma di Add e Multiply, quindi le operazioni contate sono quelle relative a questi livelli.

| ReLU con Add       | ![[ReLU_FLOPS.png\|400]]    |
| ------------------ | --------------------------- |
| **ReLU senza Add** | ![[ReLU_No_FLOPS.png\|400]] |

A questo punto credo che l'opzione migliore e più comoda sia l'altra: anche se questa permette di fare tutto in un modo, è un po' troppo macchinosa.

## Calcolo dei FLOPS Con Classe Tracker
Per tracciare l'esecuzione dei tempi avevo usato una callback che faceva il wrap della funzione di call per ogni operazione in un'altra funzione.

Il problema è che poi la funzione di call viene modificata in modo irreversibile: anche se reimpostato con il metodo *on_predict_end* non si riesce a reimpostare la call originale!! Il motivo è che la predict fa passare il modello ad uno stato in cui gli attributi del modello sono immodificabili e quindi non si può reimpostare la call originale anche se salvata.

Visto che le callback si possono impostare solo nei metodi di *predict* o di *fit*, continuare ad usare le callback non è fattibile.

Altro problema dell'implementazione precedente nel calcolo dei FLOPS era che i nomi delle operazioni non sempre coincidevano, quindi non si riusciva a mappare dato il nome dell'operazione sui rispettivi FLOPS e tempi di esecuzione.

### Calcolo dei Tempi di Esecuzione
Procediamo in questo modo:
```python
for op in model.operations:
	timeTracker <- create TimeTracker from op
	op.call = timeTracker.track
	map op.name on timeTracker

run model(input)

for op in model.operations :
	op.call <- tracker.originalCall

```

All'interno della classe Tracker teniamo traccia di:
- funzione di call originale
- nome dell'operazione
- tempi di esecuzione
Quando facciamo la call sull'operazione, salviamo il tempo di esecuzione.

Il metodo track ha la stessa segnatura della call dell'operazione: all'interno della track:
1. Prendiamo istante di start
2. eseguiamo
3. Prendiamo istante di stop e calcoliamo il tempo di esecuzione
4. Salviamo il tempo di esecuzione in una lista della classe Tracker

In questo modo, non usando la predict riusciamo a ripristinare la funzione di call originale.

### Calcolo delle Float Operations
Dato uno specifico input, ci serve sapere per ogni operazione quale è l'input che riceverebbe come conseguenza di quell'esecuzione. Se io mando un input di shape (1, 64, 64, 3) al modello, un'operazione intermedia riceve un input di shape conseguente all'esecuzione che è stata fatta su questo input. 

Dato l'input al modello, dobbiamo trovare PER OGNI operazione quale è l'input che riceve in conseguenza di quell'input. Possiamo quindi fare la stessa cosa che abbiamo fatto con i tempi ma facendo uno ShapeTracker.

Il metodo *track* dello ShapeTracker si segna, alla chiamata dell'operazione, quale è l'input che sta ricevendo: questo ci permette di ricostruire le shape degli input di tutte le operazioni conseguenti ad un certo input del modello.

Per trovare il numero di operazioni float point per ogni operazione, non potendo usare il profile del modello completo (perché ho discrepanza tra i nomi), si fa una cosa diversa.
Per ogni operazione si fa il wrap di questa operazione in una tf.function e si fa il profile dei flops della tf.function costruita in questo modo: così sono sicuro di poter ricostruire le corrispondenze dei nomi che altrimenti erano sballate.

Questo approccio ha un problema: ci sono alcuni tipi che non sono accettati come input di una tf.function (come slice per fare i tagli dei tensori), quindi viene sollevato errore: considerando che queste operazioni sono rare (solo 2 in YOLOv8) e che tendenzialmente hanno un numero di Float Operations nullo, si considerano i loro FLOPS nulli per default.

> [!Tip] Altro Approccio
> Un alternativa potrebbe essere, per avere una stima forse più accurata:
> - prendere i flops per operazione e le operazioni su cui non si possono calcolare
> - prendere i flops per modello (che posso calcolare sempre)
> - Faccio la differenza dei totali e se c'è avanzo ripartisco l'avanzo tra le operazioni per cui non posso calcolare direttamente i flops

