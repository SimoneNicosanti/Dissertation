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