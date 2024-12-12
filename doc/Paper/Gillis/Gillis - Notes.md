## Background
### Serverless
Uso delle VM:
- Pro
	- Maggiori risorse
- Cons
	- Startup lungo
	- Richiede high-provisioning per far fronte a picchi di carico

Le funzioni serverless:
- pros
	- Tempi startup minori
	- Facile scaling
- Cons
	- Non per carichi stabili
		- Prezzi alti
	- Risorse limitate
		- Calcolo
		- Comunicazione
		- Banda di rete

Possibile unione dei due approcci

### Inefficienza servizio modelli grandi
Essendo i modelli grandi, le funzioni serverless faticano a gestire la richiesta del modello intero perché devono gestire lo switch delle risorse (esempio memoria) e hanno delle capacità di calcolo limitate.

### Studi precedenti gestione modelli grandi
Due approcci:
- Model Compression
	- Riduzione parametri
	- Tecniche
		- Network Pruning
		- Weight Quantization
	- Conseguenze
		- Riduzione accuratezza
			- Può essere gestita con tuning o retraining del modello compresso
- Model Partitioning
	- Divisione della rete in parti eseguite in parallelo

L'approccio scelto è quello di Model Partitioning. La tecnica è Tensor Partitioning:
- Esempio. Partizione di tensore a tre dimensioni (escluso batch; esempio immagine) 
	- Partizionamento per larghezza e altezza
	- Partizionamento dei filtri
Vedere [[gillis-icdcs21.pdf#page=3&selection=201,0,202,34|gillis-icdcs21, pagina 3]]


## Overview
### Workflow
1. Runtime profiling
	1. Per ogni livello, valuta:
		1. Tempo di esecuzione del livello in una funzione singola
		2. Latenza della funzione
	2. Costruisci un modello di performance
	3. Usa il modello per valutare diversi schemi di parallelizzazione
2. Model partitioning
	1. Genera il partizionamento a seconda dell'obbiettivo seguendo il modello generato in precedenza
3. Deployment
	1. Le parti del modello sono depliate in funzioni serverless

Problema cold start risolto con ping periodici


### Fork-Join Paradigm ([[gillis-icdcs21.pdf#page=4&selection=73,0,73,46|gillis-icdcs21, pagina 4]])

> [!Quote] 
>A master function is triggered to run upon receiving an inference query. Following the computed partitioning scheme, the master asynchronously invokes multiple worker functions. Each worker computes a partition of the served model, returns the result to the master, and ends its execution. The master can also help to compute a partition if having sufficient memory, which can result in fewer workers and less cost. The master assembles the returned results from all workers into a full tensor, and may initiate more workers to continue parallelizing model execution.
[[gillis-icdcs21.pdf#page=4&selection=134,42,151,55|gillis-icdcs21, pagina 4]]

Vantaggi:
- Comunicazione solo master-worker
	- Comunicazione worker-worker non supportata in serverless o comunque non efficiente
- Anche il master è una funzione
	- Non ho una componente serverful (e.g VM) che coordina

### Coarse-Grained Parallelization

> [!Quote] 
> To reduce the communication overhead, Gillis instead performs coarse-grained parallelization: it combines multiple consecutive layers into a single group and parallelizes each group across serverless functions. All layers in a group are hence computed locally within a function
[[gillis-icdcs21.pdf#page=4&selection=184,10,196,44|gillis-icdcs21, pagina 4]]

Il raggruppamento è su tutti i tipi di livelli: ciò porta alla crescita dello spazio di ricerca. Per risolvere il problema:
- element-wise layers
	- Fusi con il livello weight-intensive precedente
- Branch
	- Uniti in unico branch
Il risultato è un modello lineare


Worker non comunicano $\rightarrow$ problema di indipendenza: le loro computazioni devono essere indipendenti tra loro per poter funzionare.

> [!Quote]
> To meet this requirement, we determine if two consecutive layers can be grouped based on the dependency of their input and output tensors. Specifically, given two layers, if their output tensors have a local response to the input along the same dimensions, they can be group-parallelized along those dimensions.
[[gillis-icdcs21.pdf#page=5&selection=55,2,70,11|gillis-icdcs21, pagina 5]]

In immagine [[gillis-icdcs21.pdf#page=5&selection=19,0,19,6|gillis-icdcs21, pagina 5]] si vede la cosa: il layer group è diviso in 4 parti in totale, quindi avrò 4 funzioni serverless che calcolona le parti collegate tra loro in successione.


> [!Quote] 
> While layer grouping reduces the communication overhead, grouping too many layers can be inefficient, especially for those with convolution-like operators. As these operators (e.g., convolution and pooling) map multiple input elements to a single output, parallelizing the output tensor results in an overlap in the input partitions (Fig. 2a). As more layers are grouped, more overlaps are added, causing more redundant computations in the intermediate layers. Also, as the layer group grows larger, its partition may not fit into the memory of a single function.
> [[gillis-icdcs21.pdf#page=5&selection=84,0,96,21|gillis-icdcs21, pagina 5]]
> ...
> Parallelizing a layer group across too many functions can also be inefficient, as it may incur significant synchronization overhead in function communications, undermining the ben- efits of parallelization.
> [[gillis-icdcs21.pdf#page=5&selection=97,0,100,24|gillis-icdcs21, pagina 5]]


## Model Partitioning

### Performance Model
Data una rete neurale, il suo runtime è preso considerando sommando le predizioni dei tempi di esecuzione dei livelli ottenute dal modello costruito (vedere prima).

> [!Quote] Ritardo di Comunicazione
> We profile it by transferring data of varying sizes through REST APIs. Recall that in the fork-join model, the master function initi- ates multiple workers, and the communication delay depends on the slowest connection. This is equivalent to predicting the maximum delay of n concurrent communications
> [[gillis-icdcs21.pdf#page=5&selection=189,48,198,25|gillis-icdcs21, pagina 5]]

### Latency-Optimal Partitioning
Vedere direttamente sul paper, senza riportare i calcoli


### Minimizing Cost with SLO Compliance
Vedere direttamente sil paper senza riportare i calcoli

