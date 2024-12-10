## Background and Motivation

### DL Inference Serving

> [!Quote]
> However, mobile and IoT devices are typically resource-constrained, incapable of completing DL-based inference on time with state-of-the-art DNNs. Furthermore, battery life is usually a big concern for these devices. Hence, DL inference tasks are often offloaded to more powerful computing platforms
[[Jellyfish.pdf#page=2&selection=51,37,56,36|Jellyfish, pagina 2]]

> [!Quote]
> Typically, inference requests with input data (e.g., an image) issued by mobile or IoT devices travel through a dynamic (wireless) network (e.g., WiFi or cellular) before they reach the edge server. As a result, the time left for computing (i.e., inference serving) on the server can experience significant variations due to the variable network time caused by the variable network performance (see Fig. 2). Consequently, the application SLO should be defined end-to-end, including both the network and compute time.
[[Jellyfish.pdf#page=2&selection=73,42,92,5|Jellyfish, pagina 2]]


### Adaptation Techniques for Inference Serving Systems
Due tecniche:
- DNN Adaptation.
	- Variare tra reti
		- Equivalenti funzionalmente
		- Diverse
			- Accuratezza
			- Latenza
	- Due modi:
		- Insieme di DNN con diverse profondità, ampiezze ecc
		- Esecuzione parziale
- Data Adaptation
	- Riduzione della dimensione del dato
		- Compressione di qualche tipo
	- Non sufficiente per raggiungere un SLO a livello end-to-end

### Limitation of Existing Approaches
> [!Quote]
> Existing works mostly focus on either data or DNN adaptation [20], [21], [23], [24]. When simply combined, they could produce misaligned adaptation decisions, leading to suboptimal performance.
[[Jellyfish.pdf#page=3&selection=14,0,17,56|Jellyfish, pagina 3]]


## Design
### Overview
> [!Quote]
>  Jellyfish supports multiple clients simultaneously, and its major components are located on the edge side. 
>  1. When the clients send the requests to the edge over the network, the dispatcher component takes the client-DNN mapping from the scheduler
>  2. distributes the requests to workers running the expected DNN. Each worker is a separate process (on one or more edge servers) holding some GPU resources to 
>  3. Serve inference requests with the batch size selected by the scheduler. 
>  4. The worker manager deploys DNNs (stored in the DNN zoo) to the workers following the DNN selection decision by the scheduler. 
>  5. The scheduler provides the intelligence of Jellyfish, where it takes the latency-accuracy profiles from the DNN zoo and the monitored information from the client daemon as input, and runs our scheduling algorithms periodically to decide the client-DNN mapping, DNN selection (adaptation), and batch size for each worker. 
>  6. Scheduler informs all the clients about the input size of their mapped DNNs to start sending new requests at that particular input size (i.e., data adaptation aligned with DNN adaptation).
[[Jellyfish.pdf#page=3&selection=112,2,161,21|Jellyfish, pagina 3]]

> [!Quote]
> The end-to-end latency consists of two parts: network time (request and response) and compute time on the edge (for request dispatching and handling, request preprocessing if any, queuing, and DNN execution).
[[Jellyfish.pdf#page=3&selection=172,0,175,28|Jellyfish, pagina 3]]

### Components
Dispatcher:
- Distribuisce richieste ai worker corrispondenti
- Vede quale è il mapping cliet-dnn dallo scheduler (per capire il worker corrispondente)
- Endpoint per gestione client connessi

Worker:
- Esegue le richieste accodate
- Drop di richieste vecchie

Worker Manager:
- Deploy delle DNN sui worker
- Carica DNN dal DNN-Zoo e carica sul Worker
- Gestisce preloading delle DNN per alleviare "cold-start"

Scheduler:
- Decisioni di Adattamento
- Obbiettivi
	- Massima Accuratezza
	- Rispetto SLO
- Informazioni gestite
	- Stato Client
	- Edge State
		- Deployed DNN
		- Mapping Client-DNN
	- DNN Disponibili in DNN-Zoo

DNN-Zoo:
- Mantiene diverse versioni delle DNN
	- Approccio Bag-Of-Model

Client Daemon:
- Eseguito da Client
- Raccolta statistiche locali


## Scheduling Algorithm

### Formulation

> [!Quote] 
> The scheduling problem of Jellyfish aims to find the optimal multiset of DNNs to be deployed on the workers, the client- DNN mapping, and the batch size for each worker, so as to maximize the expected accuracy of all served inference requests.
> [[Jellyfish.pdf#page=4&selection=260,0,274,64|Jellyfish, pagina 4]]


> [!Quote] 
> To handle the complexity, we propose to tackle the problem by splitting it into two sub-problems: (1) client-DNN mapping and (2) DNN selection. We optimize each sub-problem iteratively to improve the overall accuracy objective without violating the latency SLO constraint
> [[Jellyfish.pdf#page=5&selection=344,2,348,54|Jellyfish, pagina 5]]


### Client-DNN Mapping


### DNN Selection


### DNN Update


