## Intro
Tre paradigmi di ML in dispositivi limitati:
- Local
	- Pro
		- Privacy
		- Indipendente dalla rete
	- Cons
		- Capacità calcolo limitate
			- Solo per task semplici a sufficienza
		- Consumo energetico maggiore
			- Attenuato con quantization del modello
- Edge
	- Pro
		- Maggiore capacità
			- Uso modelli più complessi e non quantizzati
	- Cons
		- Limitati rispetto cloud server
			- Thr minore rispetto a quello che si avrebbe con server
		- Richide rete
			- Stabile
			- Latenza in risposta
- Split. 
	- Prevede
		- divisione del modello
		- Pre-elaborazione dell'input (esempio, codifica)
			- Scopo
				- Riduzione dati da trasferire
				- Riduzione carico server
	- Pro
		- Gestione situazioni
			- Canali poco affidabili (trasferisco meno dati)
			- Scarse capacità calcolo

L'approccio di Split prevede che il modello sia diviso in modo da:
- Ridurre l'uso della rete
	- Si divide il modello in punti strategici in modo che la quantità di dati inviata sia bassa
		- Operazioni locali per
			- Preelaborazione
			- Encoding


Furcifer usa un approccio di online adaptation tra i tre paradigmi usando dei container alla base.
I pro di questi approcci sono:
- Monitoraggio real-time delle risorse nel continuum
- Switch real-time tra gli approcci usando i container
- Test con dataset sia indoor che outdoor


### Edge Computing


### Image Compression and Object Detection

Le prestazioni dei modelli si abbassano quando gli viene dato un input che è stato compresso con un algoritmo di compressione di immagini: serve quindi un compromesso tra prestazioni e dati trasferiti.
> [!Quote]
> these evaluations often overlook the significant performance degradation caused by image compression [15], [16], which is inevitable in practical EC systems. In partic- ular, widespread image compression techniques are designed for human perception rather than for image analysis. As a consequence, high performance requires the transfer of large volumes of data over capacity-constrained channel
[[Furcifer_PerCom24.pdf#page=2&selection=249,15,259,49|Furcifer_PerCom24, pagina 2]]

Per sopperire a questi problemi si può pensare di addestrare dei modelli sui dati già compressi o comunque su cui è stata fatta una preelaborazione volta ad estrarne le caratteristiche significative.
> [!Quote]
> these evaluations often overlook the significant performance degradation caused by image compression [15], [16], which is inevitable in practical EC systems. In partic- ular, widespread image compression techniques are designed for human perception rather than for image analysis. As a consequence, high performance requires the transfer of large volumes of data over capacity-constrained channel
[[Furcifer_PerCom24.pdf#page=2&selection=249,15,259,49|Furcifer_PerCom24, pagina 2]]

### Energy Consumption

> [!Quote]
>  In the domain of real-time computer vision, energy con- sumption is not solely determined by the number of Floating Point Operations (FLOPs) or Multiply-Accumulate (MAC) op- erations indicative of the model’s complexity. Indeed, energy consumption is also proportional to the number of frame per seconds (F P S) processed by the system [27], [28].
[[Furcifer_PerCom24.pdf#page=3&selection=44,0,51,37|Furcifer_PerCom24, pagina 3]]


> [!Warning] FLOPS Calcolati
> Controlla se quelli trovati sono FLOPS o MAC

> [!Quote] Understand the Definitions
> - A FLOP (Floating Point OPeration) is considered to be either an addition, subtraction, multiplication, or division operation.
>- A MAC (Multiply-ACCumulate) operation is essentially a multiplication followed by an addition, i.e., MAC = a * b + c. It counts as two FLOPs (one for multiplication and one for addition).
>
>https://medium.com/@pashashaik/a-guide-to-hand-calculating-flops-and-macs-fa5221ce5ccc

Nel minimizzare il consumo energetico si prendono in considerazione:
- Precisione Media
- FPS rate
>[!Quote]
>Furcifer aims to strike a balance between resource efficiency and predictive precision, spanning from the edge to the cloud, and catering to the comprehensive energy optimiza- tion needs of modern mobile computing. By minimizing the energy consumption of mobile devices based on the desired mean Average Precision (mAP ) score F P S rate, our solution represents a leap forward in realizing practical ubiquitous computer vision applications.
[[Furcifer_PerCom24.pdf#page=3&selection=60,13,74,29|Furcifer_PerCom24, pagina 3]]


## Problem Statement
Confronto strategie:
- Metriche: mAP = Mean Average Precision
	- Combina precision e recall basate su IoU
- Fattori da considerare
	- Risoluzione camera
	- Fattore di scala
	- Quantizzazione del Modello
	- Compressione dell'immagine

> [!Quote] 
> However, when deploying an OD engine in a real-world setting, various factors such as camera resolution or scaling factor alterations come into play to determine the performance perceived by the application. Additional factors such as model quantization and image compression also play a significant role
[[Furcifer_PerCom24.pdf#page=3&selection=217,0,234,4|Furcifer_PerCom24, pagina 3]]

## Design e Implementazione

> [!Quote] 
> the – time varying – state of the system, which is influenced by mobility and load dynamics, determines the best computing configura- tion
[[Furcifer_PerCom24.pdf#page=4&selection=208,49,211,4|Furcifer_PerCom24, pagina 4]]

Uso dei container permette di gestire facilmente le dipendenze dei modelli e permettergli di eseguire su dispositivi di diversa natura.
> [!Quote] 
> Furcifer realizes an adaptation engine composed of highly effective containerized models whose activation is determined by a control module informed by comprehensive system monitoring
[[Furcifer_PerCom24.pdf#page=4&selection=212,40,215,43|Furcifer_PerCom24, pagina 4]]

### Energon: Monitoring Energetico

> [!Quote] 
> Energon focuses primarily on energy consumption and resource utilization in M Ds, while also providing insights into additional metrics, including network quality, packet transmission and drop rates, CPU usage for individual cores, storage utilization, GPU usage percentage, and temperature measurements from various regions of the board. Scraped metrics are made available through an HTTP endpoint that can be queried on demand by the orchestrator.
> [[Furcifer_PerCom24.pdf#page=4&selection=247,0,281,45|Furcifer_PerCom24, pagina 4]]


### On Demand Image Pulling

> [!Quote] 
> We choose to apply containerization as a practical way to guarantee flexibility and fast reactiveness of the framework to future environment states. Our evaluation of the size of resulting container images reveals that less than 1% of the image comprises application-level files. This efficient design enables the download of only the last layer of the image, which has the same footprint size of the model itself. This approach minimizes network usage, since the model would anyway need to be transferred no matter whether a containerized approach is used or not.
[[Furcifer_PerCom24.pdf#page=5&selection=97,43,106,48|Furcifer_PerCom24, pagina 5]]

### Protocollo di Comunicazione
I container rappresentano degli endpoint (microservizi) nei dispositivi: questi microservizi interagiscono con l'orchestratore tramite REST API.

Fluttuazione della rete:
- Usato TCP per tracciare la latenza della rete
	- Cambio strategia se si rivela necessario
Messaggi:
- keep_alive
	- Presenza di dispositivi nella rete
- start/stop_OD
	- Per avviare interrompere object detection
	- Specifica anche la strategia (LC, EC, SC)
- set_target_frame_rate
	- Imposta i FPS della fotocamera basandosi su requisiti applicativi
- set_compression_rate
	- Imposta la compressione dell'immagine in caso di EC per ridurre uso della rete
	- Considerato trade-off compressione e mAP

### Orchestratore

> [!Quote] 
>  the orchestrator’s scaling and management protocols are controlled by the pareidolia policy. This policy evaluates the energy consumption metrics of the MD, as well as an array of context-sensitive metrics. This evaluation enables the orchestrator to dynamically adjust its operational strategy, seamlessly transitioning between two or more con- tainers to optimize performance and resource utilization
[[Furcifer_PerCom24.pdf#page=6&selection=25,12,31,56|Furcifer_PerCom24, pagina 6]]

### Engine

> [!Quote] 
> we use a modified version of the knowledge distillation process adopted in SC2 Benchmark [37] to design a compact encoder optimized for constrained devices. This encoder serves a dual purpose: minimizing channel occupancy and effectively distributing computation load between mobile devices and the edge server
[[Furcifer_PerCom24.pdf#page=6&selection=56,15,61,27|Furcifer_PerCom24, pagina 6]]

> [!Quote] 
> The optimized encoder heavily relies on quantization and channel compression to reduce execution time as much as possible. To enhance data compression, we strategically place a one-channel bottleneck in the initial layers of the feature extraction segment of the network. This choice leads to further data reduction, increasing the efficiency of the whole process. Additionally, we incorporate INT8 quantization at the end of the encoder. This quantization approach optimizes the representation of the data, contributing to both improved data compression and streamlined computation. 
> ...
>  These values are then communicated to the decoder located at the ES, along with the resulting INT8 tensor from the encoder inference process.
[[Furcifer_PerCom24.pdf#page=6&selection=90,0,123,18|Furcifer_PerCom24, pagina 6]]


### Camera Sampling

> [!Quote] 
> this module offers to the user the ability to set precise directives for the desired camera sampling rate and image resolution [41]. Such dynamic adjustments align with distinct embedded OD models stored within the container registry located on the ES.
[[Furcifer_PerCom24.pdf#page=6&selection=198,29,205,1|Furcifer_PerCom24, pagina 6]]


### Pareidolia

> [!Quote] 
> Each participating MD maintains a record of previously completed tasks. This historical context empowers the node to discern which computing strategy aligns best with the current system state by identifying analogous past scenarios. When a sufficiently similar context is detected, the ES intervention may not be required. Conversely, if an analogous context is not found, pertinent task details are shared with the ES to collaboratively determine the optimal model and computing configuration (EC, LC or SC) that best matches the current system state
[[Furcifer_PerCom24.pdf#page=6&selection=236,26,266,37|Furcifer_PerCom24, pagina 6]]
