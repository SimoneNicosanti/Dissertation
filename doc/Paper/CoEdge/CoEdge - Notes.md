## Introduzione

Computazione Locale:
- Pro
	- Dati Locali
		- Privacy
		- Non serve la Rete
- Contro
	- Capacità calcolo limitata

Condivisione carico computazionale tra dispositivi vicini
- Problemi
	- Ripartizione considerando l'eterogeneità dei diversi vicini
	- Ottimizzazione considerando la dinamica della rete
	- Orchestrazione della comunicazione in fase di inferenza

Esempio: inferenza in CNN.
L'inferenza nelle CNN è divisa in due fasi di solito:
- Feature Extraction. Fatta con livelli convoluzionali e simili
- Classificazione. Fatta con livelli completamente connessi
Supponendo di avere due dispositivi organizzati in un paradigma master-worker possiamo:
1. Dividere l'input
2. Fare feature extraction in parallelo su porzioni diverse degli input
3. Aggregare le feature estratte
4. Classificare nel master

La metriche che possiamo tenere in considerazione per valutare le soluzioni sono:
- Latenza. Tempo tra input e output
- Energia

## CoEdge
### Architettura
Master-Worker:
- Master
	- Registry
	- Produce un piano di partizionamento
	- Gestisce la cooperazione e la parte di predizione
- Worker
	- Gli viene delegata parte dell'inferenza

### Fasi di Elaborazione
Due fasi:
1. Setup
	1. Il worker trova i suoi parametri
2. Runtime
	1. Master raccoglie i parametri dai worker
	2. Master produce il piano di elaborazione
	3. Avvio dell'elaborazione distribuita

### Parallelismo
Adotta il parallelismo di modello:
1. Vengono divisi i parametri tra i diversi nodi
2. Viene diviso l'input in parti e ogni parte viene data al nodo rispettivo
3. Il nodo produce la feature map relativa alla sua parte
4. Concatenazione delle feature map prodotto
5. Predizione in UNICO nodo

| Divisione del modello tra i vari nodi      |
| ------------------------------------------ |
| ![[Schermata del 2024-11-30 15-27-21.png]] |

La divisione fatta in questo modo può portare ad un problema. La convoluzione fatta ai bordi della parte assegnata ad un nodo può richiedere dati che sono stati assegnati ad un altro nodo. Nell'immagine ad esempio per calcolare la convoluzione centrata sull'ultima riga dei dati di A, servono i dati della prima riga di B.

| Dati necessari da altro nodo               |
| ------------------------------------------ |
| ![[Schermata del 2024-11-30 15-29-15.png]] |
Per generalizzare questo aspetto, dato un kernel $k * k$, servono $\frac{k}{2}$ righe di altri nodi.
Si può andare su molti nodi se:
- k grande
- Dimensione degli split dati ai diversi nodi è piccola
Per evitare si IMPONE che $splitSize \geq paddingSize$: in questo modo dato un dispositivo, i dati mancanti per calcolare la convoluzione, se servono, si troveranno solo in un altro dispositivo.

### Algoritmo di Partizionamento
Formulazioni:
- Vincolo Latenza
- Minimizzazione energia

Assunzioni:
- Dispositivi sempre Available
- Stabilità
- Modello pre caricato

Notazioni:
- L livelli
- N nodi
- $a_i$ righe di input al device i
- $\pi = (a_1, a_2, ..., a_N)$ partizionamento sui nodi
- $r_{li}$ carico nodo i-esimo sul livello l-esimo
- $(k, c_{in}, c_{out}, s, p)_{li}$ operazione che i-esimo nodo fa su l-esimo livello
	- k = size kernel
	- s = stride
	- p = padding
- $(\rho, f, m, p^c, p^x)_{i}$ profilo delle risorse del dispositivo i-esimo
	- $\rho$ è intensità di calcolo
	- $f$ è frequenza di calcolo
	- m è memoria del dispositivo disponibile al calcolo
	- $p^c$ è la potenza di calcolo
	- $p^x$ è la potenza di trasmissione

VEDERE DIRETTAMENTE ARTICOLO SU QUESTA PARTE.



> [!NOTE] Cose di interesse
> - Architetura master-worker
> - Algoritmo divisione del lavoro
> 	- PLI con calcolo dell'energia basato sulle capacità del dispositivo
> 		- Obiettivo minimizzazione energia
> 		- Vincolo latenza inferiore ad un QoS
> - Parallelismo di modello con
> 	- Feature Extraction in parallelo
> 	- Predizione in unico nodo

