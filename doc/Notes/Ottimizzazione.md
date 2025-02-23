L'ottimizzazione viene modellata come problema di assegnamento di un grafo logico (il modello) su un grafo fisico (la rete).

Sia $G_M = (V_M, E_M)$ il grafo del modello e sia $G_N = (V_N, E_N)$ il grafo della rete. Quello che vogliamo è mappare ogni nodo del modello su un nodo della rete e mappare ogni arco del modello su un arco della rete. Nello specifico siano:
- $G_M$ un grafo diretto aciclico
- $G_N$ un grafo diretto. Supponiamo che all'interno dell'insieme $E_N$ siano presenti anche archi di tipo $(v, v)$ $\forall v \in V_N$ per modellare il passaggio di dai da un nodo a se stesso.

Sia dato un nodo $v \in V_M$. Definiamo per questo livello una tupla $\phi_v$ di caratteristiche del nodo; in particolare sia $\phi_v = (flops\_livello)$ .

Sia dato un nodo $v \in V_N$. Definiamo per questo livello una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia $\psi_v = (flops\_al\_secondo, consumo\_energetico\_calcolo, consumo\_energetico\_trasmissione)$.

Sia dato un arco $e \in E_M$. Definiamo per questo arco una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia $\eta_e = (size\_dati\_arco)$.

Sia dato un arco $e \in E_N$. Definiamo per questo arco una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia $\epsilon_e = (banda\_arco)$.

Definiamo le variabili:
- $x_{ik} \in \{0, 1\}$ tc   $x_{ik} = 1$ sse $i \in V_M$ assegnato a $k \in V_N$
- $y_{ab} \in \{0, 1\}$ tc  $y_{ab} = 1$ sse $a \in E_M$ assegnato a $b \in E_N$

Notare che per come è modellata la rete di server, i server potrebbero non essere connessi in una mesh: si potrebbe anche pensare a delle connessioni per cui, pensando ad una divisioni in livelli del continuum (edge, fog, cloud), ogni livello è connesso solo a nodi di livelli superiori e a nodi dello stesso livello, ovvero la computazione non può "tornare indietro" (a meno di non voler spostare la parte finale della computazione sul nodo iniziale per non dover ritrasmettere il risultato).

# Modello di Latenza

## Latenza di Calcolo
La latenza di calcolo di un livello su un nodo della rete la possiamo calcolare come segue; dato $k \in V_N$ abbiamo

$$t_k^{c} = \sum_{i \in V_M} \frac{\phi_{i}[0]}{\psi_k[0]} x_{ik}$$  Da cui, la latenza complessiva dovuta al calcolo su tutti i server è data da:
$$t^{c} = \sum_{k \in V_N} t_{k}^{c}$$


## Latenza di Trasmissione
La latenza di trasmissione dipende invece dalla somma delle latenze di trasmissione in funzione delle bande dei link fisici a cui i link logici sono collegati. 

$$t_{k}^{x} = \sum_{a = (i, j) \in E_M} \sum_{b \in E_N \wedge k == b[0]} \frac{\eta_a[0]}{\epsilon_e[0]}y_{ab}$$

Da cui la latenza totale dovuta alla trasmissione è data da:
$$t^x = \sum_{k \in V_N} t_{k}^{x}$$
## Latenza totale
La latenza totale è quindi data da
$$t = t^c + t^x$$
# Modello di Energia

## Energia di Calcolo
L'energia data dal calcolo per $k \in V_N$ è data da:
$$E_k^c = t_k^c * \psi_k[1]$$
Da cui l'energia complessiva data dal calcolo è data da
$$E^c = \sum_{k \in V_N} E_k^c$$


## Energia di Trasmissione
L'energia data dalla trasmissione per $k \in V_N$ è data da:
$$E_k^x = t_k^x * \psi_k[2]$$
Da cui l'energia complessiva data dal calcolo è data da
$$E^x = \sum_{k \in V_N} E_k^x$$
## Energia Totale
$$E = E^x + E^c$$


# Modello di Memoria


# Modello di Penalità di Quantizzazione


# Problema

In definitiva quindi il problema diventa.
Indichiamo con $V_I \subset V_M$ il sottoinsieme di nodi di input; indichiamo con $V_O \subset V_M$ il sottoinsieme di nodi di output

$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \alpha*t + \beta*E \\
\\
Vincoli:\\
\sum_{k \in V_N} x_{ik} = 1 \hspace{1cm}  \forall i \in V_M \\
\sum_{b \in E_N} y_{ab} = 1 \hspace{1cm} \forall a \in E_M \\
x_{ik} = \sum_{h \in V_N} y_{(i,j)(k,h)} \hspace{1cm} \forall a=(i,j) \in E_M, \forall k \in V_N \\
x_{jh} = \sum_{k \in V_N} y_{(i,j)(k,h)} \hspace{1cm} \forall a=(i,j) \in E_M, \forall h \in V_N \\
x_{i0} = 1 \hspace{1cm} \forall i \in V_I \\ 
x_{j0} = 1 \hspace{1cm} \forall j \in V_O \\
\\
x_{ik} \in \{0, 1\} \hspace{1cm} \forall i \in V_M, \forall k \in V_N \\
y_{ab} \in \{0, 1\} \hspace{1cm} \forall a \in E_M, \forall b \in E_N \\

\end{matrix}

$$
Analisi vincoli:
1. Il primo vincolo serve ad imporre che ogni nodo del modello sia mappato su esattamente un server
2. Il secondo vincolo serve ad imporre che ogni arco del modello sia mappato su esattamente un arco della rete di server
3. Il terzo e il quarto vincolo sono vincoli di flusso
	1. Il terzo dice che se $i \in V_M$ è assegnato a $k \in V_N$, allora i dati prodotti da $i$ sono inviati ai server che hanno i nodi successori di $i$
	2. Il quarto vincolo è simile al terzo ma per la ricezione
4. Il quinto vincolo impone che almeno i livelli di input si trovino sul server 0 che fa partire l'inferenza (per assunzione)
5. Il sesto vincolo impone che i livelli di output si trovino sul server 0 che fa partire l'inferenza (per assunzione)


# Test dell'Implementazione
Test del modello solo con latenza, escludendo l'energia

Parametri di test:
- Due server
	- Server_0 = (10)
	- Server_1 = (11)
- Link tra i server
	- Banda (bidirezionale) = 1000 (B/s)
- Modello
	- ResNet50, tagliato come in figura
	- Flops dei nodi tutti impostati a 1000
	- Size input assunta (1, 3, 448, 448)
- Si è aggiunto come vincolo aggiuntivo che anche il primo livello convoluzionale si trovi su server 0 (vedere dimensioni)

La normalizzazione dell'obiettivo è fatta min-max.

![[Schermata del 2025-02-22 09-43-29.png|Sotto modello di Test|200]]

Nota le dimensioni di out della tabella sono scorrette, fare riferimento alle proporzioni che stanno sotto la tabella.
![[Schermata del 2025-02-22 09-45-41.png|Dimensioni e Tempi di trasmissione]]
Come si vede la dimensioni dei dati dal primo nodo al secondo è molto piccola, quindi verrebbe trasferita immediatamente, spostando tutta la computazione sul server 1: per questo si vincola che anche quel nodo stia sul primo server.


Divisione ottenuta dall'ottimizzatore:

| ![[Schermata del 2025-02-22 09-47-59.png\|Server 0]] |
| ---------------------------------------------------- |
| ![[Schermata del 2025-02-22 09-48-33.png\|Server 1]] |
Come si vede dal risultato ottenuto la computazione rimane sul server 0 fino all'esecuzione della MaxPool, a seguito della quale si ottiene una dimensione dei dati per cui risulta conveniente spostare il calcolo sul server_1 più veloce piuttosto che rimanere sul server_0 più lento.