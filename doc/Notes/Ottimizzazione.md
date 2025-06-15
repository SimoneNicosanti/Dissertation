L'ottimizzazione viene modellata come problema di assegnamento di un grafo logico (il modello) su un grafo fisico (la rete).

Sia $G_M = (V_M, E_M)$ il grafo del modello e sia $G_N = (V_N, E_N)$ il grafo della rete. Quello che vogliamo è mappare ogni nodo del modello su un nodo della rete e mappare ogni arco del modello su un arco della rete. Nello specifico siano:
- $G_M$ un grafo diretto aciclico
- $G_N$ un grafo diretto. Supponiamo che all'interno dell'insieme $E_N$ siano presenti anche archi di tipo $(v, v)$ $\forall v \in V_N$ per modellare il passaggio di dati da un nodo a se stesso.

Sia dato un nodo $v \in V_M$. Definiamo per questo livello una tupla $\phi_v$ di caratteristiche del nodo; in particolare sia $\phi_v = (flops\_livello)$ .

Sia dato un nodo $v \in V_N$. Definiamo per questo livello una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia $\psi_v = (flops\_al\_secondo, consumo\_energetico\_calcolo, consumo\_energetico\_trasmissione)$.

Sia dato un arco $e \in E_M$. Definiamo per questo arco una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia $\eta_e = (size\_dati\_arco)$.

Sia dato un arco $e \in E_N$. Definiamo per questo arco una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia $\epsilon_e = (banda\_arco)$.

Definiamo le variabili:
- $x_{ik} \in \{0, 1\}$ tc   $x_{ik} = 1$ sse $i \in V_M$ assegnato a $k \in V_N$
- $y_{ab} \in \{0, 1\}$ tc  $y_{ab} = 1$ sse $a \in E_M$ assegnato a $b \in E_N$

Notare che per come è modellata la rete di server, i server potrebbero non essere connessi in una mesh: si potrebbe anche pensare a delle connessioni per cui, pensando ad una divisioni in livelli del continuum (edge, fog, cloud), ogni livello è connesso solo a nodi di livelli superiori e a nodi dello stesso livello, ovvero la computazione non può "tornare indietro" (a meno di non voler spostare la parte finale della computazione sul nodo iniziale per non dover ritrasmettere il risultato).


> [!TODO] 
> Controllare normalizzazione per parte di energia e parte di latenza.
> In teoria devo escludere le variabili dalla normalizzazione, perché altrimenti il problema è circolare: dovrei prendere il massimo come se le assegnazioni fossero fatte sembre.


# Modello di Latenza

## Latenza di Calcolo
La latenza di calcolo di un livello su un nodo della rete la possiamo calcolare come segue; dato $k \in V_N$ abbiamo

$$t_k^{c} = \sum_{i \in V_M} \frac{\phi_{i0}}{\psi_{k0}} x_{ik}$$

Normalizzando otteniamo:
$$\hat{t}_k^{c} = \frac{t_k^c}{\max_{h \in V_N} t_h^c} $$

Da cui, la latenza complessiva dovuta al calcolo su tutti i server è data da:
$$T^{c} = \sum_{k \in V_N} \hat{t}_{k}^{c}$$


## Latenza di Trasmissione
La latenza di trasmissione dipende invece dalla somma delle latenze di trasmissione in funzione delle bande dei link fisici a cui i link logici sono collegati. 

$$t_{k}^{x} = \sum_{a = (i, j) \in E_M} \hspace{0.1cm} \sum_{b \in E_N \wedge k == b[0]} \frac{\eta_{a0}}{\epsilon_{e0}}y_{ab}$$

Normalizzando otteniamo:
$$\hat{t}_k^{x} = \frac{t_k^x}{\max_{h \in V_N} t_h^x} $$

Da cui la latenza totale dovuta alla trasmissione è data da:
$$T^x = \sum_{k \in V_N} \hat{t}_{k}^{x}$$

## Latenza totale
La latenza totale è quindi data da
$$T = T^c + T^x$$
# Modello di Energia

## Energia di Calcolo
L'energia data dal calcolo per $k \in V_N$ è data da:
$$E_k^c = t_k^c * \psi_{k1}$$

Normalizzando:
$$\hat{E}_k^c = \frac{E_k^c}{\max_{h \in V_N} E_h^c}$$

Da cui l'energia complessiva data dal calcolo è data da
$$E^c = \sum_{k \in V_N} \hat{E}_k^c$$


## Energia di Trasmissione
L'energia data dalla trasmissione per $k \in V_N$ è data da:
$$E_k^x = t_k^x * \psi_{k2}$$
Normalizzando:
$$\hat{E}_k^x = \frac{E_k^x}{\max_{h \in V_N} E_h^x}$$

Da cui l'energia complessiva data dal calcolo è data da
$$E^x = \sum_{k \in V_N} \hat{E}_k^x$$
## Energia Totale
$$E = E^x + E^c$$


# Modello di Memoria


# Modello di Penalità di Quantizzazione


# Problema
Indichiamo con:
- $V_I \subset V_M$ il sottoinsieme di nodi di input; 
- $V_O \subset V_M$ il sottoinsieme di nodi di output

Siano:
- $\alpha_c$ e $\alpha_x$ i pesi della latenza di calcolo e di trasmissione
- $\beta_c$ e $\beta_x$ i pesi dell'energia di calcolo e di trasmissione
Tali che:
$$\sum_{i \in \{x,c\}} \alpha_i + \sum_{i \in \{x,c\}} \beta_i = 1$$

In definitiva il problema diventa il seguente:
$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \alpha_c*T^c + \alpha_x*T^x + \beta_c*E^c + \beta_x*E^x \\
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

^91b978

Test del modello solo con latenza, escludendo l'energia

Parametri di test:
- Due server
	- Server_0 = (flops = 10)
	- Server_1 = (flops = 11)
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

## Astrazione del Modello
Sia il modello sia la rete su cui questo viene mandato in deployment vengono modellati come dei grafi.

All'interno del grafo del modello sono aggiunti due nodi fittizi, ovvero ModelInputNode e ModelOutputNode, che servono a semplificare la gestione della posizione dell'input e dell'output (input ed output si trovano sul server che avvia l'inferenza per default).

Allo stesso modo definiamo degli archi di input e di output sul modello che servono a trasportare l'input e l'output tra i vari sottomodelli (o da/verso il modello completo).



# Ottimizzazione 2.0

L'ottimizzazione viene modellata come problema di assegnamento di grafi logici (il modello e le sue varianti) su un grafo fisico (la rete di server).

Sia dato un task $T$ e sia $A$, l'insieme di varianti di un modello $M$ per la risoluzione del task; sia quindi $A = \{G_M^1, G_M^2, ..., G_M^{|A|}\}$ l'insieme delle varianti. Sia definito $A_{idx} = \{x : G_M^{x} \in A\}$ l'insieme degli indici delle varianti nell'insieme $A$.

Dato un $a \in A_{idx}$ definiamo $G_M^a = (V_M^a, E_M^a, \delta^a)$ il grafo del modello; nello specifico:
- La coppia $(V_M^a, E_M^a)$ rappresenta un DAG
- Il valore $\delta^a$ rappresenta l'accuratezza del modello $a$-esimo

Dato un $a \in A_{idx}$, definiamo:
- $\Lambda^a$ come il numero di richieste fatte al sistema nell'unità di tempo per la variante $a$-esima;
- $\Lambda = \sum_{a \in A_{idx}} \Lambda^a$ Il numero totale di richieste fatte al sistema 
- $\lambda^a = \frac{\Lambda^a}{\Lambda}$ Il tasso di richieste fatto per la variante $a$-esima

 Sia $G_N = (V_N, E_N)$ il grafo della rete. $G_N$ è un grafo diretto tale per cui supponiamo che all'interno dell'insieme $E_N$ siano presenti anche archi di tipo $(v, v)$ $\forall v \in V_N$ per modellare il passaggio di dati da un nodo a se stesso.

Quello che vogliamo è: $\forall a \in A_{idx}$ mappare ogni nodo e ogni arco di $G_M^a$ rispettivamente su un nodo e su un arco della rete $G_N$.

> [!NOTE] Connessioni di Rete
> Notare che per come è modellata la rete di server, i server potrebbero non essere connessi in una mesh: si potrebbe anche pensare a delle connessioni per cui, pensando ad una divisioni in livelli del continuum (edge, fog, cloud), ogni livello è connesso solo a nodi di livelli superiori e a nodi dello stesso livello, ovvero la computazione non può "tornare indietro" (a meno di non voler spostare la parte finale della computazione sul nodo iniziale per non dover ritrasmettere il risultato).



Sia dato un $G_M^a \in A$. Definiamo:
- Dato un nodo $v \in V_M^a$, una tupla $\phi_v$ di caratteristiche del nodo del modello; in particolare sia:
	- $\phi_{v0}$ i flops del livello
	- $\phi_{v1}$ la memoria dovuta ai pesi del livello
	- $\phi_{v2}$ la memoria dovuta all'output
- Dato un arco $e \in E_M^a$, una tupla $\eta_e$ di caratteristiche dell'arco del modello; in particolare sia:
	- $\eta_{e0}$ la size dei dati che caratterizzano l'arco

Consideriamo $G_N$. Definiamo:
- Dato un nodo $v \in V_N$, una tupla $\psi_v$ di caratteristiche del nodo della rete; in particolare sia:
	- $\psi_{v0}$ i flops al secondo calcolabili
	- $\psi_{v1}$ il consumo energetico dato dal calcolo
	- $\psi_{v2}$ il consumo energetico dato dalla trasmissione
	- $\psi_{v3}$ la memoria disponibile nel nodo
- Dato un arco $e \in E_N$, una tupla $\epsilon_e$ di caratteristiche dell'arco della rete; in particolare sia: 
	- $\epsilon_{e0}$ la banda di trasmissione dell'arco
	- $\epsilon_{e1}$ la latenza di trasmissione dell'arco

Dato una variante $G_M^a$ definiamo una tupla $\sigma^a$ di dimensione $|V_N|$, definita come:
- $\sigma_k^a$ rappresenta lo speedup di esecuzione del modello $a$-esimo sul server $k$-esimo
Lo speedup è inteso come speedup di un modello quantizzato rispetto al modello di base. Di conseguenza avremo:
- $\sigma_k^a = 1 \hspace{0.5cm} \forall k \in V_n$ se il modello $a$-esimo non è quantizzato
- $\sigma_k^a$ che può assumere un valore dipendente dalla profilazione del modello $a$-esimo quantizzato sul nodo $k$-esimo

Definiamo le variabili:
- $x_{ik}^a \in \{0, 1\}$ tc   $x_{ik}^a = 1$ sse $i \in V_M^a$ assegnato a $k \in V_N$
- $y_{mn}^a \in \{0, 1\}$ tc  $y_{mn}^a = 1$ sse $m \in E_M^a$ assegnato a $n \in E_N$

## Modello di Tempo

### Tempo di Calcolo
Il tempo di calcolo sul server $k$ per richieste fatte alla variante $a$-esima la possiamo scrivere come:
$$T_k^{c-a} =  \sum_{i \in V_M^a} \frac{\phi_{i0}}{\psi_{k0} * \sigma_k^a} x_{ik}^a$$
Definiamo il tempo massimo di calcolo tra i vari nodi della variante $a$-esima sul server $k$ come:
$$t_k^{c-a} = \max_{i \in V_M^a} \frac{\phi_{i0}}{\psi_{k_0} * \sigma_k^a}$$

Il tempo di calcolo totale per una singola richiesta per la variante $a$-esima è definito come:
$$T^{c-a} = \sum_{k \in V_N} T_k^{c-a}$$
Definiamo il massimo dei tempi di calcolo dei nodi della variante $a$-esima come :
$$t^{c-a} = \max_{k \in V_N} t^{c-a}_k$$

### Tempo di Trasmissione
La latenza di trasmissione dipende invece dalla somma dei tempi di trasmissione in funzione delle bande dei link fisici a cui i link logici sono assegnati. Per la variante $a$-esima sul server $k$:

$$T_{k}^{x-a} =  \sum_{m = (i, j) \in E_M^a} \hspace{0.1cm} \sum_{n \in E_N \wedge k == n[0]} \left( \frac{\eta_{m0}}{\epsilon_{n0}} + \epsilon_{n1} \right) * y_{mn}^a$$
Definiamo il tempo massimo per la trasmissione tra i nodi della variante $a$-esima come:
$$t^{x-a}_k = \max_{m \in E_M^a, n \in E_N \wedge k == n[0]} \left( \frac{\eta_{m0}}{\epsilon_{n0}} + \epsilon_{n1} \right)$$

Il tempo di trasmissione totale per singola richiesta sulla variante $a$-esima è dato da:
$$T^{x-a} = \sum_{k \in V_N} T_k^{x-a}$$
Il tempo di trasmissione massimo per singola richiesta ala variante $a$-esima è dato da:
$$t^{x-a} = \max_{k \in V_N} t_k^{x-a}$$

### Costo del tempo 
Definiamo $\mu_t^a = \max \{t^{x-a}, t^{c-a}\}$ il termine di normalizzazione per la latenza del modello $a$-esimo
Il costo dato dal tempo di inferenza del modello $a$-esimo è dato da:
$$C^a_t = \lambda^a \frac{T^{c-a} + T^{x-a}}{\mu_t^a}$$
Da cui il costo del tempo di inferenza complessivo è definito da:
$$C_t = \sum_{a \in A_{idx}} C_t^a$$

## Modello di Energia

### Energia di Calcolo
L'energia data dal calcolo della variante $a$-esima sul server $k$ è definita come:
$$E_k^{c-a} = T_k^{c-a} * \psi_{k1}$$
Il consumo energetico maggiore per la singola inferenza del modello $a$ sul server $k$ è dato da:
$$e_k^{c-a} = t_k^{c-a} * \psi_{k1}$$

Il consumo energetico complessivo dato dal calcolo della variante $a$-esima è dato da 
$$E^{c-a} = \sum_{k \in V_N} E_k^{c-a}$$
Mentre il consumo massimo è dato da:
$$e^{c-a} = \max_{k \in V_N} e_k^{c-a}$$


### Energia di Trasmissione
L'energia data dal calcolo della variante $a$-esima sul server $k$ è definita come:
$$E_k^{x-a} = T_k^{x-a} * \psi_{k2}$$
Il consumo energetico maggiore per la singola inferenza del modello $a$ sul server $k$ è dato da:
$$e_k^{x-a} = t_k^{x-a} * \psi_{k2}$$

Il consumo energetico complessivo dato dal calcolo della variante $a$-esima è dato da 
$$E^{x-a} = \sum_{k \in V_N} E_k^{x-a}$$
Mentre il consumo massimo è dato da:
$$e^{x-a} = \max_{k \in V_N} e_k^{x-a}$$


### Costo dell'energia
Definiamo $\mu_e^a = \max \{e^{x-a}, e^{c-a}\}$ il termine di normalizzazione per l'energia del modello $a$-esimo
Il costo di inferenza del modello $a$-esimo è dato da:
$$C^a_t = \lambda^a \frac{E^{c-a} + E^{x-a}}{\mu_e^a}$$
Da cui il costo dell'energia per l'inferenza è dato da:
$$C_t = \sum_{a \in A_{idx}} C_e^a$$
### Energia del device
Sia il device il server ad indice 0 in $V_N$.
La quantità di energia consumata da questo nodo di rete è data da:
$$E_0 = \sum_{a \in A_{idx}} \left( E_0^{x-a} + E_0^{c-a} \right)$$


## Modello di Memoria
Dato un livello $v \in V_M^a$, la memoria da lui occupata è data da $\phi_{v1} + \phi_{v2}$. Assunto che la memoria allocata per l'output sia riusata per l'output dei livelli successivi, la memoria totale allocata per ogni nodo $k$ per la variante $a$ è data da:
$$M_k^a = \sum_{i \in V_N^a} (\phi_{i1}*x_{ik}^a) + \max_{i \in V_N^a}(\phi_{i2} * x_{ik}^a) $$
Che però porta ad una formulazione non lineare del problema.
Definiamo quindi una variabile ausiliaria $m_k^a$, definita in maniera tale che:
$$m_k^a \ge \phi_{i2} * x_{ik}^a \hspace{0.75cm} \forall i \in V_N^a$$



Da ciò quindi la memoria usata sul server $k$ dalla variante $a$-esima diventa:
$$M_k^a = \sum_{i \in V_N^a} \left( \phi_{i1}*x_{ik}^a \right)  + m_k^a $$

Da cui, la memoria totale necessaria sul singolo server $k$ è data da:
$$M_k = \sum_{a \in A_{idx}}  M_k^a$$

> [!Note] Parallelismo
> Stiamo assumendo che per ogni variante, vi sia una sola copia attiva, quindi non c'è bisogno di moltiplicare per $\Lambda^a$ l'occupazione di memoria

## Formulazione
Indichiamo con:
- $V_I^a \subset V_M^a$ il sottoinsieme di nodi di input della variante $a$; 
- $V_O^a \subset V_M^a$ il sottoinsieme di nodi di output della variante $a$

Siano:
- $\omega_t$ il peso associato al tempo di inferenza
- $\omega_e$ il peso associato all'energia di inferenza

Sia $J_0$ l'energia massima consumabile sul device $0 \in V_N$.

Tali che:
$$ \omega_t + \omega_e = 1$$



In definitiva il problema diventa il seguente:
$$

\begin{matrix}

Obiettivo: \\
min \hspace{0.5cm} \omega_t*C_t + \omega_e*C_e\\
\\
Vincoli:\\
\sum_{k \in V_N} x_{ik}^a = 1 \hspace{1cm}  \forall a \in A_{idx}, \forall i \in V_M^a \\
\sum_{n \in E_N} y_{mn}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall m \in E_M^a \\
x_{i0}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall i \in V_I^a \\ 
x_{j0}^a = 1 \hspace{1cm} \forall a \in A_{idx}, \forall j \in V_O^a \\
\\

x_{ik}^a = \sum_{h \in V_N} y_{(i,j)(k,h)}^a \hspace{1cm} \forall a \in A_{idx}, \forall (i,j) \in E_M^a, \forall k \in V_N \\
x_{jh}^a = \sum_{k \in V_N} y_{(i,j)(k,h)}^a \hspace{1cm} \forall a \in A_{idx}, \forall (i,j) \in E_M^a, \forall h \in V_N \\
\\

M_k \le \psi_{k3} \hspace{1cm} \forall k \in V_N \\
m_k^a \ge \phi_{i2}^a * x_{ik}^a \hspace{0.75cm} \forall a \in A_{idx}, \forall i \in V_N^a \\
\\

E_0 \le J_0\\
\\

x_{ik}^a \in \{0, 1\} \hspace{1cm} \forall a \in A_{idx}, \forall i \in V_M^a, \forall k \in V_N \\
y_{mn}^a \in \{0, 1\} \hspace{1cm} \forall a \in A_{idx}, \forall m \in E_M^a, \forall n \in E_N \\

\end{matrix}

$$
Analisi vincoli:
1. Imporre che ogni nodo di ogni variante sia mappato su esattamente un server
2. Imporre che ogni arco di ogni variante sia mappato su esattamente un arco della rete di server
3. Imporre che almeno i livelli di input di ogni variante si trovino sul server 0 che fa partire l'inferenza (per assunzione)
4. Imporre che i livelli di output si trovino sul server 0 che fa partire l'inferenza (per assunzione)
5. Imporre rispetto flusso di invio: $i \in V_M^a$ è assegnato a $k \in V_N$, allora i dati prodotti da $i$ sono inviati ai server che hanno i nodi successori di $i$
6.  Imporre rispetto flusso di ricezione
7. Imporre il vincolo sulla memoria utilizzabile
8. Imporre che le variabili ausiliari per la memoria siano coerenti con il massimo
9. Imporre che il consumo energetico sul device non superi una soglia massima stabilita


> [!NOTE] Consumo energetico device
> Anziché aggiungere il vincolo sul consumo del device, si potrebbe ritornare il consumo calcolato per il device come risultato dell'ottimizzazione. A quel punto è il device che decide, in base al suo stato di carica, quale soluzione usare. Stessa cosa si potrebbe fare per la latenza complessiva delle richieste.

> [!TODO] Parallelismo
> Questa formulazione tiene conto del tempo di calcolo totale, ma non della latenza per l'inferenza di una singola richiesta perché questa dipende dall'eventuale parallelismo sul nodo del sistema. Si potrebbe assumere un tempo medio prendendo il numero di thread che eseguono richieste sul modello??
> 
> Il parallelismo alla fine lo gestiamo così:
> - Data una coppia (server, sotto-modello), abbiamo un thread che si occupa dell'inferenza per quel sotto-modello
> - Il parallelismo di Onnx (o di altri engine di inferenza) è assunto uno nella modellazione, non necessariamente negli esperimenti


> [!Warning] Gestione della memoria
> La gestione della memoria potrebbe essere più complicata di così: dato un sottomodello c'è bisogno di mantenere tutti i suoi input in memoria prima di far partire l'inferenza su quel sotto modello. Questa cosa va modellata in qualche modo, altrimenti si perde la coerenza sulla memoria. Di fatto anche l'aspetto di riuso della memoria potrebbe non essere completamente verosimile perché (potrebbero) esserci più nodi in esecuzione contemporaneamente (parallelismo interno all'engine ignorato in teoria).



# Ottimizzazione 3.0 - Con Quantizzazione

Consideriamo un unico modello (Si dovrebbe poter adattare facilmente al deployment di n modelli).

Siano dati:
- Un modello visto come un DAG logico $G_M = (V_M, E_M)$
- Un grafo di rete $G_N = (V_N, E_N)$; si tratta di un grafo diretto in cui è possibile la presenza di cappi, quindi di nodi $(v, v) \in E_N$ per rappresentare il passaggio di dati da un nodo a se stesso.
- $Q = \{0, 1\}$ ad indicare quindi se il modello è non quantizzato o quantizzato.

Il problema viene formulato come il mapping di un grafo logico (il dag del modello) sul grafo fisico (il grafo di rete).

Definiamo le seguenti variabili:
- $x_{ik} = 1$ se $i \in V_M$ assegnato a $k \in V_N$
- $y_{mn} = 1$ se $m \in E_M$ assegnato a $n \in E_N$
- $q_{i} = 1$ se $i \in V_M$ è quantizzato. Assumendo solo il caso di quantizzato vs non-quantizzato è sufficiente definire una solo variabile

## Modello di Tempo

### Tempo di Calcolo

Indichiamo con $f_k(i,q)$ un modello che ci dà una stima del tempo di calcolo del livello $i$ quantizzato con $q$ sul server $k$. Questa $f$ viene calcolata in fase iniziale quando il server inizia a partecipare al sistema.

Il tempo di esecuzione di un livello $i$ sul server $k$ è dato da:
$$
T^c_{ik} = f_k(i, 0)\cdot x_{ik} - (f_k(i,0) - f_k(i,1))\cdot x_{ik} \cdot q_i  = f_k(i, 0)\cdot x_{ik} - (f_k(i,0) - f_k(i,1))\cdot x_{ik}^q
$$

Notiamo che il termine tra parentesi rappresenta il guadagno di quantizzazione.

Dove la variabile $x_{ik}^q = x_{ik} \cdot q_i$ con vincoli:
1. $x_{ik}^q \le x_{ik}$
2. $x_{ik}^q \le q_i$
3. $x_{ik}^q \ge x_{ik} + q_i - 1$
Cioè la variabile assume valore 1 sse il livello $i$ si trova sul server $k$ ed è quantizzato.

Il tempo di calcolo del server $k$ è dato da:
$$
T_k^c = \sum_{i \in V_M} T_{ik}^c
$$

Il tempo di esecuzione del modello all'interno del sistema è dato quindi da:
$$
T^c = \sum_{k \in V_N} T_k^c
$$

### Tempo di Trasmissione

Indichiamo con $g(m,n,q)$ la funzione che stima il tempo di trasmissione dell'arco logico $m$ sull'arco fisico $n$ quando $m[0]$ ha uno stato di quantizzazione $q$ (i.e. quantizzato vs non-quantizzato). Nel nostro caso possiamo dire ad esempio che:

$$
base\_tx\_time(m,n) = \frac{m.dataSize}{n.bandWidth}
$$

$$
T_{ik}^x = base\_tx\_time(m,n) \cdot y_{mn} - \left(base\_tx\_time(m,n) - \frac{base\_tx\_time(m,n)}{8} \right) \cdot y_{mn}^q + n.latency
$$

Il tempo di trasmissione di un server $k$ è dato da:
$$T_{k}^x =  \sum_{m = (i, j) \in E_M^a} \hspace{0.1cm} \left( \sum_{n \in E_N \wedge  k = n[0] = n[1] } T_{ik}^x \hspace{0.5cm} + \sum_{n \in E_N \wedge  k = n[0] \wedge k \neq n[1] } T_{ik}^x \right) = \sum_{m = (i, j) \in E_M^a} \left( 
T_{ik}^{x-self} + T_{ik}^{x-oth}
\right)$$

Ovvero è dato dal contributo dei tempi di trasmissione ai livelli successori che si trovano sullo stesso server e su server diversi.

Dove la variabile $y_{mn}^q = y_{mn} \cdot q_i$ con vincoli seguenti:
- $y_{mn}^q \le y_{mn}$
- $y_{mn}^q \le q_{i}$
- $y_{mn}^q \ge y_{mn} + q_i -1$
In particolare abbiamo che $m[0] = i$

Il tempo di trasmissione totale per la richiesta è dato da:
$$
T^x = \sum_{k \in V_N} T_k^x
$$

### Costo del Tempo
Definito $\mu_T$ il termine di normalizzazione per il tempo, abbiamo che:

$$
T = T^c + T^x
$$

$$
C_T = \frac{T^c + T^x}{\mu_T}
$$


## Modello di Energia

### Energia di Calcolo

Sia $h_k(t)$ la funzione che stima l'energia usata per il calcolo di durata $t$ dal dispositivo $k$.

L'energia data dal calcolo sul server $k$ è definita come:
$$E_k^{c} = h_k(T_k^c)$$
Il consumo energetico complessivo dato dal calcolo  è dato da:
$$E^{c} = \sum_{k \in V_N} E_k^{c}$$

### Energia di Trasmissione

Sia $l_k(t)$ la funzione che stima l'energia usata per la trasmissione di durata $t$ dal dispositivo $k$.

L'energia data di trasmissione sul server $k$ è definita come:
$$E_k^{x} = l_k^{self}(T_k^{x-self}) + l_{k}^{oth}(T_{k}^{x-oth})$$

Quindi anche in questo caso abbiamo il contributo delle energie per trasmettere al server medesimo (questo dipenderà dal consumo sull'interfaccia di loopback --> memoria e potenzialmente trascurabile) e per trasmettere ad altri server (cosa che comporta l'attivazione dell'interfaccia di rete e il consumo energetico maggiore).

Il consumo energetico complessivo per la trasmissione è dato da 
$$E^{x} = \sum_{k \in V_N} E_k^{x}$$

### Costo dell'energia
Definiamo $\mu_E$ il termine di normalizzazione dell'energia; allora il costo dato dal consumo energetico è dato da:
$$
C_E = \frac{E^c + E^x}{\mu_E}
$$

$$
E = E^c + E^x
$$


### Energia del device
Sia il device il server ad indice 0 in $V_N$.
La quantità di energia consumata da questo nodo di rete è data da:
$$E_0 = E_0^{x} + E_0^{c}$$

## Modello di Memoria
Il modello di memoria più o meno è sempre uguale, al netto dell'effetto della quantizzazione sui pesi.

## Modello di Rumore
I problemi principali dell'uso della quantizzazione per livelli singoli è il seguente:
1. Non sappiamo come le quantizzazioni tra livelli interagiscono tra di loro
2. Ci sono molte possibili combinazioni di quantizzazione: se ogni livello della rete può essere quantizzato e non quantizzato, allora ci sono $2^{|V_M|}$ possibilità, troppo per indagarle tutte 
3. Il tempo per la valutazione di una certa combinazione di quantizzazione non è poco: per ogni combinazione bisogna, calibrare, quantizzare e poi valutare l'errore

Quello che si potrebbe fare è considerare un sottoinsieme di livelli della rete che siano significativi per la quantizzazione:
- Livelli che hanno un carico alto (alto numero di flops)
- Livelli che hanno un output grande in dimensione (ne beneficia il tempo di trasmissione)
E quantizzare SOLO questo sottoinsieme, imponendo che gli altri non lo siano.

Sia quindi $V_{Q} \subseteq V_M$ dove $V_Q = \{ i \in V_n : i.quantizable = True \}$ il sottoinsieme di possibili livelli quantizzabili.

Comunque anche con pochi livelli (ad esempio tra i 12 e 15), il numero di possibilità è troppo alto per poter essere precisi. Si può introdurre un modello di regressione tale che:
$$
e : \{0,1\}^{|V_Q|} \rightarrow \mathbb{R}^{\ge 0}
$$

Di base serve un modello di regressione che sia esprimibile con vincoli lineari: a questo scopo si potrebbe usare un albero o una semplice regressione lineare.

Alcune prove fanno vedere che (almeno per Yolo11n-seg) una regressione polinomiale funziona bene.

Supponiamo di usare una regressione polinomiale di grado $d$. Sia $q = (q_1, q_2, ..., q_{|V_Q|})$ il vettore delle variabili $q_i$ solo dei livelli quantizzabili: questa variabile ha valore 1 sse il livello in questione è quantizzato.

Definiamo $\hat{q}$ il vettore delle variabili prodotto. In particolare, un elemento in posizione $i$ di $\hat{q}$ rappresenta il prodotto di alcune variabili in $q$ ovvero $\hat{q}_k = \prod_{i \in V_{Q_k}} q_i$ ; per definire questa variabile possiamo definirne una equivalente soggetta ai seguenti vincoli:
1. $\hat{q}_k \le q_i$      $\forall i \in V_{Q_k}$
2. $\hat{q}_k \ge \sum_{i \in V_{Q_k}} q_i - (|V_{Q_k}| - 1)$ 

Definiamo una variabile di esistenza $p = 1$ sse $\exists i \in V_Q$  tc $q_i = 1$; soggetta ai vincoli:
- $p \ge q_i$   $\forall i \in V_Q$
- $p \le \sum_{i \in V_Q} q_i$

Da cui quindi il regressore è definito da:
$$
e(\hat{q}) = w^T \cdot \hat{q} + c \cdot p
$$

Definito questo regressore dipendente dalle variabili di quantizzazione, possiamo aggiungere al problema il vincolo:

$e(\hat{q}) \le e_{max}$

Dove $e_{max}$ rappresenta il rumore massimo che si è disposti a tollerare sull'output del modello.


