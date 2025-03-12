
La divisione del modello potrebbe creare delle dipendenze cicliche. Per risolvere questi problemi si costruiscono delle componenti connesse per ogni server in modo tale che non vi siano delle dipendenze cicliche. Passi dell'algoritmo:
1. Ordinamento topologico del Grafo
2. Per ogni server
	1. Inizializza a zero la componente corrente di quel server (in un dict)
3. Inizializza
	1. DependencyMap NodeId -> set(CompId)
		1. Mappa di componenti da cui il nodo è dipendente
	2. PossibleComponentsMap: NodeId -> set(CompId)
		2. Mappa di possibili componenti a cui il nodo può appartenere
	3. ComponSizeCount: CompId -> int
		1. Mappa che dice la size corrente di ogni componente
4. Per ogni nodo nell'ordinamento topologico
	1. Se node.dependency_set $\cap$ node.possible_components $= \phi$ 
		1. Non ci sono dipendenze tra le componenti predecessore e quelle a cui il nodo può appartenere
		2. Se node.possible_components è vuoto
			1. GenerateComponentId
		3. Altrimenti
			1. Tra le componenti possibili prendi quella a size maggiore
	2. Altrimenti c'è dipendenza
		1. Non possiamo assegnare nessuna delle componenti_possibili al nodo
		2. GenerateComponentId
	3. Imposta node.comp = node_comp e incrementa size componente
	4. Per ogni nodo vicino a node
		1. if node.server == next_node.server
			1. next_node.poss_components.add(node.comp)
				1. Questo nodo potrebbe far parte della componente connessa del nodo corrente
		2. else
			1. next_node.dependency_set.add(node.comp)
				1. Il next_node dipende da questa componente del server corrente
		3. next_node.dependency_set.extend(node.dep_set)
			1. Propaga le dipendenze del nodo corrente a tutti i nodi successori

GenerateComponentId
1. Prendi l'indice della componente corrente per il server di appartenenza
2. Se questa componente è indipendenta dal nodo corrente
	1. Usa questa componente
	2. Questo serve per la gestione di rami paralleli in modo da ridurre il numero complessivo di componenti
3. Altrimenti
	1. Incrementa l'indice della componente attuale per il server
	2. Usa la nuova componente (questa componente è stata creata ora per la prima volta, quindi nessuno può dipendere da lei).



Nota aggiuntiva:
I nodi di input ed output vengono gestiti in modo tale da fare sempre parte di componenti separate: in questo modo si semplifica la gestione del pre e del post processing. I gestori di queste componenti all'interno della *do_inference* non faranno altro che le operazioni di pre e post processing più eventualmente il salvataggio dell'output.

A livello di FrontEnd questo non deve fare altro che connettersi su localhost al servizio ed inviare l'input alla componente del piano. Eventuali parametri per il pre e il post processing possono essere letti da un file di configurazione. Questa cosa permette anche la modellazione del passaggio di dati di pre-processing (come quelli generati per la segmentazione tra generator node ed input node).