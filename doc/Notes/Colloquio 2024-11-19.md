## Risolvere il problema di MobileNetV3
 * Capito dove sta il problema: sta nel fatto che se si fanno delle Operazioni inline a queste operazioni non viene associato esplicitamente un livello
 * Riuscito a distribuire modello in modo più generico...
* TODO 
	* Gestire ordine con cui gli input vengono dati al livello in caso di operazioni non commutative
	* Vedere se continua a funzionare anche con modelli più complessi

## Profiling della comunicazione
* Quanti dati sono trasferiti tra le parti (dimensioni dei tensori) (diff tra size di TensorProto e dimensione del tensore)
* Protocollo utilizzato (vedere se usa eventuale compressione)

## Quantità di elaborazione per singoli livelli (capire se i FLOPS dipendono dall'architettura)

## Studiare distribuzione mista (concentrando la parte del device sulla parte iniziale della computazione)
* In generale migliorare la distribuzione del modello

## Componente di front-end che restituisce al client il risultato
* Disaccoppiamento client dall'interfaccia esposta dal server/registry

## Vedere di caricare solo i sottolivelli necessari in caso di distribuzione

## Vedere se si può far girare modello Torch su Tensorflow
* Provare formato ONNX