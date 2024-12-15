## Conversioni

Si può convertire direttamente un modello Keras in un modello LiteRT usando il converter di TF. Il problema è che la conversionde del modello cos' com'è solleva un errore in conversione: questo è dovuto al fatto che non tutte le operazioni TF sono supportate da TFLite (ad esempio i Conv2D non sono supportati).

| Errore di Compatibilità delle Operazioni   |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 09-29-28.png]] |

Per supportare queste operazioni si può abilitare l'integrazione della definizione delle operazioni non supportate in TFLite; riferimenti a:
- Integrazione altre operazioni
	- https://ai.google.dev/edge/litert/models/ops_select
	-  https://ai.google.dev/edge/litert/models/op_select_allowlist.md
- Operatori supportati nativamente e relazioni con altri operatori
	- https://ai.google.dev/edge/litert/models/ops_compatibility.md

Con il blocco di codice seguente si possono integrare le operazioni TF nel modello LiteRT.
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```


> [!Warning] Dimensione modello
> Capire se solo le operazioni mancanti sono aggiunte alla definizione del modello o se lo sono tutte (dal nome si parla di SELECT, quindi probabilmente solo quelle che mancano sono inserite).


Dopo questa integrazione il modello viene creato correttamente, ma c'è ancora un warning che viene sollevato sulla mancanza di un'operazione, in particolare su FlexConv2D.

| Warning su FlexConv2D                      |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 09-56-31.png]] |

### Segnature del modello
Per quanto riguarda le output signature sono mantenute uguali a quelle del modello keras:

| Output Signature uguale a quella di Keras  |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 18-54-10.png]] |


Il problema sono le input signature: la chiave non è presa considerando il nome dato all'input del dizionario all'atto della creazione in keras, ma il nome del livello di input!! Bisogna cambiare il nome del livello all'atto della conversione del modello...

| Input Signature con stesso nome livello di input - CLONE in Append |
| ------------------------------------------------------------------ |
| ![[Schermata del 2024-12-14 18-56-46.png]]                         |

Una volta aggiunto il blocco per il cambio nome come riportato in [[Split del Modello]] il problema si risolve: questo permette di raccordare l'output dato da un sotto modello in TFLite e da un sotto modello eseguito direttamente in Keras.

| Input Signature con stesso nome livello di input - NO CLONE in Append |
| --------------------------------------------------------------------- |
| ![[Schermata del 2024-12-14 19-14-19.png]]                            |

Il problema sull'input rimane per il primo input al modello...

## Inferenza
Sembra esserci un'incompatibilità tra versioni di numpy usate se non se ne specifica una:

| Incompatibilità tra versioni di Numpy      |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 16-39-10.png]] |
Per risolvere il problema facciamo il downgrade ad una versione di numpy precedente alla 2.0 eseguendo; `pip install "numpy<2"`. Questo sembra garantire il funzionamento sia di questo che delle altre cose implementate.

La SignatureRunner di LiteRT non accetta parametri senza che gli sia associato nome:

| Errore sugli argomenti                     |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 16-47-24.png]] |

`SignatureRunner.get_input_details()` permette di ottenere info sui parametri che il runner si aspetta, tra cui il nome di questo parametro.


Gli output ottenuti con Yolo sembrano diversi, anche se la norma della differenza è abbastanza bassa; il motivo potrebbe essere dovuto alle conversioni fatte internamente quando viene costruito il nuovo modello TFLite.

| Output diversi su Yolo                     |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 17-45-05.png]] |

Stessa cosa succede con MobileNetV3Large: la differenza è molto piccola e la classe predetta è la stessa.

| Output Diversi su MobileNetV3Large         |
| ------------------------------------------ |
| ![[Schermata del 2024-12-14 18-16-21.png]] |

L'uso di `allocate_tensor()` sembra cambiare l'errore sui tensori prodotti in output: in figura l'esempio con MobileNetV3Large.

| Senza allocate_tensor   | ![[Schermata del 2024-12-14 18-24-21.png]] |
| ----------------------- | ------------------------------------------ |
| **Con allocate_tensor** | ![[Schermata del 2024-12-14 18-25-04.png]] |


