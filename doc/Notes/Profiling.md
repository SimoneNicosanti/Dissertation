Facciamo il profiling livello per livello (più flessibile e semplice da gestire).

Per fare il profiling del livello quantizzato si gioca sui nomi degli input/output:
- Agli input bisogna aggiungere un "\_QuantizeLinear\_Output"
- Agli output bisogna aggiungere un "\_QuantizeLinear\_Output"

In questo modo il modello si può quantizzare una sola volta e non per ogni singolo livello (cosa che lo renderebbe abbastanza scomodo e lungo, soprattuto per modelli molto grandi).

NON MI CONVIENE COSì!! Sto sul server e la quantizzazione potrebbe richiedere molta memoria per essere fatta: se il server non è in grado di gestire la cosa rischio rottura del programma. Quantizzo livello per livello