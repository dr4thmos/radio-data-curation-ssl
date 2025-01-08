# radio-data-curation-ssl

Note

Manca:
url: https://lofar-surveys.org/public/DR2/mosaics/P004+26/fits_headers.tar per error 404
P000+38 da controllare che da TypeError: buffer is too small for requested array


- Dobbiamo finire di scaricar tutte le mappe
- Dobbiamo tentare la data curation
- Dobbiamo recuperare metodo di estrazione feature di bologna wavelet (Paolo Campeti) INFN ferrara
    - (scrivergli perchè non ho accesso alle slides) https://github.com/bregaldo/pywph/tree/main/examples
- Dobbiamo trainare i modelli
- Valutare inserimento di immagini simulate
- https://owncloud.ia2.inaf.it/index.php/s/IbFPlCCcPUresrr
- Valutare inserimento di immagini di sorgenti diffuse proposte da chiara stuardi
- Salvare o calcolare i neighbors di ogni patch
    - per il ritaglio sequenziale i vicini, #anche in diagonale?
    - per il ritaglio a maschera le 3 maschere sovrapposte + tutte quelle che hanno un overlap del 50% (comprese quelle sequenziali)
- Ci si aspetta che le immagini sovrapposte vengano perlopiù sottocampionate dalla data curation
    - Avere questa informazione salvata nel json ci permette di estrarre statistiche sul sottocampionemento
    - Statistiche su immagini vicine e immagini sovrapposte (maschere di dimensione crescente)
- 

