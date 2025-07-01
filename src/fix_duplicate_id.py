import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil
import tempfile

def copia_run_con_nuovo_id(run_id_originale, experiment_id=None, run_name=None):
    """
    Copia una run MLflow esistente in una nuova run con un nuovo ID.
    
    Args:
        run_id_originale: ID della run da copiare
        experiment_id: ID dell'esperimento in cui creare la nuova run (se None, usa lo stesso della run originale)
        run_name: Nome per la nuova run (se None, usa 'Copia di {nome_originale}')
    
    Returns:
        ID della nuova run creata
    """
    client = MlflowClient()
    
    # Ottieni la run originale
    run_originale = client.get_run(run_id_originale)
    
    # Usa lo stesso experiment_id se non specificato
    if experiment_id is None:
        experiment_id = run_originale.info.experiment_id
    
    # Prepara il nome della nuova run
    if run_name is None:
        nome_originale = run_originale.data.tags.get("mlflow.runName", run_id_originale)
        run_name = f"Copia di {nome_originale}"
    
    # Crea una nuova run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as nuova_run:
        nuovo_run_id = nuova_run.info.run_id
        
        # Copia parametri
        for key, value in run_originale.data.params.items():
            mlflow.log_param(key, value)
        
        # Copia metriche - versione corretta
        # Ottieni tutte le metriche disponibili
        metriche = run_originale.data.metrics
        for key, ultimo_valore in metriche.items():
            # Per ogni metrica, ottieni la sua storia completa
            for metrica in client.get_metric_history(run_id_originale, key):
                mlflow.log_metric(
                    key=metrica.key, 
                    value=metrica.value, 
                    step=metrica.step,
                    timestamp=metrica.timestamp
                )
        
        # Copia tag (escludendo i tag di sistema)
        for key, value in run_originale.data.tags.items():
            # Salta i tag di sistema che iniziano con "mlflow."
            if not key.startswith("mlflow."):
                mlflow.set_tag(key, value)
        
        # Aggiungi un tag per indicare che è una copia
        mlflow.set_tag("copied_from_run_id", run_id_originale)
        
        # Copia artefatti
        temp_dir = tempfile.mkdtemp()
        try:
            # Scarica gli artefatti originali
            artifacts_uri = run_originale.info.artifact_uri
            local_path = artifacts_uri
            
            # Se l'URI è remoto, scarica gli artefatti localmente
            if artifacts_uri.startswith("http://") or artifacts_uri.startswith("https://") or artifacts_uri.startswith("s3://"):
                local_path = os.path.join(temp_dir, "artifacts")
                client.download_artifacts(run_id_originale, "", local_path)
            elif artifacts_uri.startswith("file:"):
                local_path = artifacts_uri[5:]  # Rimuovi "file:"
            
            # Carica gli artefatti nella nuova run
            if os.path.exists(local_path) and os.path.isdir(local_path):
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, local_path)
                        mlflow.log_artifact(file_path, os.path.dirname(relative_path))
        finally:
            # Pulisci la directory temporanea
            shutil.rmtree(temp_dir)
    
    print(f"Run copiata con successo da {run_id_originale} a {nuovo_run_id}")
    return nuovo_run_id

# Esempio di utilizzo
if __name__ == "__main__":
    # Imposta l'URI di tracking di MLflow (modifica con il tuo URI)
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # ID della run duplicata che vuoi copiare
    run_id_originale = "c6e73956a66a49f98117157279342b47"  # sostituisci con l'ID della run duplicata
    
    # Copia la run
    nuovo_run_id = copia_run_con_nuovo_id(
        run_id_originale=run_id_originale,
        experiment_id="193139885113803519",  # usa lo stesso esperimento della run originale
        run_name="extract_features"
    )
    
    print(f"Nuova run creata con ID: {nuovo_run_id}")
    print(f"Puoi visualizzare questa run all'URL: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_run(nuovo_run_id).info.experiment_id}/runs/{nuovo_run_id}")