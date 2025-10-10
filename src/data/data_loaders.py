"""
Questo modulo fornisce un'interfaccia robusta, type-safe e ben documentata
per caricare tutti gli artefatti di dati prodotti dalla pipeline.

Ogni Loader è una classe senza stato con metodi statici che restituisce
un Dataclass immutabile e validato, garantendo che i dati siano corretti
nel momento in cui vengono caricati.
"""

import json
import pathlib
from dataclasses import dataclass
import mlflow
import h5py
import numpy as np


# --- ECCEZIONE CUSTOM UNIFICATA ---
class SourceNotFoundError(Exception):
    """Sollevata quando una sorgente dati (file, etc.) non è trovabile, valida o completa."""

    pass


# --- DEFINIZIONE DEI DATACLASS ---


@dataclass(frozen=True)
class CutoutItem:
    """Rappresenta i metadati di un singolo cutout (una patch di immagine)."""

    id: int
    file_path: pathlib.Path
    survey: str
    mosaic_name: str
    position: tuple[int, int]
    size: int


@dataclass(frozen=True)
class CutoutListData:
    """Rappresenta una collezione di CutoutItem, caricata da un info.json."""

    cutouts: list[CutoutItem]

    def __post_init__(self):
        if not self.cutouts:
            raise ValueError("La lista dei cutout non può essere vuota.")


@dataclass(frozen=True)
class FeaturesData:
    """Contenitore immutabile per un array di features."""

    features: np.ndarray

    def __post_init__(self):
        if not isinstance(self.features, np.ndarray):
            raise TypeError("L'attributo 'features' deve essere un array NumPy.")
        if self.features.ndim != 2:
            raise ValueError("L'array 'features' deve essere bidimensionale.")
        if self.features.size == 0:
            raise ValueError("L'array 'features' non può essere vuoto.")


@dataclass(frozen=True)
class ImagePathsData:
    """Contenitore immutabile per una lista di percorsi di immagini."""

    paths: list[str]

    def __post_init__(self):
        if not self.paths:
            raise ValueError("La lista 'paths' non può essere vuota.")


@dataclass(frozen=True)
class ClusteringLevel:
    """Rappresenta i dati di un singolo livello di clustering."""

    level_index: int
    centroids: np.ndarray
    clusters: list[np.ndarray]

    def __post_init__(self):
        if self.centroids.ndim != 2 or self.centroids.size == 0:
            raise ValueError(
                f"L'array 'centroids' al livello {self.level_index} deve essere 2D e non vuoto."
            )
        if not self.clusters:
            raise ValueError(
                f"La lista 'clusters' al livello {self.level_index} non può essere vuota."
            )


@dataclass(frozen=True)
class HierarchicalClusteringData:
    """Rappresenta l'intera gerarchia di un risultato di clustering."""

    n_levels: int
    levels: list[ClusteringLevel]

    def __post_init__(self):
        if self.n_levels != len(self.levels):
            raise ValueError(
                "Il numero di livelli 'n_levels' non corrisponde al numero di oggetti ClusteringLevel forniti."
            )
        if not self.levels:
            raise ValueError("La lista dei livelli di clustering non può essere vuota.")


@dataclass(frozen=True)
class SampledIndicesData:
    """Rappresenta un set di indici selezionati da un processo di campionamento."""

    indices: np.ndarray

    def __post_init__(self):
        if self.indices.ndim != 1:
            raise ValueError("L'array 'indices' deve essere monodimensionale.")
        if not np.issubdtype(self.indices.dtype, np.integer):
            raise TypeError("Gli indici devono essere di tipo intero.")


# --- IMPLEMENTAZIONE DEI LOADER (LE "FABBRICHE" DI DATI) ---


class CutoutLoader:
    """Carica le liste di cutout dal formato info.json."""

    @staticmethod
    def from_json(file_path: pathlib.Path) -> CutoutListData:
        """Carica una lista di cutout da un file info.json."""
        try:
            with file_path.open("r") as f:
                raw_data = json.load(f)

            cutout_list = [
                CutoutItem(
                    id=int(idx),
                    file_path=pathlib.Path(item["file_path"]),
                    survey=item["survey"],
                    mosaic_name=item["mosaic_name"],
                    position=tuple(item["position"]),
                    size=item["size"],
                )
                for idx, item in raw_data.items()
            ]
            return CutoutListData(cutouts=cutout_list)
        except FileNotFoundError as e:
            raise SourceNotFoundError(f"File info.json non trovato: {file_path}") from e
        except (json.JSONDecodeError, KeyError) as e:
            raise SourceNotFoundError(
                f"File info.json malformattato o incompleto: {file_path}"
            ) from e


class FeaturesLoader:
    """Carica artefatti relativi alle features (array, percorsi) da file HDF5."""

    @staticmethod
    def load_features_from_path(file_path: pathlib.Path) -> FeaturesData:
        """Carica solo le features da un file .h5 o .npy."""
        try:
            if file_path.suffix == ".h5":
                with h5py.File(file_path, "r") as h5f:
                    features_array = h5f["features"][:]
            elif file_path.suffix == ".npy":
                features_array = np.load(file_path)
            else:
                raise ValueError(
                    f"Formato file non supportato per le features: {file_path.suffix}"
                )
            return FeaturesData(features=features_array)
        except (FileNotFoundError, KeyError) as e:
            raise SourceNotFoundError(
                f"Sorgente features non trovata o incompleta in {file_path}"
            ) from e

    @staticmethod
    def load_image_paths_from_h5(
        file_path: pathlib.Path, key: str = "image_paths"
    ) -> ImagePathsData:
        """Carica solo i percorsi delle immagini da un file HDF5."""
        try:
            if file_path.suffix != ".h5":
                raise ValueError(
                    "I percorsi delle immagini possono essere caricati solo da file .h5"
                )
            with h5py.File(file_path, "r") as h5f:
                paths = [p.decode("utf-8") for p in h5f[key][:]]
            return ImagePathsData(paths=paths)
        except (FileNotFoundError, KeyError) as e:
            raise SourceNotFoundError(
                f"Dataset '{key}' per i percorsi non trovato in {file_path}"
            ) from e


# Nel tuo dataloaders.py, SOSTITUISCI il vecchio ClusteringLoader con questo

class ClusteringLoader:
    """Carica i risultati del clustering gerarchico da una directory di output."""
    @staticmethod
    def from_directory(dir_path: pathlib.Path) -> HierarchicalClusteringData:
        """
        Carica l'intera gerarchia di clustering leggendo i metadati
        e assemblando i dati dalle sottodirectory 'levelN'.
        """
        try:
            # La logica per trovare il numero di livelli rimane la stessa
            metadata_path = dir_path / "metadata.json"
            if not metadata_path.exists():
                raise SourceNotFoundError(f"File 'metadata.json' non trovato in {dir_path}")

            with metadata_path.open('r') as f:
                metadata = json.load(f)
            
            n_levels = metadata.get("n_levels")
            if n_levels is None:
                raise SourceNotFoundError(f"Chiave 'n_levels' non trovata in {metadata_path}")

            loaded_levels = []
            for i in range(n_levels):
                # --- MODIFICA CHIAVE QUI ---
                # I livelli sono 1-indicizzati, il loop è 0-indicizzato.
                level_index = i + 1
                level_path = dir_path / f"level{level_index}"
                # -------------------------

                centroids_path = level_path / "centroids.npy"
                # Il nome del file dei cluster potrebbe essere diverso, assumiamo 'sorted_clusters.npy'
                # basandoci sullo script originale. Adatta se necessario.
                clusters_path = level_path / "sorted_clusters.npy" 

                if not level_path.exists() or not centroids_path.exists() or not clusters_path.exists():
                    raise SourceNotFoundError(f"Dati mancanti per il livello {level_index} nella directory {level_path}")

                centroids = np.load(centroids_path)
                clusters = np.load(clusters_path, allow_pickle=True)
                
                level_data = ClusteringLevel(
                    level_index=i, # Manteniamo l'indice logico 0-based
                    centroids=centroids,
                    clusters=list(clusters)
                )
                loaded_levels.append(level_data)
            
            return HierarchicalClusteringData(n_levels=n_levels, levels=loaded_levels)

        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise SourceNotFoundError(f"Impossibile caricare i dati di clustering da {dir_path}. Dettagli: {e}") from e


class SamplingLoader:
    """Carica gli indici campionati."""

    @staticmethod
    def from_npy(file_path: pathlib.Path) -> SampledIndicesData:
        """Carica un array di indici campionati da un file .npy."""
        try:
            indices = np.load(file_path)
            return SampledIndicesData(indices=indices)
        except FileNotFoundError as e:
            raise SourceNotFoundError(
                f"File di indici campionati non trovato: {file_path}"
            ) from e


# --- FACADE PER IL CARICAMENTO DA MLFLOW ---


class MLflowArtifactLoader:
    """
    Unico punto di ingresso per caricare artefatti usando un MLflow Run ID.
    Questa classe è una "facade": non contiene logica di caricamento da file,
    ma orchestra la risoluzione del path da MLflow e delega ai loader specifici.
    """

    @staticmethod
    def _resolve_cutout_list_path(
        run_id: str, project_root: pathlib.Path
    ) -> pathlib.Path:
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            root = params.get("root_folder")
            run_folder = params.get("run_folder")
            filename = params.get("merged_cutouts_info_filename") or params.get(
                "cutouts_info_filename"
            )

            if not all([root, run_folder, filename]):
                raise SourceNotFoundError(
                    f"Parametri MLflow mancanti (root_folder, run_folder, info_filename) nella run {run_id}"
                )

            relative_path = pathlib.Path(root) / run_folder / filename
            return project_root / relative_path
        except mlflow.exceptions.MlflowException as e:
            raise SourceNotFoundError(
                f"Impossibile trovare o accedere alla run MLflow con ID: {run_id}"
            ) from e

    @staticmethod
    def _resolve_features_archive_path(
        run_id: str, project_root: pathlib.Path
    ) -> pathlib.Path:
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            root = params.get("root_folder")
            run_folder = params.get("run_folder")
            filename = params.get("features_filename")

            if not all([root, run_folder, filename]):
                raise SourceNotFoundError(
                    f"Parametri MLflow mancanti (root_folder, run_folder, features_filename) nella run {run_id}"
                )

            relative_path = pathlib.Path(root) / run_folder / filename
            return project_root / relative_path
        except mlflow.exceptions.MlflowException as e:
            raise SourceNotFoundError(
                f"Impossibile trovare o accedere alla run MLflow con ID: {run_id}"
            ) from e

    @staticmethod
    def load_cutout_list_from_mlflow(
        run_id: str, project_root: pathlib.Path
    ) -> CutoutListData:
        path = MLflowArtifactLoader._resolve_cutout_list_path(run_id, project_root)
        return CutoutLoader.from_json(path)

    @staticmethod
    def load_features_from_mlflow(
        run_id: str, project_root: pathlib.Path, use_npy_compatible: bool = True
    ) -> FeaturesData:
        h5_path = MLflowArtifactLoader._resolve_features_archive_path(
            run_id, project_root
        )
        target_path = h5_path.with_suffix(".npy") if use_npy_compatible else h5_path
        return FeaturesLoader.load_features_from_path(target_path)

    @staticmethod
    def load_image_paths_from_mlflow(
        run_id: str, project_root: pathlib.Path
    ) -> ImagePathsData:
        h5_path = MLflowArtifactLoader._resolve_features_archive_path(
            run_id, project_root
        )
        return FeaturesLoader.load_image_paths_from_h5(h5_path)

    @staticmethod
    def _resolve_clustering_path(
        run_id: str, project_root: pathlib.Path
    ) -> pathlib.Path:
        """Risolve il path della directory di output del clustering da una run MLflow."""
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            # Il path cruciale è 'exp_dir', che è la root dell'output del clustering
            exp_dir = params.get("exp_dir")

            if not exp_dir:
                raise SourceNotFoundError(
                    f"Parametro 'exp_dir' non trovato nella run MLflow {run_id}"
                )

            relative_path = pathlib.Path(exp_dir)
            return project_root / relative_path
        except mlflow.exceptions.MlflowException as e:
            raise SourceNotFoundError(
                f"Impossibile trovare o accedere alla run MLflow con ID: {run_id}"
            ) from e

    @staticmethod
    def load_clustering_from_mlflow(
        run_id: str, project_root: pathlib.Path
    ) -> HierarchicalClusteringData:
        """Carica l'intera gerarchia di clustering da un MLflow Run ID."""
        dir_path = MLflowArtifactLoader._resolve_clustering_path(run_id, project_root)
        return ClusteringLoader.from_directory(dir_path)
