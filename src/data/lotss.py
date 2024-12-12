from dataclasses import dataclass, field
from typing import List, Iterator
import os

@dataclass
class LoTTSMosaic:
    mosaic_name: str  # Name of the mosaic
    mosaic_path: str  # Full path to the mosaic-blanked.fits file
    mask_path: str  # Full path to the mosaic.pybdsmmask.fits file

@dataclass
class LoTTSCollection:
    name: str  # Name of the dataset (e.g., "LoTTS")
    path: str  # Path to the main directory of the collection
    mosaics_path: str = ""
    mosaics: List[LoTTSMosaic] = field(default_factory=list)  # List of mosaics
    mosaics_subdir: str = "mosaics"

    def __post_init__(self):
        """
        Initializes the list of mosaics by scanning the main directory.
        Each subdirectory represents a mosaic.
        """
        if not self.mosaics:
            self.mosaics_path = os.path.join(self.path, self.mosaics_subdir)
            if os.path.isdir(self.mosaics_path):
                subdirs = [d for d in os.listdir(self.mosaics_path) if os.path.isdir(os.path.join(self.mosaics_path, d))]
                for subdir in subdirs:
                    mosaic_dir = os.path.join(self.mosaics_path, subdir)
                    mosaic_file = os.path.join(mosaic_dir, "mosaic-blanked.fits")
                    mask_file = os.path.join(mosaic_dir, "mosaic.pybdsmmask.fits")
                    if os.path.isfile(mosaic_file) and os.path.isfile(mask_file):
                        self.mosaics.append(LoTTSMosaic(
                            mosaic_name=subdir,
                            mosaic_path=mosaic_file,
                            mask_path=mask_file
                        ))
            else:
                raise ValueError(f"The directory {self.path} does not exist.")

    def __iter__(self) -> Iterator[LoTTSMosaic]:
        """
        Makes the class iterable, returning LoTTSMosaic objects.
        """
        return iter(self.mosaics)

    def get_mosaic(self, mosaic_name: str) -> LoTTSMosaic:
        """
        Returns a LoTTSMosaic object specified by the mosaic name.
        """
        for mosaic in self.mosaics:
            if mosaic.mosaic_name == mosaic_name:
                return mosaic
        raise ValueError(f"The mosaic '{mosaic_name}' does not exist in the collection.")

    def __repr__(self):
        return f"LoTTSCollection(name='{self.name}', path='{self.path}', mosaic_count={len(self.mosaics)})"

# Example usage
if __name__ == "__main__":
    # Initialize a LoTTS collection
    collection = LoTTSCollection(name="LoTTS", path="LoTTS")

    # Print information about the collection
    print(collection)

    # Iterate over the mosaics in the collection
    for mosaic in collection:
        print(f"Mosaic: {mosaic.mosaic_name}, Path: {mosaic.mosaic_path}, Mask: {mosaic.mask_path}")

    # Get information about a specific mosaic
    mosaic = collection.get_mosaic("mosaic_01")
    print(f"Mosaic: {mosaic.mosaic_name}, Path: {mosaic.mosaic_path}, Mask: {mosaic.mask_path}")