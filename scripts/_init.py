from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Union
from download_corpus_V2 import download_and_save_dataset

def read_json_to_python_object(file_path: Union[str, Path]) -> Any:
	"""Lit un fichier JSON et retourne l'objet Python correspondant."""
	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"JSON file not found: {path}")
	try:
		with path.open("r", encoding="utf-8") as f:
			return json.load(f)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid JSON in file: {path}") from exc

def read_csv_dataset(
    file_path: Union[str, Path],
    start_row: int = 0,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Lit un fichier CSV et retourne une liste de dictionnaires représentant les lignes.
    Args:
        file_path (Union[str, Path]): Chemin vers le fichier CSV.
        start_row (int): Ligne de départ pour la lecture (0-indexé).
        max_rows (int | None): Nombre maximum de lignes à lire. Si vide, lit tout.
    Returns:
        list[dict[str, Any]]: Liste de dictionnaires représentant les lignes du CSV.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if start_row < 0:
        raise ValueError("start_row must be >= 0")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be >= 0")

    # Certains jeux de donnees contiennent des cellules CSV tres longues.
    # On releve la limite du module csv pour eviter _csv.Error: field larger than field limit.
    max_csv_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_csv_field_size)
            break
        except OverflowError:
            max_csv_field_size //= 10

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for _ in range(start_row):
            next(reader, None)

        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows

def prepare_dataset_and_config(config_path):    
    """Prépare le dataset et la configuration à partir du fichier JSON."""
    try:
        config = read_json_to_python_object(config_path)
        dataset_path = download_and_save_dataset(config["datasets"]["name"])
        config["datasets"]["download_path"] = dataset_path
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
    return config