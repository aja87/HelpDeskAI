import os
import argparse
import pandas as pd
from datasets import load_dataset

def download_and_save_dataset(dataset_name, size=None) -> str:
    os.makedirs('data/raw', exist_ok=True)
    
    print(f"Téléchargement du dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"Erreur  : {e}")
        exit(2)
            
    # Sélectionner le nombre de lignes
    if size is not None and size < len(dataset):
        print(f"Extraction de {size} lignes...")
        dataset = dataset.select(range(size))
    else:
        print(f"Téléchargement complet ({len(dataset)} lignes).")
        
    # Conversion en Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Nettoyage du nom pour le fichier de sortie
    safe_name = dataset_name.replace('/', '_')
    output_path = f"data/raw/{safe_name}.csv"

    # Sauvegarde en CSV    
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé avec succès dans: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge un dataset Hugging Face et le sauvegarde en CSV.")
    parser.add_argument("dataset_name", type=str, help="Nom du dataset")
    parser.add_argument("--size", type=int, default=None, help="Nombre de lignes à télécharger (par défaut: tout)")
    
    args = parser.parse_args()
    if not args.dataset_name or ' ' in args.dataset_name or len(args.dataset_name) < 5:
        print("Erreur : Le nom du dataset ne doit pas être vide, ne doit pas contenir d'espaces et doit faire au moins 5 caractères.")
        exit(1)
    download_and_save_dataset(args.dataset_name, args.size)