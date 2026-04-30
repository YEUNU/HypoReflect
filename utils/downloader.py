import os
import requests
import zipfile
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

def download_benchmark_dataset(repo_url: str, target_dir: str):
    """
    Downloads a dataset from a GitHub repository (zip format) and extracts it.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dataset.zip")
    
    logger.info(f"Downloading dataset from {repo_url}...")
    try:
        response = requests.get(repo_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove(zip_path)
        logger.info(f"Dataset extracted to {target_dir}")
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

def clone_official_repos():
    """Clones the three target RAG repositories for reference."""
    repos = {
        "HopRAG": "https://github.com/LIU-Hao-2002/HopRAG",
        "CRAG": "https://github.com/HuskyInSalt/CRAG",
        "MS_GraphRAG": "https://github.com/microsoft/graphrag"
    }
    
    target_base = "third_party"
    os.makedirs(target_base, exist_ok=True)
    
    for name, url in repos.items():
        target_path = os.path.join(target_base, name)
        if os.path.exists(target_path):
            logger.info(f"{name} already exists. Skipping clone.")
            continue
            
        logger.info(f"Cloning {name} from {url}...")
        try:
            subprocess.run(["git", "clone", url, target_path], check=True)
            logger.info(f"Successfully cloned {name}")
        except Exception as e:
            logger.error(f"Failed to clone {name}: {e}")