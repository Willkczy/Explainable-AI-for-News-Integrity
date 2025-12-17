"""
Utility functions for downloading models from Google Cloud Storage
"""
import os
from google.cloud import storage
from pathlib import Path


def download_model_from_gcs(bucket_name: str, gcs_path: str, local_path: str) -> str:
    """
    Download model files from Google Cloud Storage to local directory.

    Args:
        bucket_name (str): GCS bucket name
        gcs_path (str): Path to model directory in GCS (e.g., 'models/checkpoint_roberta')
        local_path (str): Local directory to save the model

    Returns:
        str: Path to the downloaded model directory
    """
    # Check if model already exists locally
    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if it has the required files
        required_files = ['config.json', 'pytorch_model.bin']
        if all(os.path.exists(os.path.join(local_path, f)) for f in required_files):
            print(f"Model already exists at {local_path}, skipping download.")
            return local_path

    print(f"Downloading model from gs://{bucket_name}/{gcs_path} to {local_path}...")

    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs in the model directory
    blobs = bucket.list_blobs(prefix=gcs_path)

    # Download each file
    downloaded_files = 0
    for blob in blobs:
        # Skip if it's a directory marker
        if blob.name.endswith('/'):
            continue

        # Get relative path within the model directory
        relative_path = blob.name[len(gcs_path):].lstrip('/')
        if not relative_path:
            continue

        # Full local file path
        local_file_path = os.path.join(local_path, relative_path)

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        print(f"  Downloading {blob.name}...")
        blob.download_to_filename(local_file_path)
        downloaded_files += 1

    print(f"✓ Downloaded {downloaded_files} files to {local_path}")
    return local_path


def get_model_path(use_gcs: bool = None) -> str:
    """
    Get the appropriate model path, downloading from GCS if needed.

    Args:
        use_gcs (bool): Whether to use GCS. If None, checks environment variable.

    Returns:
        str: Path to the model directory
    """
    from config.config import (
        DETECTOR_MODEL_PATH,
        GCS_MODEL_BUCKET,
        GCS_MODEL_PATH
    )

    # Determine if we should use GCS
    if use_gcs is None:
        use_gcs = os.getenv("USE_GCS_MODEL", "false").lower() == "true"

    if use_gcs:
        # Download from GCS
        local_model_path = "./models/checkpoint_roberta"
        return download_model_from_gcs(
            bucket_name=GCS_MODEL_BUCKET,
            gcs_path=GCS_MODEL_PATH,
            local_path=local_model_path
        )
    else:
        # Use local path
        return DETECTOR_MODEL_PATH
