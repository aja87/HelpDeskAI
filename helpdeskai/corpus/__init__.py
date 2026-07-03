"""Corpus download package for raw dataset acquisition and shaping."""

from .config import DownloadConfig
from .downloader import run_download

__all__ = ["DownloadConfig", "run_download"]
