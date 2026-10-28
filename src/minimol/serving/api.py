"""Re-export the FastAPI app under the name used in the drug-discovery plan."""

from minimol.serving.app import create_app

__all__ = ["create_app"]
