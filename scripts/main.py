
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from _setup import get_or_create_all_collection
def main():
    get_or_create_all_collection()

if __name__ == "__main__":
    main()
