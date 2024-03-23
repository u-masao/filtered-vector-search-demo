import logging
import time

import qdrant_client
from fastapi import FastAPI
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# FastAPI アプリケーションの作成
app = FastAPI()

# SentenceTransformerモデルの初期化
model = SentenceTransformer("intfloat/multilingual-e5-small")

# init qdrant client
client = qdrant_client.QdrantClient(
    "127.0.0.1",
    port=6333,
)

config = {"collection_name": "livedoor_news"}


def search(query_vector, with_vectors=False, with_payload=False):
    logger = logging.getLogger(__name__)
    # search
    ts = time.perf_counter()
    search_result = client.search(
        collection_name=config["collection_name"],
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="sports-watch"),
                ),
            ],
        ),
        limit=5,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        with_vectors=with_vectors,
        with_payload=with_payload,
    )
    search_elapsed_time = time.perf_counter() - ts
    logger.info(f"search_result: \n{search_result}")
    logger.info(f"search_elapsed_time: \n{search_elapsed_time}")
    return search_result


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger(__name__)
    logger.info("モデルを初期化します。")
    # ここでモデルをプリロードするなどの初期化処理を行うことができます。


@app.get("/embed")
async def get_embedding(sentence: str):
    # 文章の埋め込みを取得
    embedding = model.encode(sentence)
    # return {"embedding": embedding.tolist()}
    return search(embedding)
