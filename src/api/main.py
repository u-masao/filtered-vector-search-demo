import logging
import time

import qdrant_client
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# configuration
config = {"collection_name": "livedoor_news"}

# FastAPI アプリケーションの作成
app = FastAPI()

# add static contents
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/static/index.html")


# SentenceTransformerモデルの初期化
model = SentenceTransformer("intfloat/multilingual-e5-small")

# init qdrant client
client = qdrant_client.QdrantClient(
    "127.0.0.1",
    port=6333,
)


# モデルの初期化処理
@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger(__name__)
    logger.info("モデルを初期化します。")
    # ここでモデルをプリロードするなどの初期化処理を行うことができます。


# Qdrant への問い合わせ
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


@app.get("/embed")
async def get_embedding(sentence: str):
    embedding = model.encode(sentence)
    return search(embedding)
