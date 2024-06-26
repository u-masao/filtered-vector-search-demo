import logging
import re
import time

import numpy as np
import polars as pl
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

# load text
text_df = pl.read_parquet("../../data/raw/sentences-limit-0.parquet")


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


def remove_non_numerics(s):
    return re.sub(r"[^0-9,]", "", s)


# Qdrant への問い合わせ
def search(
    query_vector=None,
    ids: str = "",
    with_vectors=False,
    with_payload=False,
    filter_category=None,
    limit=5,
):

    if (query_vector is None) and ids == "":
        return {
            "items": None,
            "qdrant_response_time": None,
        }
    # search
    ts = time.perf_counter()

    ids = remove_non_numerics(ids)
    if ids != "":
        result_ids = client.retrieve(
            collection_name=config["collection_name"],
            ids=[int(x) for x in ids.split(",")],
            with_vectors=True,
            with_payload=False,
        )
        embed_ids = []
        if len(result_ids) > 0:
            [embed_ids.append(x.vector) for x in result_ids]
            embed_ids = np.average(np.array(embed_ids), axis=0)
        if query_vector is None:
            query_vector = embed_ids
        else:
            query_vector = query_vector + embed_ids * 0.3

    query_filter = None
    if filter_category is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=filter_category),
                ),
            ],
        )

    search_result = client.search(
        collection_name=config["collection_name"],
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        with_vectors=with_vectors,
        with_payload=with_payload,
    )
    qdrant_response_time = time.perf_counter() - ts
    return {
        "items": search_result,
        "qdrant_response_time": qdrant_response_time,
    }


@app.get("/item")
async def get_item(id: int):
    result = text_df.filter(pl.col("id") == id).to_dicts()
    return result


@app.get("/search")
async def get_embedding(
    sentence: str = "",
    ids: str = "",
    with_vectors: bool = False,
    with_payload: bool = False,
    filter_category: str = None,
    limit: int = 5,
):
    # フィルタ文字列がない場合の処理
    if filter_category == "":
        filter_category = None

    # 埋め込み表現を取得
    start_ts = time.perf_counter()
    if sentence != "":
        embedding = model.encode(sentence)
        embedding_time = time.perf_counter() - start_ts
    else:
        embedding = None
        embedding_time = None

    # 検索
    result = search(
        embedding,
        ids=ids,
        with_vectors=with_vectors,
        with_payload=with_payload,
        filter_category=filter_category,
        limit=limit,
    )
    total_time = time.perf_counter() - start_ts

    # 結果を返す
    result["embedding_time"] = embedding_time
    result["total_time"] = total_time
    return result
