import json
import logging
import time
from datetime import datetime
from pathlib import Path

import click
import cloudpickle
import mlflow
import polars as pl
import qdrant_client
from qdrant_client.http import models


def log_params(params):
    logger = logging.getLogger(__name__)
    logger.info(params)
    mlflow.log_params(params)


def load_dataset(text_filepath, embeds_filepath):
    logger = logging.getLogger(__name__)

    text_df = pl.read_parquet(text_filepath)
    embeds = cloudpickle.load(open(embeds_filepath, "rb"))

    logger.info(f"text_df: \n{text_df.head()}")
    logger.info(f"embeds.shape: \n{embeds.shape}")
    logger.info(f"embeds[0]: \n{embeds[0]}")

    return text_df, embeds


def build_features(text_df):

    text_df = (
        text_df.with_columns(
            text_df["sentence"].str.len_bytes().alias("sentence_length")
        )
        .with_columns(text_df["title"].str.slice(0, 40).alias("title_summary"))
        .with_columns(
            text_df["sentence"].str.slice(0, 300).alias("sentence_summary")
        )
    )
    return text_df


def init_qdrant_collection(kwargs):
    """
    Qdrant の collection を初期化しデータを投入します。
    軽い動作チェックもします。
    """

    logger = logging.getLogger(__name__)

    # load dataset
    text_df, embeds = load_dataset(
        kwargs["input_text_filepath"], kwargs["input_embeds_filepath"]
    )

    if kwargs["limit"] > 0:
        text_df = text_df.head(kwargs["limit"])
        embeds = embeds[: kwargs["limit"]]

    # build features
    text_df = build_features(text_df)

    # init qdrant client
    client = qdrant_client.QdrantClient(
        kwargs["qdrant_host"], port=kwargs["qdrant_port"]
    )

    # delete collection
    client.delete_collection(collection_name=kwargs["collection_name"])

    # create collection
    client.create_collection(
        collection_name=kwargs["collection_name"],
        vectors_config=models.VectorParams(
            size=embeds.shape[1], distance=models.Distance.COSINE
        ),
    )

    # upload data
    ts = time.perf_counter()
    client.upload_collection(
        collection_name=kwargs["collection_name"],
        ids=text_df["id"],
        payload=text_df[
            [
                "id",
                "date",
                "category",
                "title_summary",
                "sentence_summary",
                "sentence_length",
                "url",
            ]
        ].to_dicts(),
        vectors=embeds,
        parallel=4,
        max_retries=3,
    )
    upload_elapsed_time = time.perf_counter() - ts

    # create payload index
    for name, schema in [
        ["id", "integer"],
        ["date", "datetime"],
        ["title", "text"],
        ["sentence_length", "integer"],
        ["category", "keyword"],
    ]:
        client.create_payload_index(
            collection_name=kwargs["collection_name"],
            field_name=name,
            field_schema=schema,
        )

    # search
    ts = time.perf_counter()
    search_result = client.search(
        collection_name=kwargs["collection_name"],
        query_vector=embeds[0],
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
        with_vectors=True,
        with_payload=True,
    )
    search_elapsed_time = time.perf_counter() - ts
    logger.info(f"search_result: \n{search_result}")

    logger.info(pl.DataFrame(search_result))

    # make result
    result = {
        "timestamp": datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"),
        "upload_elapsed_time": upload_elapsed_time,
        "search_elapsed_time": search_elapsed_time,
    }

    return result


@click.command()
@click.argument("input_text_filepath", type=click.Path(exists=True))
@click.argument("input_embeds_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--collection_name", type=str, required=True)
@click.option("--qdrant_host", type=str, default="127.0.0.1")
@click.option("--qdrant_port", type=int, default=6333)
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--limit", type=int, default=0)
def main(**kwargs):
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("init_qdrant_collection")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])
    log_params({f"args.{k}": v for k, v in kwargs.items()})

    # process
    result = init_qdrant_collection(kwargs)

    # output result
    save_path = Path(kwargs["output_filepath"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(save_path, "w"), ensure_ascii=False, indent=4)
    logger.info(f"result: \n{result}")
    log_params(result)

    # clenup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
