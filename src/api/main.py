from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

# FastAPI アプリケーションの作成
app = FastAPI()

# SentenceTransformerモデルの初期化
model = SentenceTransformer("intfloat/multilingual-e5-small")


@app.on_event("startup")
async def startup_event():
    print("モデルを初期化します。")
    # ここでモデルをプリロードするなどの初期化処理を行うことができます。


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/embed")
async def get_embedding(sentence: str):
    # 文章の埋め込みを取得
    embedding = model.encode(sentence)
    return {"embedding": embedding.tolist()}
