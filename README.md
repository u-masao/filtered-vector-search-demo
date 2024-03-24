# Filtered Vector Serach Demo

このリポジトリは、フィルター処理付きのベクトル検索のデモです。
ベクトルデータとテキストデータは別途用意してください。

## 環境

- Docker
- Python: 3.10 以上
- poetry


## 起動方法

```
poetry install
make qdrant_start
make repro
make api
```

## 利用方法

http://localhost:8000/
