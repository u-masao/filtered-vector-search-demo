```mermaid
flowchart TD
	node1["data/raw/embeds_limit-0_chunk_split.cloudpickle.dvc"]
	node2["data/raw/sentences-limit-0.parquet.dvc"]
	node3["qdrant_init_collection"]
	node1-->node3
	node2-->node3
```
