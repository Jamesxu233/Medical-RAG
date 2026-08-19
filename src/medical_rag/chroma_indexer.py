from typing import Any, Dict, Optional
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import chromadb
from tqdm.auto import tqdm
import gc
import json
from sentence_transformers import SentenceTransformer


class PubMedChromaIndexer:
    """
    使用 BGE 嵌入模型和 ChromaDB 构建或加载 PMC/PubMed 文献 Chunk 索引。

    典型用法
    --------
    1. 加载已有/恢复后的数据库：
        indexer = PubMedChromaIndexer(
            model_name="BAAI/bge-small-en-v1.5",
            persist_directory=r"D:\chroma_recovery\chroma_pubmed_db_rebuilt",
            collection_name="pubmed_fulltext_bge_small",
            device="cuda",
            load_existing=True,
        )

    2. 首次创建新索引：
        indexer = PubMedChromaIndexer(
            persist_directory=r"F:\RAG\chroma_pubmed_db",
            collection_name="pubmed_fulltext_bge_small",
            load_existing=False,
        )
        indexer.build_index(chunks_df)
    """

    QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    DOCUMENT_KEY = "chroma:document"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        persist_directory: str = "chroma_pubmed_db",
        collection_name: str = "pubmed_fulltext_chunks",
        device: Optional[str] = None,
        embedding_batch_size: int = 64,
        insert_batch_size: int = 2000,
        load_existing: bool = True,
        query_max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.persist_directory = Path(persist_directory).expanduser()
        self.collection_name = collection_name
        self.embedding_batch_size = int(embedding_batch_size)
        self.insert_batch_size = int(insert_batch_size)
        self.load_existing = bool(load_existing)
        self.query_max_tokens = int(query_max_tokens)

        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be greater than zero.")

        if self.insert_batch_size <= 0:
            raise ValueError("insert_batch_size must be greater than zero.")

        if self.query_max_tokens <= 0:
            raise ValueError("query_max_tokens must be greater than zero.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # 加载已有数据库时，不自动创建目录，避免路径写错后生成空数据库。
        if self.load_existing:
            if not self.persist_directory.exists():
                raise FileNotFoundError(
                    "Persist directory does not exist:\n"
                    f"{self.persist_directory.resolve()}"
                )

            if not self.persist_directory.is_dir():
                raise NotADirectoryError(
                    "Persist path is not a directory:\n"
                    f"{self.persist_directory.resolve()}"
                )
        else:
            self.persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        print(
            "Persist directory:",
            self.persist_directory.resolve(),
        )
        print(f"Collection name: {self.collection_name}")

        self.embedding_model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        dimension = self.embedding_model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        self.embedding_dimension = int(dimension)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory.resolve())
        )

        if self.load_existing:
            self.collection = self._load_existing_collection()
        else:
            self.collection = self._create_collection()

        self._validate_loaded_collection()

    @staticmethod
    def _collection_name(item: Any) -> str:
        """
        兼容不同 Chroma 版本中 list_collections() 的返回格式。
        """
        if isinstance(item, str):
            return item

        name = getattr(item, "name", None)

        if name is None:
            return str(item)

        return str(name)

    def _list_collection_names(self) -> list[str]:
        return [
            self._collection_name(item)
            for item in self.client.list_collections()
        ]

    def _load_existing_collection(self):
        """
        加载已存在的 collection。

        不使用 get_or_create_collection，避免目录或名称写错时静默创建空 collection。
        """
        existing_names = self._list_collection_names()

        print("Existing collections:", existing_names)

        if self.collection_name not in existing_names:
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist.\n"
                f"Available collections: {existing_names}\n"
                "Persist directory: "
                f"{self.persist_directory.resolve()}"
            )

        return self.client.get_collection(
            name=self.collection_name
        )

    def _create_collection(self):
        """
        仅用于首次构建全新索引。
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.model_name,
                "embedding_dimension": self.embedding_dimension,
            },
        )

    def _validate_loaded_collection(self) -> None:
        """
        验证 collection 可被重新加载，并检查基本配置。
        """
        try:
            count = int(self.collection.count())
        except Exception as exc:
            raise RuntimeError(
                "Collection was found, but Chroma could not read it. "
                "The vector segment may be damaged or incompatible."
            ) from exc

        metadata = self.collection.metadata or {}

        stored_model = metadata.get("embedding_model")
        stored_dimension = metadata.get("embedding_dimension")
        stored_metric = metadata.get("hnsw:space")

        if stored_model and stored_model != self.model_name:
            raise ValueError(
                "Embedding model mismatch:\n"
                f"Stored model: {stored_model}\n"
                f"Current model: {self.model_name}"
            )

        if stored_dimension is not None:
            try:
                stored_dimension = int(stored_dimension)
            except (TypeError, ValueError):
                stored_dimension = None

        if (
            stored_dimension is not None
            and stored_dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding dimension mismatch:\n"
                f"Stored dimension: {stored_dimension}\n"
                f"Current dimension: {self.embedding_dimension}"
            )

        print(
            "Collection loaded successfully. "
            f"Count: {count:,}"
        )
        print(
            "Embedding dimension:",
            self.embedding_dimension,
        )
        print(
            "Distance metric:",
            stored_metric or "not recorded",
        )

    @staticmethod
    def _clean_scalar(value: Any) -> str:
        """
        清洗 metadata 字符串，避免 NaN、None 等不能正常写入。
        """
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value).strip()

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default

            return int(value)
        except (TypeError, ValueError):
            return default

    def _prepare_chunks_dataframe(
        self,
        chunks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "text",
            "doc_id",
            "chunk_index",
            "source_title",
        }

        missing_columns = required_columns - set(chunks_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        df = chunks_df.copy()

        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df = df[df["text"].ne("")].copy()

        df["doc_id"] = (
            df["doc_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if df["doc_id"].eq("").any():
            raise ValueError(
                "doc_id contains empty values after cleaning."
            )

        df["chunk_index"] = pd.to_numeric(
            df["chunk_index"],
            errors="coerce",
        ).fillna(0).astype(int)

        df["vector_id"] = (
            df["doc_id"]
            + "_chunk_"
            + df["chunk_index"].map(lambda x: f"{x:04d}")
        )

        duplicate_count = int(
            df["vector_id"].duplicated().sum()
        )

        if duplicate_count > 0:
            duplicated = df.loc[
                df["vector_id"].duplicated(keep=False),
                ["vector_id", "doc_id", "chunk_index"],
            ].head(20)

            raise ValueError(
                f"Found {duplicate_count} duplicated vector IDs.\n"
                f"Examples:\n{duplicated}"
            )

        return df.reset_index(drop=True)

    def _build_metadata(
        self,
        row: pd.Series,
    ) -> Dict[str, Any]:
        """
        构建 Chroma metadata。

        Chroma metadata 仅保存标量值；document 由 documents 参数单独写入。
        """
        pub_date = self._clean_scalar(
            row.get("pub_date", "")
        )

        publication_year = 0

        if pub_date:
            match = re.search(
                r"\b(?:18|19|20)\d{2}\b",
                pub_date,
            )

            if match:
                publication_year = int(match.group(0))

        return {
            "doc_id": self._clean_scalar(
                row.get("doc_id", "")
            ),
            "chunk_index": self._safe_int(
                row.get("chunk_index", 0)
            ),
            "total_chunks": self._safe_int(
                row.get("total_chunks", 1),
                default=1,
            ),
            "source_title": self._clean_scalar(
                row.get("source_title", "")
            ),
            "journal": self._clean_scalar(
                row.get("journal", "")
            ),
            "pub_date": pub_date,
            "publication_year": publication_year,
            "pmid": self._clean_scalar(
                row.get("pmid", "")
            ),
            "token_count": self._safe_int(
                row.get("token_count", 0)
            ),
            "split_strategy": self._clean_scalar(
                row.get("split_strategy", "")
            ),
        }

    def truncate_text_by_tokens(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        按 tokenizer token 数截断文本。

        对查询文本的 instruction 和正文一起截断，避免原实现中
        instruction 拼接后仍使用未截断文本编码的问题。
        """
        if max_tokens is None:
            max_tokens = self.query_max_tokens

        tokenizer = self.embedding_model.tokenizer

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens,
        )

        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        生成文档 embedding。

        文档不添加 BGE query instruction。
        """
        if not texts:
            raise ValueError("texts cannot be empty.")

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != self.embedding_dimension
        ):
            raise RuntimeError(
                "Unexpected document embedding shape: "
                f"{embeddings.shape}"
            )

        return embeddings

    def encode_query(
        self,
        query_text: str,
    ) -> np.ndarray:
        """
        对查询添加 BGE 检索指令，并按模型最大输入长度截断。
        """
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        query_text = query_text.strip()

        if not query_text:
            raise ValueError("query_text cannot be empty.")

        query_with_instruction = (
            self.QUERY_INSTRUCTION + query_text
        )

        query_with_instruction = self.truncate_text_by_tokens(
            query_with_instruction,
            max_tokens=self.query_max_tokens,
        )

        embedding = self.embedding_model.encode(
            [query_with_instruction],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            self.embedding_dimension,
        )

        if embedding.shape != expected_shape:
            raise RuntimeError(
                "Unexpected query embedding shape: "
                f"{embedding.shape}; expected {expected_shape}"
            )

        return embedding

    def recreate_collection(self) -> None:
        """
        删除当前 collection 并重新创建。

        仅用于明确需要从头重建索引时。恢复后的数据库不要调用此方法。
        """
        existing_names = self._list_collection_names()

        if self.collection_name in existing_names:
            self.client.delete_collection(
                name=self.collection_name
            )

        self.collection = self._create_collection()

    def build_index(
        self,
        chunks_df: pd.DataFrame,
        stats_path: str = "vector_index_stats.json",
        recreate_collection: bool = False,
    ) -> Dict[str, Any]:
        """
        构建或增量更新索引。

        注意：
        - 加载恢复后的数据库做查询时，不要调用本方法。
        - recreate_collection=True 会删除并重建当前 collection。
        """
        df = self._prepare_chunks_dataframe(chunks_df)

        if recreate_collection:
            self.recreate_collection()

        total_rows = len(df)

        if total_rows == 0:
            raise ValueError(
                "No valid chunks are available for indexing."
            )

        print(f"Valid chunks to index: {total_rows:,}")
        print(
            f"Embedding dimension: {self.embedding_dimension}"
        )

        for start in tqdm(
            range(0, total_rows, self.insert_batch_size),
            desc="Embedding and indexing",
        ):
            end = min(
                start + self.insert_batch_size,
                total_rows,
            )

            batch_df = df.iloc[start:end]

            documents = batch_df["text"].tolist()
            ids = batch_df["vector_id"].tolist()

            metadatas = [
                self._build_metadata(row)
                for _, row in batch_df.iterrows()
            ]

            embeddings = self.encode_documents(
                documents
            )

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

            del embeddings
            del documents
            del metadatas

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            gc.collect()

        indexed_count = int(
            self.collection.count()
        )

        metadata_fields = [
            "doc_id",
            "chunk_index",
            "total_chunks",
            "source_title",
            "journal",
            "pub_date",
            "publication_year",
            "pmid",
            "token_count",
            "split_strategy",
        ]

        stats = {
            "collection_name": self.collection_name,
            "total_chunks": indexed_count,
            "expected_chunks": int(total_rows),
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": "cosine",
            "index_built_at": pd.Timestamp.now().isoformat(),
            "persist_directory": str(
                self.persist_directory.resolve()
            ),
            "chunk_size_stats": (
                {
                    "mean": float(
                        df["token_count"].mean()
                    ),
                    "max": int(
                        df["token_count"].max()
                    ),
                    "min": int(
                        df["token_count"].min()
                    ),
                    "median": float(
                        df["token_count"].median()
                    ),
                }
                if "token_count" in df.columns
                else {}
            ),
            "metadata_fields": metadata_fields,
            "collection_count_after_build": indexed_count,
        }

        stats_path_obj = Path(stats_path)
        stats_path_obj.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with stats_path_obj.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                stats,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            json.dumps(
                stats,
                indent=2,
                ensure_ascii=False,
            )
        )

        return stats

    def validate_index(
        self,
        sample_size: int = 3,
    ) -> Dict[str, Any]:
        """
        验证恢复后或新建的数据库是否可以正常 count、get 和读取 embedding。
        """
        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        count = int(
            self.collection.count()
        )

        report: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "count": count,
            "embedding_dimension_expected": (
                self.embedding_dimension
            ),
            "sample_ids": [],
            "sample_embedding_dimension": None,
            "valid": False,
        }

        if count == 0:
            return report

        limit = min(
            sample_size,
            count,
        )

        sample = self.collection.get(
            limit=limit,
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

        sample_ids = sample.get("ids") or []
        sample_embeddings = sample.get("embeddings")

        report["sample_ids"] = sample_ids

        if (
            sample_embeddings is not None
            and len(sample_embeddings) > 0
        ):
            report["sample_embedding_dimension"] = len(
                sample_embeddings[0]
            )

        report["valid"] = (
            len(sample_ids) == limit
            and report["sample_embedding_dimension"]
            == self.embedding_dimension
        )

        return report

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        执行向量检索。

        Parameters
        ----------
        query_text:
            查询文本。
        n_results:
            返回结果数量。
        where_filter:
            Chroma metadata filter，例如：
            {"journal": "Nature Communications"}

            或：
            {
                "$and": [
                    {"journal": "Nature Communications"},
                    {"publication_year": {"$gte": 2021}}
                ]
            }

        Returns
        -------
        pd.DataFrame
            包含 rank、vector_id、distance、similarity、text 和 metadata。
        """
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string.")

        query_text = query_text.strip()

        if not query_text:
            raise ValueError("query_text cannot be empty.")

        if n_results <= 0:
            raise ValueError(
                "n_results must be greater than zero."
            )

        collection_count = int(
            self.collection.count()
        )

        if collection_count == 0:
            raise RuntimeError(
                f"Collection '{self.collection_name}' is empty."
            )

        query_embedding = self.encode_query(
            query_text
        )

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": min(
                int(n_results),
                collection_count,
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter is not None:
            query_kwargs["where"] = where_filter

        results = self.collection.query(
            **query_kwargs
        )

        rows = []

        result_ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for rank, (
            vector_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                result_ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            metadata = metadata or {}
            distance_value = float(distance)

            rows.append(
                {
                    "rank": rank,
                    "vector_id": vector_id,
                    "distance": distance_value,
                    # cosine distance 越小越相似
                    "similarity": 1.0 - distance_value,
                    "text": document,
                    **metadata,
                }
            )

        return pd.DataFrame(rows)

    def get_stats(self) -> Dict[str, Any]:
        """
        返回当前 collection 的基本统计信息。
        """
        return {
            "collection_name": self.collection_name,
            "total_chunks": int(
                self.collection.count()
            ),
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "persist_directory": str(
                self.persist_directory.resolve()
            ),
            "collection_metadata": (
                self.collection.metadata or {}
            ),
        }