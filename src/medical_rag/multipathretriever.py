from typing import Dict, Any
import pandas as pd
import numpy as np

from medical_rag.query_services import enhance_medical_query
from medical_rag.query_processor import MedicalQueryProcessor
from medical_rag.chroma_indexer import PubMedChromaIndexer
from medical_rag.retriever import MedicalChromaRetriever, MedicalBM25Retriever


#多路检索索引器

class MultiPathRetriever:
    def __init__(
        self,
        query_processor: MedicalQueryProcessor,
        vector_indexer: PubMedChromaIndexer,
        medical_retriever:
            MedicalChromaRetriever,
        bm25_retriever:
            MedicalBM25Retriever,
        vector_weight=0.65,
        bm25_weight=0.35,
        rrf_k=60,
    ):
        self.query_processor = query_processor
        self.vector_indexer = vector_indexer
        self.medical_retriever = (
            medical_retriever
        )
        self.bm25_retriever = (
            bm25_retriever
        )

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    @staticmethod
    def _normalize_score(series):
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        minimum = values.min()
        maximum = values.max()

        if maximum <= minimum:
            return pd.Series(
                np.ones(len(values)),
                index=values.index,
            )

        return (
            (values - minimum)
            / (maximum - minimum)
        )

    def prepare_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        直接复用Notebook中的查询增强流程。
        """
        return enhance_medical_query(
            query=query,
            query_processor=self.query_processor,
            retriever=self.medical_retriever,
            indexer=self.vector_indexer,
        )

    def vector_search(
        self,
        query_info,
        top_k=50,
    ) -> pd.DataFrame:
        """
        不传bge_query_for_display。

        vector_indexer.query()内部会自动添加：
        Represent this sentence for searching
        relevant passages:
        """
        results = self.vector_indexer.query(
            query_text=query_info[
                "vector_query"
            ],
            n_results=top_k,
            where_filter=query_info[
                "where_filter"
            ],
        )

        if results.empty:
            return results

        results = results.copy()

        results["vector_rank"] = (
            np.arange(1, len(results) + 1)
        )

        results["vector_score"] = (
            self._normalize_score(
                results["similarity"]
            )
        )

        results["retrieval_path"] = (
            "vector"
        )

        return results

    def keyword_search(
        self,
        query_info,
        top_k=50,
    ) -> pd.DataFrame:
        """
        使用keyword_terms，而不是带OR的keyword_query。
        """
        return self.bm25_retriever.search(
            keyword_terms=query_info[
                "keyword_terms"
            ],
            top_k=top_k,
            where_filter=query_info[
                "where_filter"
            ],
        )

    @staticmethod
    def _merge_results(
        vector_results: pd.DataFrame,
        bm25_results: pd.DataFrame,
    ) -> pd.DataFrame:
        candidate_pool = {}

        if not vector_results.empty:
            for _, row in vector_results.iterrows():
                record = row.to_dict()
                vector_id = str(
                    record["vector_id"]
                )

                candidate_pool[vector_id] = record

        if not bm25_results.empty:
            for _, row in bm25_results.iterrows():
                record = row.to_dict()
                vector_id = str(
                    record["vector_id"]
                )

                if vector_id not in candidate_pool:
                    candidate_pool[vector_id] = record
                else:
                    candidate_pool[
                        vector_id
                    ].update({
                        "bm25_rank":
                            record.get("bm25_rank"),
                        "bm25_score":
                            record.get("bm25_score"),
                        "bm25_initial_rank":
                            record.get(
                                "bm25_initial_rank"
                            ),
                    })

        if not candidate_pool:
            return pd.DataFrame()

        candidates = pd.DataFrame(
            candidate_pool.values()
        )

        # 保证融合需要的字段始终存在
        default_columns = {
            "vector_rank": np.nan,
            "vector_score": 0.0,
            "similarity": np.nan,
            "bm25_rank": np.nan,
            "bm25_score": 0.0,
            "bm25_initial_rank": np.nan,
        }

        for column, default in (
            default_columns.items()
        ):
            if column not in candidates.columns:
                candidates[column] = default

        return candidates
    
    @staticmethod
    def _numeric_column(
        dataframe: pd.DataFrame,
        column: str,
        default=np.nan,
    ) -> pd.Series:
            """
            安全读取DataFrame数值列。

            当列不存在时，返回与DataFrame索引长度一致的Series，
            避免pd.to_numeric(None)生成单个numpy.float64。
            """
            if column not in dataframe.columns:
                return pd.Series(
                    default,
                    index=dataframe.index,
                    dtype="float64",
                )

            return pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    def _apply_rrf(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reciprocal Rank Fusion。

        支持：
        - 向量和BM25均返回结果
        - 只有向量结果
        - 只有BM25结果
        """
        candidates = candidates.copy()

        vector_rank = self._numeric_column(
            candidates,
            "vector_rank",
        )

        bm25_rank = self._numeric_column(
            candidates,
            "bm25_rank",
        )

        candidates["vector_rrf"] = np.where(
            vector_rank.notna(),
            1.0 / (
                self.rrf_k
                + vector_rank.fillna(0.0)
            ),
            0.0,
        )

        candidates["bm25_rrf"] = np.where(
            bm25_rank.notna(),
            1.0 / (
                self.rrf_k
                + bm25_rank.fillna(0.0)
            ),
            0.0,
        )

        candidates["fusion_score"] = (
            candidates["vector_rrf"]
            + candidates["bm25_rrf"]
        )

        candidates["retrieval_path_count"] = (
            vector_rank.notna().astype(int)
            + bm25_rank.notna().astype(int)
        )

        return candidates

    def _apply_weighted(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        candidates = candidates.copy()

        vector_score = self._numeric_column(
            candidates,
            "vector_score",
            default=0.0,
        ).fillna(0.0)

        bm25_raw_score = self._numeric_column(
            candidates,
            "bm25_score",
            default=0.0,
        ).fillna(0.0)

        if bm25_raw_score.max() > bm25_raw_score.min():
            normalized_bm25_score = (
                bm25_raw_score
                - bm25_raw_score.min()
            ) / (
                bm25_raw_score.max()
                - bm25_raw_score.min()
            )
        elif (bm25_raw_score > 0).any():
            normalized_bm25_score = pd.Series(
                1.0,
                index=candidates.index,
            )
        else:
            normalized_bm25_score = pd.Series(
                0.0,
                index=candidates.index,
            )

        candidates[
            "normalized_bm25_score"
        ] = normalized_bm25_score

        candidates["fusion_score"] = (
            self.vector_weight * vector_score
            + self.bm25_weight
            * normalized_bm25_score
        )

        return candidates

    @staticmethod
    def _apply_simple(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        candidates = candidates.copy()

        vector_rank = self._numeric_column(
            candidates,
            "vector_rank",
        )

        bm25_rank = self._numeric_column(
            candidates,
            "bm25_rank",
        )

        candidates["retrieval_path_count"] = (
            vector_rank.notna().astype(int)
            + bm25_rank.notna().astype(int)
        )

        best_rank = pd.concat(
            [
                vector_rank.rename(
                    "vector_rank"
                ),
                bm25_rank.rename(
                    "bm25_rank"
                ),
            ],
            axis=1,
        ).min(
            axis=1,
            skipna=True,
        )

        candidates["fusion_score"] = (
            candidates["retrieval_path_count"]
            * 1000
            - best_rank.fillna(999)
        )

        return candidates

    @staticmethod
    def diversify_documents(
        candidates,
        top_k,
        max_chunks_per_doc=2,
    ):
        if candidates.empty:
            return candidates

        selected = []
        document_counts = {}

        for _, row in candidates.iterrows():
            doc_id = str(
                row.get("doc_id", "")
            )

            if (
                document_counts.get(doc_id, 0)
                >= max_chunks_per_doc
            ):
                continue

            selected.append(row)

            document_counts[doc_id] = (
                document_counts.get(doc_id, 0)
                + 1
            )

            if len(selected) >= top_k:
                break

        if not selected:
            return pd.DataFrame(
                columns=candidates.columns
            )

        return pd.DataFrame(
            selected
        ).reset_index(drop=True)

    def retrieve(
        self,
        query: str,
        top_k_vector=50,
        top_k_keyword=50,
        top_k_fused=30,
        fusion_strategy="rrf",
        max_chunks_per_doc=2,
    ) -> Dict[str, Any]:
        query = str(query).strip()

        if not query:
            raise ValueError("查询不能为空。")

        # 1. 查询理解与增强
        query_info = self.prepare_query(query)

        # 2. 向量召回
        vector_results = self.vector_search(
            query_info=query_info,
            top_k=top_k_vector,
        )

        # 3. BM25召回
        bm25_results = self.keyword_search(
            query_info=query_info,
            top_k=top_k_keyword,
        )

        # 4. 合并去重
        candidates = self._merge_results(
            vector_results,
            bm25_results,
        )

        if candidates.empty:
            return {
                "query_info": query_info,
                "vector_results":
                    vector_results,
                "bm25_results":
                    bm25_results,
                "fused_results":
                    candidates,
                "statistics": {
                    "vector_count": 0,
                    "bm25_count": 0,
                    "fused_count": 0,
                },
            }

        # 5. 融合
        if fusion_strategy == "rrf":
            candidates = self._apply_rrf(
                candidates
            )

        elif fusion_strategy == "weighted":
            candidates = self._apply_weighted(
                candidates
            )

        elif fusion_strategy == "simple":
            candidates = self._apply_simple(
                candidates
            )

        else:
            raise ValueError(
                "fusion_strategy必须是"
                "'rrf'、'weighted'或'simple'"
            )

        candidates = candidates.sort_values(
            "fusion_score",
            ascending=False,
        ).reset_index(drop=True)

        candidates["fusion_rank"] = (
            np.arange(1, len(candidates) + 1)
        )

        # 6. 文档级多样化
        fused_results = (
            self.diversify_documents(
                candidates,
                top_k=top_k_fused,
                max_chunks_per_doc=(
                    max_chunks_per_doc
                ),
            )
        )

        fused_results["fusion_rank"] = (
            np.arange(
                1,
                len(fused_results) + 1,
            )
        )

        return {
            "query_info": query_info,
            "vector_results":
                vector_results,
            "bm25_results":
                bm25_results,
            "fused_results":
                fused_results,
            "statistics": {
                "vector_count":
                    int(len(vector_results)),
                "bm25_count":
                    int(len(bm25_results)),
                "merged_candidate_count":
                    int(len(candidates)),
                "fused_count":
                    int(len(fused_results)),
                "fusion_strategy":
                    fusion_strategy,
            },
        }