import re
from typing import Any, Dict
import pandas as pd
import numpy as np



from medical_rag.multipathretriever import MultiPathRetriever
from medical_rag.reranker import MedicalBGECrossEncoderReranker


class MedicalHybridRetrievalPipeline:
    """
    完整医学检索流水线：

    1. 医学查询理解与增强
    2. Chroma向量召回
    3. BM25关键词召回
    4. RRF/Weighted/Simple融合
    5. BGE Cross-Encoder重排
    6. 时效性和期刊权威性综合评分
    """

    def __init__(
        self,
        multi_path_retriever:
            MultiPathRetriever,
        reranker:
            MedicalBGECrossEncoderReranker,
    ):
        self.multi_path_retriever = (
            multi_path_retriever
        )

        self.reranker = reranker

    @staticmethod
    def _prepare_rerank_query(
        query_info: Dict[str, Any],
    ) -> str:
        """
        重排器不需要BGE向量模型的指令前缀。

        优先使用vector_query，因为它已经完成：
        - 医学实体增强
        - 同义词扩展
        - 中文查询英文转换或英文术语回退
        """
        rerank_query = str(
            query_info.get(
                "vector_query",
                "",
            )
            or query_info.get(
                "cleaned_query",
                "",
            )
            or query_info.get(
                "original_query",
                "",
            )
        ).strip()

        # 防止外部调用时意外传入带指令的查询
        rerank_query = re.sub(
            r"^Represent this "
            r"(?:sentence|question) "
            r"for searching relevant "
            r"passages:\s*",
            "",
            rerank_query,
            flags=re.IGNORECASE,
        ).strip()

        return rerank_query

    @staticmethod
    def _reference_year_from_query(
        query_info: Dict[str, Any],
    ):
        """
        如果查询提取出了end_year，
        使用end_year作为时效性参考年份；
        否则使用当前年份。
        """
        extracted_filters = (
            query_info.get(
                "extracted_filters",
                {},
            )
            or {}
        )

        end_year = extracted_filters.get(
            "end_year"
        )

        if end_year is None:
            return pd.Timestamp.now().year

        try:
            return int(end_year)
        except (TypeError, ValueError):
            return pd.Timestamp.now().year

    def search(
        self,
        query: str,
        top_k_vector=50,
        top_k_keyword=50,
        top_k_fused=30,
        final_top_k=10,
        fusion_strategy="rrf",
        max_chunks_per_doc=2,
        apply_reranking=True,
    ) -> Dict[str, Any]:
        if not isinstance(query, str):
            raise TypeError(
                "query必须是字符串。"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query不能为空。"
            )

        # 第一阶段：
        # 查询增强 + 向量召回 + BM25召回 + 融合
        retrieval_output = (
            self.multi_path_retriever.retrieve(
                query=query,
                top_k_vector=top_k_vector,
                top_k_keyword=top_k_keyword,
                top_k_fused=top_k_fused,
                fusion_strategy=(
                    fusion_strategy
                ),
                max_chunks_per_doc=(
                    max_chunks_per_doc
                ),
            )
        )

        query_info = retrieval_output[
            "query_info"
        ]

        fused_results = retrieval_output[
            "fused_results"
        ]

        rerank_query = (
            self._prepare_rerank_query(
                query_info
            )
        )

        reference_year = (
            self._reference_year_from_query(
                query_info
            )
        )

        # 第二阶段：Cross-Encoder重排
        if (
            apply_reranking
            and not fused_results.empty
        ):
            final_results = (
                self.reranker.rerank(
                    query_text=rerank_query,
                    candidates=fused_results,
                    top_k=final_top_k,
                    reference_year=(
                        reference_year
                    ),
                )
            )
        else:
            final_results = (
                fused_results
                .head(final_top_k)
                .copy()
                .reset_index(drop=True)
            )

            if not final_results.empty:
                final_results["final_rank"] = (
                    np.arange(
                        1,
                        len(final_results) + 1,
                    )
                )

        statistics = dict(
            retrieval_output.get(
                "statistics",
                {}
            )
        )

        statistics.update({
            "reranking_applied": bool(
                apply_reranking
            ),
            "reranker_model": (
                self.reranker.model_name
                if apply_reranking
                else None
            ),
            "rerank_candidate_count":
                int(len(fused_results)),
            "final_result_count":
                int(len(final_results)),
            "reference_year":
                int(reference_year),
        })

        return {
            "original_query": query,

            "query_info":
                query_info,

            "rerank_query":
                rerank_query,

            "vector_results":
                retrieval_output[
                    "vector_results"
                ],

            "bm25_results":
                retrieval_output[
                    "bm25_results"
                ],

            "fused_results":
                fused_results,

            "final_results":
                final_results,

            "statistics":
                statistics,
        }