from typing import Callable, Dict, List, Optional, Any
import re
import pandas as pd
import bm25s
import Stemmer
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import json

from medical_rag.chroma_indexer import PubMedChromaIndexer
from medical_rag.query_processor import MedicalQueryProcessor
from medical_rag.schemas import EnhancedMedicalQuery


#Chroma风格元数据过滤
def metadata_matches(
    row: pd.Series,
    where_filter: Optional[Dict[str, Any]],
) -> bool:
    if not where_filter:
        return True

    if "$and" in where_filter:
        return all(
            metadata_matches(row, condition)
            for condition in where_filter["$and"]
        )

    if "$or" in where_filter:
        return any(
            metadata_matches(row, condition)
            for condition in where_filter["$or"]
        )

    for field, expected in where_filter.items():
        actual = row.get(field)

        if pd.isna(actual):
            return False

        if isinstance(expected, dict):
            for operator, target in expected.items():
                try:
                    if operator == "$gte" and actual < target:
                        return False
                    if operator == "$lte" and actual > target:
                        return False
                    if operator == "$gt" and actual <= target:
                        return False
                    if operator == "$lt" and actual >= target:
                        return False
                    if operator == "$ne" and actual == target:
                        return False
                    if operator == "$in" and actual not in target:
                        return False
                    if operator == "$nin" and actual in target:
                        return False
                except TypeError:
                    return False
        elif actual != expected:
            return False

    return True


#医学信息检索器

class MedicalChromaRetriever:
    """
    MedicalQueryProcessor
            +
    PubMedChromaIndexer
            =
    医学智能向量检索器
    """

    def __init__(
        self,
        query_processor:
            MedicalQueryProcessor,
        indexer:
            PubMedChromaIndexer,
        known_journals:
            Optional[List[str]] = None,
        query_translator:
            Optional[
                Callable[[str], str]
            ] = None
    ):
        self.query_processor = (
            query_processor
        )

        self.indexer = indexer

        self.known_journals = (
            known_journals or []
        )

        self.query_translator = (
            query_translator
        )

    @staticmethod
    def _english_expansion_terms(
        synonym_expansions:
            Dict[str, List[str]]
    ) -> List[str]:

        terms = []
        seen = set()

        for values in (
            synonym_expansions.values()
        ):
            for value in values:
                value = str(value).strip()

                if not re.fullmatch(
                    r"[A-Za-z0-9\-\s]+",
                    value
                ):
                    continue

                normalized = value.lower()

                if normalized in seen:
                    continue

                seen.add(normalized)
                terms.append(value)

        return terms

    def prepare_vector_query(
        self,
        enhanced_query: EnhancedMedicalQuery,
    ) -> tuple[str, List[str]]:

        warnings = list(
            enhanced_query.warnings
        )

        vector_query = (
            enhanced_query.vector_query
        )

        # 英文语料 + 中文查询
        if (
            enhanced_query.detected_language
            == "zh"
            and self.query_processor
            .corpus_language == "en"
        ):
            english_terms = (
                self._english_expansion_terms(
                    enhanced_query
                    .synonym_expansions
                )
            )

            if self.query_translator:
                translated = (
                    self.query_translator(
                        enhanced_query
                        .cleaned_query
                    )
                ).strip()

                vector_query = translated

                if english_terms:
                    vector_query += (
                        " Related medical "
                        "concepts: "
                        + "; ".join(
                            english_terms
                        )
                        + "."
                    )

            elif english_terms:
                # 未配置翻译器时，
                # 至少使用识别出的英文标准术语
                vector_query = (
                    "Medical evidence about "
                    + "; ".join(
                        english_terms
                    )
                    + "."
                )

                warnings.append(
                    "未配置中文到英文翻译器；"
                    "本次向量检索主要使用"
                    "英文医学同义词。"
                )

            else:
                warnings.append(
                    "中文查询未识别出英文术语，"
                    "使用英文BGE模型时检索效果"
                    "可能较差。"
                )

        return vector_query, warnings

    @staticmethod
    def _diversify_results(
        results_df: pd.DataFrame,
        n_results: int,
        max_chunks_per_doc: int
    ) -> pd.DataFrame:

        if results_df.empty:
            return results_df

        if "doc_id" not in (
            results_df.columns
        ):
            result = (
                results_df
                .head(n_results)
                .copy()
            )

            result["rank"] = range(
                1,
                len(result) + 1
            )

            return result.reset_index(
                drop=True
            )

        selected = []
        doc_counts = {}

        ordered = (
            results_df
            .sort_values(
                by="similarity",
                ascending=False
            )
        )

        for _, row in ordered.iterrows():
            doc_id = str(
                row.get("doc_id", "")
            )

            count = doc_counts.get(
                doc_id,
                0
            )

            if count >= max_chunks_per_doc:
                continue

            selected.append(row)

            doc_counts[doc_id] = (
                count + 1
            )

            if len(selected) >= n_results:
                break

        if not selected:
            return pd.DataFrame(
                columns=results_df.columns
            )

        final_df = pd.DataFrame(
            selected
        ).reset_index(drop=True)

        final_df["rank"] = range(
            1,
            len(final_df) + 1
        )

        return final_df

    def search(
        self,
        query: str,
        n_results: int = 10,
        candidate_multiplier: int = 5,
        max_chunks_per_doc: int = 2,
        diversify_documents: bool = True
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

        if (
            not isinstance(n_results, int)
            or isinstance(n_results, bool)
            or n_results <= 0
        ):
            raise ValueError(
                "n_results必须是大于0的整数。"
            )

        if (
            not isinstance(
                candidate_multiplier,
                int
            )
            or isinstance(
                candidate_multiplier,
                bool
            )
            or candidate_multiplier <= 0
        ):
            raise ValueError(
                "candidate_multiplier必须是"
                "大于0的整数。"
            )

        if (
            not isinstance(
                max_chunks_per_doc,
                int
            )
            or isinstance(
                max_chunks_per_doc,
                bool
            )
            or max_chunks_per_doc <= 0
        ):
            raise ValueError(
                "max_chunks_per_doc必须是"
                "大于0的整数。"
            )

        # 关键：不在此处添加BGE指令
        enhanced = (
            self.query_processor.process(
                query=query,
                known_journals=(
                    self.known_journals
                ),
                add_instruction=False
            )
        )

        vector_query, warnings = (
            self.prepare_vector_query(
                enhanced
            )
        )

        candidate_k = max(
            n_results,
            n_results
            * candidate_multiplier
        )

        raw_results = (
            self.indexer.query(
                query_text=vector_query,
                n_results=candidate_k,
                where_filter=(
                    enhanced.where_filter
                )
            )
        )

        if diversify_documents:
            final_results = (
                self._diversify_results(
                    raw_results,
                    n_results=n_results,
                    max_chunks_per_doc=(
                        max_chunks_per_doc
                    )
                )
            )
        else:
            final_results = (
                raw_results
                .head(n_results)
                .copy()
                .reset_index(drop=True)
            )

            if not final_results.empty:
                final_results["rank"] = (
                    range(
                        1,
                        len(final_results) + 1
                    )
                )

        return {
            "original_query": query,
            "cleaned_query":
                enhanced.cleaned_query,
            "vector_query":
                vector_query,
            "keyword_query":
                enhanced.keyword_query,
            "keyword_terms":
                enhanced.keyword_terms,
            "entities":
                enhanced.entities,
            "synonym_expansions":
                enhanced.synonym_expansions,
            "extracted_filters":
                enhanced.extracted_filters,
            "where_filter":
                enhanced.where_filter,
            "detected_language":
                enhanced.detected_language,
            "warnings": warnings,
            "candidate_count":
                len(raw_results),
            "result_count":
                len(final_results),
            "results": final_results
        }
    


#BM25检索索引器

class MedicalBM25Retriever:
    def __init__(
        self,
        index_directory=r"F:\RAG\data\pubmed_bm25_index",
    ):
        self.index_directory = Path(index_directory)

        self.bm25_path = (
            self.index_directory / "bm25s_index"
        )

        self.mapping_path = (
            self.index_directory
            / "bm25_chunk_mapping.parquet"
        )

        self.stats_path = (
            self.index_directory
            / "bm25_index_stats.json"
        )

        self.stemmer = Stemmer.Stemmer("english")

        self.bm25 = None
        self.mapping_df = None

    @staticmethod
    def _prepare_dataframe(
        chunks_df: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "text",
            "doc_id",
            "chunk_index",
        }

        missing = (
            required_columns
            - set(chunks_df.columns)
        )

        if missing:
            raise ValueError(
                f"BM25输入缺少字段：{sorted(missing)}"
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

        df["chunk_index"] = pd.to_numeric(
            df["chunk_index"],
            errors="coerce",
        ).fillna(0).astype(int)

        # 与 PubMedChromaIndexer 保持一致
        df["vector_id"] = (
            df["doc_id"]
            + "_chunk_"
            + df["chunk_index"].map(
                lambda value: f"{value:04d}"
            )
        )

        if "publication_year" not in df.columns:
            if "pub_date" in df.columns:
                df["publication_year"] = (
                    df["pub_date"]
                    .astype(str)
                    .str.extract(
                        r"\b((?:18|19|20)\d{2})\b"
                    )[0]
                )

                df["publication_year"] = (
                    pd.to_numeric(
                        df["publication_year"],
                        errors="coerce",
                    )
                )
            else:
                df["publication_year"] = np.nan

        duplicate_count = (
            df["vector_id"].duplicated().sum()
        )

        if duplicate_count:
            raise ValueError(
                f"存在{duplicate_count}个重复vector_id"
            )

        return df.reset_index(drop=True)

    @staticmethod
    def get_chroma_ids(
        collection,
        batch_size=50_000,
    ) -> set:
        """
        获取Chroma中真正已经完成索引的vector_id。
        用于保证BM25和Chroma检索相同的文档集合。
        """
        total = collection.count()
        indexed_ids = set()

        for offset in tqdm(
            range(0, total, batch_size),
            desc="Reading Chroma IDs",
        ):
            batch = collection.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=[],
            )

            indexed_ids.update(batch["ids"])

        return indexed_ids

    def build(
        self,
        chunks_df: pd.DataFrame,
        chroma_collection=None,
        align_with_chroma=True,
    ) -> Dict[str, Any]:
        self.index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        df = self._prepare_dataframe(chunks_df)

        original_count = len(df)

        # 推荐启用，避免BM25检索到尚未进入Chroma的Chunk
        if (
            align_with_chroma
            and chroma_collection is not None
        ):
            chroma_ids = self.get_chroma_ids(
                chroma_collection
            )

            df = df[
                df["vector_id"].isin(chroma_ids)
            ].copy().reset_index(drop=True)

        print(
            f"Original chunks: {original_count:,}"
        )
        print(
            f"BM25 indexed chunks: {len(df):,}"
        )

        corpus = df["text"].tolist()

        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=True,
        )

        self.bm25 = bm25s.BM25(
            method="lucene"
        )

        self.bm25.index(
            corpus_tokens,
            show_progress=True,
        )

        mapping_columns = [
            column
            for column in [
                "vector_id",
                "chunk_id",
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
                "text",
            ]
            if column in df.columns
        ]

        self.mapping_df = (
            df[mapping_columns].copy()
        )

        self.mapping_df.to_parquet(
            self.mapping_path,
            index=False,
        )

        self.bm25.save(
            str(self.bm25_path)
        )

        stats = {
            "index_type": "BM25S",
            "original_chunks": int(original_count),
            "indexed_chunks": int(len(df)),
            "aligned_with_chroma": bool(
                align_with_chroma
                and chroma_collection is not None
            ),
            "built_at": pd.Timestamp.now().isoformat(),
            "index_directory": str(
                self.index_directory.resolve()
            ),
        }

        with self.stats_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                stats,
                file,
                ensure_ascii=False,
                indent=4,
            )

        return stats

    def load(self, mmap=True):
        if not self.bm25_path.exists():
            raise FileNotFoundError(
                f"BM25索引不存在：{self.bm25_path}"
            )

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"BM25映射文件不存在："
                f"{self.mapping_path}"
            )

        self.bm25 = bm25s.BM25.load(
            str(self.bm25_path),
            mmap=mmap,
        )

        self.mapping_df = pd.read_parquet(
            self.mapping_path
        )

        print(
            f"BM25 loaded: "
            f"{len(self.mapping_df):,} chunks"
        )

        return self

    def search(
        self,
        keyword_terms,
        top_k=50,
        where_filter=None,
        candidate_multiplier=20,
    ) -> pd.DataFrame:
        if (
            self.bm25 is None
            or self.mapping_df is None
        ):
            raise RuntimeError(
                "请先build()或load() BM25索引。"
            )

        if isinstance(keyword_terms, str):
            query_text = re.sub(
                r"\bOR\b",
                " ",
                keyword_terms,
                flags=re.IGNORECASE,
            ).replace('"', " ")
        else:
            query_text = " ".join(
                str(term)
                for term in keyword_terms
                if str(term).strip()
            )

        query_text = re.sub(
            r"\s+",
            " ",
            query_text,
        ).strip()

        if not query_text:
            return pd.DataFrame()

        query_tokens = bm25s.tokenize(
            [query_text],
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=False,
        )

        candidate_k = min(
            max(
                top_k * candidate_multiplier,
                top_k,
            ),
            len(self.mapping_df),
        )

        positions, scores = (
            self.bm25.retrieve(
                query_tokens,
                k=candidate_k,
                show_progress=False,
            )
        )

        positions = (
            np.asarray(positions).reshape(-1)
        )

        scores = (
            np.asarray(scores).reshape(-1)
        )

        records = []

        for initial_rank, (
            position,
            score,
        ) in enumerate(
            zip(positions, scores),
            start=1,
        ):
            position = int(position)

            if not (
                0 <= position
                < len(self.mapping_df)
            ):
                continue

            row = self.mapping_df.iloc[position]

            if not metadata_matches(
                row,
                where_filter,
            ):
                continue

            record = row.to_dict()

            record.update({
                "bm25_rank": len(records) + 1,
                "bm25_initial_rank":
                    initial_rank,
                "bm25_score": float(score),
                "retrieval_path": "bm25",
            })

            records.append(record)

            if len(records) >= top_k:
                break

        return pd.DataFrame(records)
