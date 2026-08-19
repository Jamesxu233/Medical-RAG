import hashlib
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter




class PubMedFullTextChunker:
    def __init__(
        self,
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        chunk_size=500,          # 留buffer，避免超过512
        chunk_overlap=80,
        min_text_tokens=30,
        max_token_limit=512,
        add_title_to_each_chunk=True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_text_tokens = min_text_tokens
        self.max_token_limit = max_token_limit
        self.add_title_to_each_chunk = add_title_to_each_chunk

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                ""
            ],
        )

    def _count_tokens(self, text):
        return len(
            self.tokenizer.encode(
                str(text),
                add_special_tokens=False
            )
        )

    def _clean_text(self, text):
        text = str(text)

        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            " ",
            text
        )

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _make_doc_id(self, row, index):
        pmid = str(row.get("pmid", "")).strip()

        if pmid and pmid.lower() not in ["nan", "none", ""]:
            return f"PMID_{pmid}"

        title = str(row.get("title", "")).strip()
        raw_id = f"{index}_{title}"

        hashed = hashlib.md5(
            raw_id.encode("utf-8")
        ).hexdigest()[:12]

        return f"DOC_{hashed}"

    def _make_chunk_id(self, doc_id, chunk_index):
        return f"{doc_id}_chunk_{chunk_index:04d}"

    def _merge_short_tail_chunks(self, chunks):
        """
        将过短chunk合并到前一个chunk。
        主要用于处理文末残片，如 Funding、Not Determined、References 残片等。
        """
        merged = []

        for chunk in chunks:
            chunk = chunk.strip()

            if not chunk:
                continue

            if (
                len(merged) > 0
                and self._count_tokens(chunk) < self.min_text_tokens
            ):
                candidate = merged[-1] + "\n\n" + chunk

                # 如果合并后不超过上限，则合并
                if self._count_tokens(candidate) <= self.max_token_limit:
                    merged[-1] = candidate
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)

        return merged

    def _enforce_token_limit(self, text):
        """
        对极少数超过max_token_limit的chunk进行二次token级切割。
        """
        token_count = self._count_tokens(text)

        if token_count <= self.max_token_limit:
            return [text]

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        step = self.max_token_limit - self.chunk_overlap

        for start in range(0, len(tokens), step):
            sub_tokens = tokens[start:start + self.max_token_limit]
            sub_text = self.tokenizer.decode(
                sub_tokens,
                skip_special_tokens=True
            )
            chunks.append(sub_text)

            if start + self.max_token_limit >= len(tokens):
                break

        return chunks

    def prepare_dataframe(
        self,
        df_raw,
        full_text_col="full_text"
    ):
        df = df_raw.copy()

        required_cols = [
            "title",
            "journal",
            "pub_date",
            "pmid",
            full_text_col
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        df["title"] = df["title"].fillna("").apply(self._clean_text)
        df["journal"] = df["journal"].fillna("").apply(self._clean_text)
        df["pmid"] = df["pmid"].fillna("").astype(str).str.strip()
        df[full_text_col] = df[full_text_col].fillna("").apply(self._clean_text)

        df["doc_id"] = [
            self._make_doc_id(row, idx)
            for idx, row in df.iterrows()
        ]

        df["full_text_token_count"] = df[full_text_col].apply(
            self._count_tokens
        )

        df = df[
            df["full_text_token_count"] >= self.min_text_tokens
        ].copy()

        return df

    def split_document(
        self,
        document,
        full_text_col="full_text"
    ):
        doc_id = document["doc_id"]
        title = str(document.get("title", "")).strip()
        full_text = str(document.get(full_text_col, "")).strip()

        title_prefix = f"Title: {title}\n\n"

        full_text_token_count = self._count_tokens(full_text)
        full_text_with_title = title_prefix + full_text
        total_token_count = self._count_tokens(full_text_with_title)

        # 情况1：全文 + 标题没有超过限制，整体不分割
        if total_token_count <= self.max_token_limit:
            text = full_text_with_title

            return [
                {
                    "chunk_id": doc_id,
                    "text": text,
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "source_title": title,
                    "journal": document.get("journal", ""),
                    "pub_date": document.get("pub_date", ""),
                    "pmid": document.get("pmid", ""),
                    "token_count": self._count_tokens(text),
                    "split_strategy": "no_split",
                }
            ]

        # 情况2：只切 full_text，不让 title 单独成为chunk
        raw_chunks = self.splitter.split_text(full_text)

        # 合并过短尾块
        raw_chunks = self._merge_short_tail_chunks(raw_chunks)

        # 给每个chunk加标题，提升检索可解释性
        chunk_texts = []

        for chunk in raw_chunks:
            if self.add_title_to_each_chunk:
                chunk_text = title_prefix + chunk
            else:
                chunk_text = chunk

            # 防止加标题后超过512
            safe_chunks = self._enforce_token_limit(chunk_text)
            chunk_texts.extend(safe_chunks)

        # 再次过滤空chunk
        chunk_texts = [
            c.strip()
            for c in chunk_texts
            if c and c.strip()
        ]

        chunk_data = []

        for i, chunk_text in enumerate(chunk_texts):
            chunk_data.append(
                {
                    "chunk_id": self._make_chunk_id(doc_id, i),
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "source_title": title,
                    "journal": document.get("journal", ""),
                    "pub_date": document.get("pub_date", ""),
                    "pmid": document.get("pmid", ""),
                    "token_count": self._count_tokens(chunk_text),
                    "split_strategy": "sliding_window",
                }
            )

        return chunk_data

    def split_dataframe(
        self,
        df_raw,
        full_text_col="full_text",
        output_path="pubmed_fulltext_chunks.parquet",
        stats_path="pubmed_fulltext_chunk_stats.json",
        data_split="full_text"
    ):
        df = self.prepare_dataframe(
            df_raw,
            full_text_col=full_text_col
        )

        all_chunks = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Splitting documents"
        ):
            chunks = self.split_document(
                row,
                full_text_col=full_text_col
            )
            all_chunks.extend(chunks)

        chunks_df = pd.DataFrame(all_chunks)

        output_path = Path(output_path)
        stats_path = Path(stats_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        stats_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if output_path.suffix == ".csv":
            chunks_df.to_csv(output_path, index=False)

        elif output_path.suffix == ".jsonl":
            chunks_df.to_json(
                output_path,
                orient="records",
                lines=True,
                force_ascii=False
            )

        else:
            chunks_df.to_parquet(output_path, index=False)

        documents_split = (
            chunks_df.groupby("doc_id")["total_chunks"]
            .first()
            .gt(1)
            .sum()
        )

        empty_chunks = (
            chunks_df["text"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        short_chunks = (
            chunks_df["token_count"] < self.min_text_tokens
        ).sum()

        over_limit_chunks = (
            chunks_df["token_count"] > self.max_token_limit
        ).sum()

        stats = {
            "processed_date": pd.Timestamp.now().isoformat(),
            "data_split": data_split,
            "original_documents": len(df_raw),
            "valid_documents": len(df),
            "total_chunks": len(chunks_df),
            "chunks_per_doc": len(chunks_df) / len(df) if len(df) > 0 else 0,
            "documents_split": int(documents_split),
            "documents_split_rate": float(documents_split / len(df)) if len(df) > 0 else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_token_limit": self.max_token_limit,
            "min_text_tokens": self.min_text_tokens,
            "empty_chunks": int(empty_chunks),
            "short_chunks": int(short_chunks),
            "chunks_over_token_limit": int(over_limit_chunks),
            "output_file": str(output_path),
            "split_strategy_counts": chunks_df["split_strategy"].value_counts().to_dict(),
            "token_count_summary": chunks_df["token_count"].describe().to_dict(),
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                stats,
                f,
                indent=4,
                ensure_ascii=False
            )

        return chunks_df, stats