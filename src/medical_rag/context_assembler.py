import math
import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import pandas as pd

'''
文档块与上下文组装器

'DocumentChunk` 统一不同检索器的数据格式。组装器执行：格式转换 → Jaccard去重 → 相关性与来源多样化排序 → token预算选择 → 完整句截断。

中文没有天然空格，因此英文按单词、中文按字符bigram计算Jaccard相似度。

`max_chunks_per_source`采用软限制：优先选择不同来源，仍有预算时再补充同来源证据。
'''

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    source: str = ""
    chunk_id: str = ""


class ContextAssembler:
    SCORE_FIELDS = (
        "final_score", "relevance_score", "fusion_score",
        "similarity", "score"
    )

    def __init__(
        self,
        tokenizer= None,
        tokenizer_name: Optional[str] = None,
        max_context_tokens: int = 6000,
        dedup_threshold: float = 0.85,
        max_chunks_per_source: int = 2,
        diversity_penalty: float = 0.15,
    ):
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens必须大于0")
        if not 0 <= dedup_threshold <= 1:
            raise ValueError("dedup_threshold必须在[0, 1]之间")
        if max_chunks_per_source <= 0 or diversity_penalty < 0:
            raise ValueError("来源上限必须为正数，惩罚不能为负数")
        self.tokenizer = tokenizer or self._load_tokenizer(tokenizer_name)
        self.max_context_tokens = max_context_tokens
        self.dedup_threshold = dedup_threshold
        self.max_chunks_per_source = max_chunks_per_source
        self.diversity_penalty = diversity_penalty

    @staticmethod
    # 加载tokenizer
    def _load_tokenizer(name):
        if not name:
            return None
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"无法从本地加载tokenizer：{name}") from exc
    # 估算token数
    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(self.tokenizer.encode(text))
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = re.sub(r"[\u4e00-\u9fff\s]", "", text)
        return chinese + math.ceil(len(other) / 4)

    @staticmethod
    # 清洗字段
    def _clean(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    @classmethod
    # 提取相关性分数
    def _score(cls, row):
        for key in cls.SCORE_FIELDS:
            try:
                score = float(row.get(key))
                if math.isfinite(score):
                    return score
            except (TypeError, ValueError):
                continue
        return 0.0

    @classmethod
    # 将字典转换为 DocumentChunk
    def _to_chunk(cls, row, index):
        source = cls._clean(
            row.get("source_title") or row.get("source")
            or row.get("journal") or row.get("doc_id") or "unknown"
        )
        chunk_id = cls._clean(
            row.get("vector_id") or row.get("chunk_id")
            or (f"{row.get('doc_id')}_chunk_{row.get('chunk_index')}"
                if row.get("doc_id") is not None else f"chunk_{index}")
        )
        return DocumentChunk(
            text=cls._clean(row.get("text", row.get("page_content", ""))),
            metadata={k: v for k, v in row.items()
                      if k not in {"text", "page_content"}},
            relevance_score=cls._score(row),
            source=source,
            chunk_id=chunk_id,
        )

    @classmethod
    # 转换输入文档集合
    def _convert_documents(cls, docs):
        if docs is None:
            return []
        if isinstance(docs, pd.DataFrame):
            docs = docs.to_dict(orient="records")
        elif isinstance(docs, Mapping):
            docs = [docs]
        result = []
        for index, item in enumerate(docs):
            if isinstance(item, DocumentChunk):
                chunk = item
            elif isinstance(item, Mapping):
                chunk = cls._to_chunk(item, index)
            elif hasattr(item, "page_content"):
                row = {"page_content": item.page_content,
                       **dict(getattr(item, "metadata", {}) or {})}
                chunk = cls._to_chunk(row, index)
            else:
                raise TypeError(f"不支持的文档类型：{type(item).__name__}")
            if chunk.text.strip():
                result.append(chunk)
        return result

    @staticmethod
    # 构建 Jaccard 特征
    def _features(text):
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        english = re.findall(r"[a-z0-9]+", normalized)
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        bigrams = [chinese[i:i + 2] for i in range(len(chinese) - 1)]
        return set(english + bigrams + ([chinese] if len(chinese) == 1 else []))

    @classmethod
    # 计算 Jaccard 相似度
    def _jaccard_similarity(cls, left, right):
        a, b = cls._features(left), cls._features(right)
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if a and b else 0.0

    # 文档去重
    def _deduplicate(self, chunks):
        unique, seen_ids = [], set()
        for chunk in sorted(chunks, key=lambda x: x.relevance_score, reverse=True):
            if chunk.chunk_id and chunk.chunk_id in seen_ids:
                continue
            if any(self._jaccard_similarity(chunk.text, kept.text)
                   >= self.dedup_threshold for kept in unique):
                continue
            unique.append(chunk)
            if chunk.chunk_id:
                seen_ids.add(chunk.chunk_id)
        return unique

    # 来源多样化排序
    def _diversified_order(self, chunks):
        remaining, ordered, counts = list(chunks), [], Counter()
        while remaining:
            best = max(
                remaining,
                key=lambda x: (
                    x.relevance_score - self.diversity_penalty * counts[x.source],
                    x.relevance_score,
                ),
            )
            ordered.append(best)
            counts[best.source] += 1
            remaining.remove(best)
        return ordered

    @staticmethod
    # 格式化文档块
    def _format_chunk(chunk, number):
        return (f"[证据 {number} | 来源: {chunk.source or 'unknown'} | "
                f"相关性: {chunk.relevance_score:.4f}]\n{chunk.text.strip()}")
    
    # 按完整句子截断
    def _truncate_at_sentence(self, text, budget):
        if budget <= 0:
            return ""
        if self.estimate_tokens(text) <= budget:
            return text.strip()
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimate_tokens(text[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        candidate = text[:low].rstrip()
        start = int(len(candidate) * 0.9)
        endings = list(re.finditer(
            r"[。！？.!?](?=\s|$|[\u4e00-\u9fff])", candidate[start:]
        ))
        if endings:
            candidate = candidate[:start + endings[-1].end()]
        else:
            paragraph = candidate.rfind("\n\n", start)
            if paragraph > 0:
                candidate = candidate[:paragraph]
        return candidate.strip()

    @staticmethod
    # 统计来源
    def _analyze_sources(chunks):
        return dict(Counter(x.source or "unknown" for x in chunks))

    # 组装上下文
    def assemble_context(self, retrieved_docs):
        chunks = self._convert_documents(retrieved_docs)
        unique = self._deduplicate(chunks)
        candidates = self._diversified_order(unique)

        preferred, overflow, counts = [], [], Counter()
        for chunk in candidates:
            if counts[chunk.source] < self.max_chunks_per_source:
                preferred.append(chunk)
                counts[chunk.source] += 1
            else:
                overflow.append(chunk)

        selected, parts, used = [], [], 0
        for chunk in preferred + overflow:
            separator = "\n\n" if parts else ""
            formatted = self._format_chunk(chunk, len(selected) + 1)
            remaining = (self.max_context_tokens - used
                         - self.estimate_tokens(separator))
            if remaining <= 0:
                break
            if self.estimate_tokens(formatted) > remaining:
                truncated = self._truncate_at_sentence(formatted, remaining)
                if truncated:
                    parts.append(separator + truncated)
                    selected.append(chunk)
                break
            parts.append(separator + formatted)
            selected.append(chunk)
            used += self.estimate_tokens(separator + formatted)

        final_context = "".join(parts).strip()
        metadata = {
            "total_chunks_retrieved": len(chunks),
            "unique_chunks_after_dedup": len(unique),
            "chunks_selected": len(selected),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected),
        }
        return {"context_text": final_context,
                "metadata": metadata,
                "selected_chunks": selected}