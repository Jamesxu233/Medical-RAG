import re
from dataclasses import asdict
from typing import Any,Dict, List, Optional
import pandas as pd

from medical_rag.config import MEDICAL_PATTERNS, MEDICAL_SYNONYMS
from medical_rag.schemas import EnhancedMedicalQuery, MedicalEntity
from medical_rag.mesh_terminology import normalize_medical_term


#医学查询处理器

class MedicalQueryProcessor:
    def __init__(
        self,
        synonyms: Optional[Dict[str, List[str]]] = None,
        patterns: Optional[Dict[str, re.Pattern]] = None,
        query_instruction: str = (
            "Represent this question for searching "
            "relevant passages: "
        ),
        corpus_language: str = "en",
        mesh_lexicon=None
    ):
        self.synonyms = synonyms or MEDICAL_SYNONYMS
        self.patterns = patterns or MEDICAL_PATTERNS
        self.query_instruction = query_instruction
        self.corpus_language = corpus_language
        self.mesh_lexicon = mesh_lexicon

    @staticmethod
    
    def _clean_query(query: str) -> str:
        #对用户输入的查询进行清理和标准化，去除多余空格和标点符号，并确保查询为非空字符串。
        if not isinstance(query, str):
            raise TypeError(
                "query必须是字符串。"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query不能为空。"
            )

        punctuation_map = {
            "，": ", ",
            "。": ". ",
            "；": "; ",
            "：": ": ",
            "？": "?",
            "（": "(",
            "）": ")"
        }

        for old, new in (
            punctuation_map.items()
        ):
            query = query.replace(old, new)

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        return query
    
    @staticmethod

    def _term_in_query(
        term: str,
        query: str
    ) -> bool:
        term = str(term).strip()
        query = str(query)

        if not term:
            return False

        # 英文、数字、连字符缩写使用单词边界
        if re.fullmatch(
            r"[A-Za-z0-9\- ]+",
            term
        ):
            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(term)
                + r"(?![A-Za-z0-9])"
            )

            return bool(
                re.search(
                    pattern,
                    query,
                    flags=re.IGNORECASE
                )
            )
            

        # 中文术语直接匹配
        return term in query

    @staticmethod
    def _detect_language(text: str) -> str:
        # 简单的语言检测：如果包含中文字符，则判定为中文，否则为英文。
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        return "en"

    def _identify_entities(
        self,
        query: str
    ) -> List[MedicalEntity]:
        """
        使用规则表达式识别药物、疾病、缩写和结局等实体。
        """
        entities = []
        seen = set()

        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(query):
                entity_text = match.group(0).strip()

                key = (
                    entity_type,
                    entity_text.lower(),
                    match.start(),
                    match.end()
                )

                if key in seen:
                    continue

                seen.add(key)

                entities.append(
                    MedicalEntity(
                        entity_type=entity_type,
                        text=entity_text,
                        start=match.start(),
                        end=match.end()
                    )
                )

        entities.sort(key=lambda item: item.start)

        return entities

    def _expand_synonyms(
        self,
        query: str,
        entities: List[MedicalEntity],
    ) -> Dict[str, List[str]]:
        expansions = {}

        # 1. 原有静态词典
        candidate_terms = [
            entity.text.strip()
            for entity in entities
            if entity.text and entity.text.strip()
        ]

        query_normalized = normalize_medical_term(query)

        for key in self.synonyms:
            if self._term_in_query(key, query):
                candidate_terms.append(key)

        # 2. 使用 MeSH 发现英文医学术语
        if self.mesh_lexicon is not None:
            candidate_terms.extend(
                self.mesh_lexicon.find_terms(query)
            )

        # 去重
        candidate_terms = list(dict.fromkeys(candidate_terms))

        for term in candidate_terms:
            synonyms = []

            # 静态同义词优先
            for key, values in self.synonyms.items():
                if (
                    normalize_medical_term(key)
                    == normalize_medical_term(term)
                ):
                    synonyms.extend(values)

            # 加入 MeSH 同义词
            if self.mesh_lexicon is not None:
                synonyms.extend(
                    self.mesh_lexicon.lookup(term)
                )

            # 去重并排除原始词
            unique_synonyms = []
            seen = {normalize_medical_term(term)}

            for synonym in synonyms:
                normalized = normalize_medical_term(synonym)

                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_synonyms.append(synonym)

            if unique_synonyms:
                expansions[term] = unique_synonyms[:8]

        return expansions

    @staticmethod
    def _unique_terms(
        terms: List[str]
    ) -> List[str]:
        """
        去重但保持原始顺序。
        """
        result = []
        seen = set()

        for term in terms:
            normalized = term.lower().strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            result.append(term.strip())

        return result

    def _build_semantic_query(
        self,
        cleaned_query: str,
        entities: List[MedicalEntity],
        expansions: Dict[str, List[str]]
    ) -> str:
        """
        向量查询增强: 构造用于向量检索的语义查询。

        对英文语料，增加英文术语扩展。
        """
        added_terms = []

        for entity in entities:
            if re.fullmatch(
                r"[A-Za-z0-9\-\s]+",
                entity.text
            ):
                added_terms.append(entity.text)

        for values in expansions.values():
            for value in values:
                if re.fullmatch(
                    r"[A-Za-z0-9\-\s]+",
                    value
                ):
                    added_terms.append(value)

        added_terms = self._unique_terms(added_terms)

        if added_terms:
            semantic_query = (
                f"{cleaned_query} "
                f"Related medical concepts: "
                f"{'; '.join(added_terms)}."
            )
        else:
            semantic_query = cleaned_query

        return semantic_query

    def _build_keyword_query(
        self,
        cleaned_query: str,
        entities: List[MedicalEntity],
        expansions: Dict[str, List[str]]
    ) -> tuple[str, List[str]]:
        """
        关键词查询增强: 为BM25/全文关键词检索准备查询词。
        """
        keyword_terms = [
            entity.text
            for entity in entities
        ]

        for term, synonyms in expansions.items():
            keyword_terms.append(term)
            keyword_terms.extend(synonyms)

        # 如果没有识别出实体，则保留原查询
        if not keyword_terms:
            keyword_terms = [cleaned_query]

        keyword_terms = self._unique_terms(keyword_terms)

        # OR语法适用于许多关键词检索引擎；
        # 实际使用时可按BM25组件要求调整。
        keyword_query = " OR ".join(
            f'"{term}"'
            if " " in term
            else term
            for term in keyword_terms
        )

        return keyword_query, keyword_terms

    @staticmethod
    def _extract_year_filter(
        query: str
    ) -> Dict[str, int]:
        """
        从查询中提取年份过滤条件。
        """

        current_year = (
            pd.Timestamp.now().year
        )

        # 近5年、过去5年、last 5 years
        recent_match = re.search(
            r"(?:近|最近|过去)\s*"
            r"(\d{1,2})\s*年"
            r"|(?:last|past)\s+"
            r"(\d{1,2})\s+years?",
            query,
            flags=re.IGNORECASE
        )

        if recent_match:
            years = int(
                recent_match.group(1)
                or recent_match.group(2)
            )

            if years <= 0:
                raise ValueError(
                    "时间范围必须大于0年。"
                )

            return {
                "start_year":
                    current_year - years + 1,
                "end_year":
                    current_year
            }

        # 2020年以来、since 2020
        since_match = re.search(
            r"((?:18|19|20)\d{2})"
            r"\s*年以来"
            r"|since\s+"
            r"((?:18|19|20)\d{2})",
            query,
            flags=re.IGNORECASE
        )

        if since_match:
            year = int(
                since_match.group(1)
                or since_match.group(2)
            )

            if year > current_year:
                raise ValueError(
                    "起始年份不能超过当前年份。"
                )

            return {
                "start_year": year,
                "end_year": current_year
            }

        # 2020-2024、2020至2024
        range_match = re.search(
            r"((?:18|19|20)\d{2})"
            r"\s*[-–—至到]\s*"
            r"((?:18|19|20)\d{2})"
            r"|between\s+"
            r"((?:18|19|20)\d{2})"
            r"\s+and\s+"
            r"((?:18|19|20)\d{2})",
            query,
            flags=re.IGNORECASE
        )

        if range_match:
            start_year = int(
                range_match.group(1)
                or range_match.group(3)
            )

            end_year = int(
                range_match.group(2)
                or range_match.group(4)
            )

            if start_year > end_year:
                raise ValueError(
                    "起始年份不能晚于结束年份。"
                )

            if end_year > current_year:
                raise ValueError(
                    "结束年份不能超过当前年份。"
                )

            return {
                "start_year": start_year,
                "end_year": end_year
            }

        return {}

    @staticmethod
    def _extract_journal_filter(
        query: str,
        known_journals: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        从查询中识别精确期刊名称。
        """
        if not known_journals:
            return None

        query_lower = query.lower()

        # 优先匹配较长名称，避免Nature先匹配到
        # Nature Communications的一部分。
        sorted_journals = sorted(
            known_journals,
            key=len,
            reverse=True
        )

        for journal in sorted_journals:
            if journal.lower() in query_lower:
                return journal

        return None

    def _extract_filters(
        self,
        query: str,
        known_journals: Optional[List[str]] = None
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        提取时间和期刊过滤条件，并转换成Chroma where格式。
        """
        extracted = {}

        year_filter = self._extract_year_filter(query)

        if year_filter:
            extracted.update(year_filter)

        journal = self._extract_journal_filter(
            query,
            known_journals
        )

        if journal:
            extracted["journal"] = journal

        chroma_conditions = []

        if journal:
            chroma_conditions.append(
                {"journal": journal}
            )

        if "start_year" in year_filter:
            chroma_conditions.append({
                "publication_year": {
                    "$gte": year_filter["start_year"]
                }
            })

        if "end_year" in year_filter:
            chroma_conditions.append({
                "publication_year": {
                    "$lte": year_filter["end_year"]
                }
            })

        if not chroma_conditions:
            where_filter = None
        elif len(chroma_conditions) == 1:
            where_filter = chroma_conditions[0]
        else:
            where_filter = {
                "$and": chroma_conditions
            }

        return extracted, where_filter

    def process(
        self,
        query: str,
        known_journals: Optional[List[str]] = None,
        add_instruction: bool = True
    ) -> EnhancedMedicalQuery:
        """
        处理医学查询并返回完整的查询增强结果。
        """
        cleaned_query = self._clean_query(query)
        language = self._detect_language(cleaned_query)

        entities = self._identify_entities(cleaned_query)

        expansions = self._expand_synonyms(
            cleaned_query,
            entities
        )

        semantic_query = self._build_semantic_query(
            cleaned_query,
            entities,
            expansions
        )

        if add_instruction:
            vector_query = (
                self.query_instruction + semantic_query
            )
        else:
            vector_query = semantic_query

        keyword_query, keyword_terms = (
            self._build_keyword_query(
                cleaned_query,
                entities,
                expansions
            )
        )

        extracted_filters, where_filter = (
            self._extract_filters(
                cleaned_query,
                known_journals
            )
        )

        warnings = []

        if (
            language == "zh"
            and self.corpus_language == "en"
        ):
            warnings.append(
                "查询包含中文，但当前BGE模型和PMC语料主要为英文；"
                "建议在向量检索前将完整查询翻译为英文。"
            )

        return EnhancedMedicalQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            entities=[
                asdict(entity)
                for entity in entities
            ],
            synonym_expansions=expansions,
            vector_query=vector_query,
            keyword_query=keyword_query,
            keyword_terms=keyword_terms,
            where_filter=where_filter,
            extracted_filters=extracted_filters,
            detected_language=language,
            warnings=warnings
        )