import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from medical_rag.config import ZH_TO_EN_TERMS



def normalize_medical_term(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def translate_known_medical_terms(query: str) -> str:
    result = query

    for chinese, english in sorted(
        ZH_TO_EN_TERMS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        result = result.replace(chinese, english)

    result = result.replace("？", "?")
    result = re.sub(r"\s+", " ", result).strip()

    return result


##MeSH词典加载器

class MeshLexicon:
    def __init__(
        self,
        alias_index_path: str,
        concepts_path: str,
        max_synonyms: int = 8,
    ):
        self.max_synonyms = max_synonyms

        with open(alias_index_path, "r", encoding="utf-8") as f:
            raw_alias_index = json.load(f)

        self.alias_index = {
            normalize_medical_term(alias): value
            for alias, value in raw_alias_index.items()
        }

        self.concepts = {}

        with open(concepts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                concept_id = self._first_value(
                    record,
                    [
                        "mesh_id",
                        "descriptor_ui",
                        "descriptor_id",
                        "concept_id",
                        "id",
                        "ui",
                    ],
                )

                if concept_id:
                    self.concepts[str(concept_id)] = record

        print(f"Loaded {len(self.alias_index):,} MeSH aliases")
        print(f"Loaded {len(self.concepts):,} MeSH concepts")

    @staticmethod
    def _first_value(
        record: Dict[str, Any],
        field_names: List[str],
    ) -> Optional[Any]:
        for field in field_names:
            value = record.get(field)

            if value not in (None, "", []):
                return value

        return None

    def _concept_terms(
        self,
        record: Dict[str, Any],
    ) -> List[str]:
        preferred = self._first_value(
            record,
            [
                "preferred_term",
                "descriptor_name",
                "preferred_name",
                "name",
                "label",
            ],
        )

        aliases = self._first_value(
            record,
            [
                "aliases",
                "synonyms",
                "entry_terms",
                "terms",
            ],
        )

        terms = []

        if preferred:
            terms.append(str(preferred))

        if isinstance(aliases, str):
            terms.append(aliases)

        elif isinstance(aliases, list):
            for item in aliases:
                if isinstance(item, str):
                    terms.append(item)

                elif isinstance(item, dict):
                    term = self._first_value(
                        item,
                        ["term", "name", "label", "text"],
                    )

                    if term:
                        terms.append(str(term))

        return self._deduplicate_terms(terms)

    @staticmethod
    def _deduplicate_terms(terms: List[str]) -> List[str]:
        output = []
        seen = set()

        for term in terms:
            cleaned = re.sub(r"\s+", " ", str(term)).strip()
            normalized = normalize_medical_term(cleaned)

            if cleaned and normalized not in seen:
                seen.add(normalized)
                output.append(cleaned)

        return output

    def lookup(self, term: str) -> List[str]:
        normalized_term = normalize_medical_term(term)
        target = self.alias_index.get(normalized_term)

        if target is None:
            return []

        # 情况1：alias -> MeSH ID
        if isinstance(target, str) and target in self.concepts:
            terms = self._concept_terms(self.concepts[target])

        # 情况2：alias -> preferred term
        elif isinstance(target, str):
            terms = [target]

        # 情况3：alias -> synonym list
        elif isinstance(target, list):
            terms = [str(item) for item in target]

        # 情况4：alias -> concept information
        elif isinstance(target, dict):
            concept_id = self._first_value(
                target,
                [
                    "mesh_id",
                    "descriptor_ui",
                    "concept_id",
                    "id",
                ],
            )

            if concept_id and str(concept_id) in self.concepts:
                terms = self._concept_terms(
                    self.concepts[str(concept_id)]
                )
            else:
                terms = self._concept_terms(target)

        else:
            terms = []

        # 不把输入词本身作为同义词再次返回
        terms = [
            value
            for value in self._deduplicate_terms(terms)
            if normalize_medical_term(value) != normalized_term
        ]

        return terms[:self.max_synonyms]

    def find_terms(
        self,
        query: str,
        max_ngram: int = 6,
    ) -> List[str]:
        """
        通过 n-gram 查找查询中存在的 MeSH 术语。
        避免遍历整个大型 alias_index。
        """
        normalized_query = normalize_medical_term(query)

        words = re.findall(
            r"[a-z0-9]+(?:'[a-z0-9]+)?",
            normalized_query,
        )

        matches = set()

        for size in range(min(max_ngram, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start:start + size])

                if phrase in self.alias_index:
                    matches.add(phrase)

        return sorted(
            matches,
            key=lambda value: (-len(value.split()), value),
        )