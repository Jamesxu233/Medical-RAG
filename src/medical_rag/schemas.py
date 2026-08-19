from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#医学查询器相关数据结构创建

@dataclass
#识别出的医学实体类
class MedicalEntity:
    entity_type: str
    text: str
    start: int
    end: int


@dataclass
#保存完整查询处理结果类
class EnhancedMedicalQuery:
    original_query: str
    cleaned_query: str

    entities: List[MedicalEntity]
    synonym_expansions: Dict[str, List[str]]

    vector_query: str
    keyword_query: str
    keyword_terms: List[str]

    where_filter: Optional[Dict[str, Any]]
    extracted_filters: Dict[str, Any]

    detected_language: str
    warnings: List[str]
