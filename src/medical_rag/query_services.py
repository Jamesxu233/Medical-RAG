from typing import Any, Dict
import json

from medical_rag.chroma_indexer import PubMedChromaIndexer
from medical_rag.query_processor import MedicalQueryProcessor
from medical_rag.retriever import MedicalChromaRetriever



#检索信息处理函数
def enhance_medical_query(
    query: str,
    query_processor: MedicalQueryProcessor,
    retriever: MedicalChromaRetriever,
    indexer: PubMedChromaIndexer,
) -> Dict[str, Any]:
    """
    处理用户医学查询，返回向量检索、BM25检索和
    Metadata Filter 所需的完整查询信息。

    处理流程：
    1. 基础清洗与语言识别
    2. 医学实体识别
    3. 静态词典和MeSH同义词扩展
    4. 中文查询的英文转换或英文术语回退
    5. 生成向量检索查询
    6. 生成BM25关键词查询
    7. 提取期刊、年份等元数据过滤条件

    注意：
    vector_query不包含BGE指令前缀。
    indexer.query()中的encode_query()会自动添加BGE指令，
    因此不能把bge_query_for_display传给indexer.query()。
    """

    if not isinstance(query, str):
        raise TypeError("query必须是字符串。")

    query = query.strip()

    if not query:
        raise ValueError("query不能为空。")

    if not isinstance(
        query_processor,
        MedicalQueryProcessor,
    ):
        raise TypeError(
            "query_processor必须是"
            "MedicalQueryProcessor实例。"
        )

    if not isinstance(
        retriever,
        MedicalChromaRetriever,
    ):
        raise TypeError(
            "retriever必须是"
            "MedicalChromaRetriever实例。"
        )

    if not isinstance(
        indexer,
        PubMedChromaIndexer,
    ):
        raise TypeError(
            "indexer必须是"
            "PubMedChromaIndexer实例。"
        )

    # 第一阶段：基础查询理解与增强
    #
    # add_instruction=False非常重要。
    # indexer.encode_query()会自动添加BGE指令，
    # 此处再次添加会造成指令重复。
    enhanced = query_processor.process(
        query=query,
        known_journals=retriever.known_journals,
        add_instruction=False,
    )

    # 第二阶段：准备实际向量查询
    #
    # 对英文查询，通常直接使用增强后的语义查询。
    # 对中文查询：
    # 1. 如果配置了query_translator，执行中英转换；
    # 2. 否则使用识别出的英文医学术语作为回退。
    vector_query, retriever_warnings = (
        retriever.prepare_vector_query(
            enhanced
        )
    )

    vector_query = str(vector_query).strip()

    if not vector_query:
        # 如果翻译或术语回退失败，
        # 至少使用清洗后的原始查询。
        vector_query = enhanced.cleaned_query

        retriever_warnings = list(
            retriever_warnings
        )

        retriever_warnings.append(
            "增强后的向量查询为空，"
            "已回退到清洗后的原始查询。"
        )

    # 仅用于展示实际送入BGE模型的完整查询。
    #
    # 不能将该字段再次传入indexer.query()，
    # 否则BGE指令会重复两次。
    bge_query_for_display = (
        indexer.QUERY_INSTRUCTION
        + vector_query
    )

    # BM25应优先使用keyword_terms。
    #
    # keyword_query中的OR只是方便展示，
    # 不能直接把OR当作BM25的普通查询词。
    keyword_terms = [
        str(term).strip()
        for term in enhanced.keyword_terms
        if str(term).strip()
    ]

    # 去重并保持原顺序
    keyword_terms = list(
        dict.fromkeys(keyword_terms)
    )

    # 如果没有识别出医学关键词，
    # BM25回退到清洗后的原始查询。
    if not keyword_terms:
        keyword_terms = [
            enhanced.cleaned_query
        ]

        retriever_warnings = list(
            retriever_warnings
        )

        retriever_warnings.append(
            "未生成有效医学关键词，"
            "BM25查询已回退到清洗后的原始查询。"
        )

    result = {
        # 原始与清洗后的查询
        "original_query":
            enhanced.original_query,

        "cleaned_query":
            enhanced.cleaned_query,

        "detected_language":
            enhanced.detected_language,

        # 医学查询理解结果
        "entities":
            enhanced.entities,

        "synonym_expansions":
            enhanced.synonym_expansions,

        # 向量检索实际使用的查询
        "vector_query":
            vector_query,

        # 仅用于调试和展示
        "bge_query_for_display":
            bge_query_for_display,

        # BM25检索字段
        "keyword_query":
            enhanced.keyword_query,

        "keyword_terms":
            keyword_terms,

        # Metadata Filter
        "extracted_filters":
            enhanced.extracted_filters,

        "where_filter":
            enhanced.where_filter,

        # 处理警告
        "warnings":
            list(dict.fromkeys(
                retriever_warnings
            )),
    }

    return result

#输出函数

def print_enhanced_query(
    result: Dict[str, Any],
) -> None:

    print("=" * 100)
    print("1. 原始查询")
    print(result["original_query"])

    print("\n2. 基础清洗结果")
    print(result["cleaned_query"])

    print("\n3. 检测语言")
    print(result["detected_language"])

    print("\n4. 识别医学实体")

    if result["entities"]:
        for entity in result["entities"]:
            print(
                f"- 类型: {entity['entity_type']}, "
                f"文本: {entity['text']}, "
                f"位置: [{entity['start']}, "
                f"{entity['end']}]"
            )
    else:
        print("未识别出规则医学实体。")

    print("\n5. 医学同义词扩展")

    if result["synonym_expansions"]:
        for term, synonyms in (
            result["synonym_expansions"].items()
        ):
            print(f"- {term}: {synonyms}")
    else:
        print("未产生同义词扩展。")

    print("\n6. 向量检索查询")
    print(result["vector_query"])

    print("\n7. BGE完整查询（仅用于检查）")
    print(result["bge_query_for_display"])

    print("\n8. 关键词检索查询")
    print(result["keyword_query"])

    print("\n9. 关键词列表")
    print(result["keyword_terms"])

    print("\n10. 提取出的过滤条件")
    print(
        json.dumps(
            result["extracted_filters"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n11. Chroma过滤条件")
    print(
        json.dumps(
            result["where_filter"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n12. 警告信息")

    if result["warnings"]:
        for warning in result["warnings"]:
            print("-", warning)
    else:
        print("无警告。")