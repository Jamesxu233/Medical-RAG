import logging
import re
import time
import json

from medical_rag.prompts import EVIDENCE_SCHEMA
from medical_rag.config import DISCLAIMER
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

from medical_rag.llm_generator import (
    logger,
    OllamaGenerationError,
    LLMGenerator
)

#MedicalGenerationPipeline流水线

class MedicalGenerationPipeline:
    """
    医学 RAG 生成流水线。

    流程：
        1. 上下文组装
        2. 证据评估与上下文筛选
        3. 生成答案草稿
        4. 批判性审查
        5. 最终答案组装
        6. 引用、格式和免责声明后处理
    """

    def __init__(
        self,
        context_assembler,
        prompt_stages: Dict[str, Any],
        llm_generator: LLMGenerator,
        logger_instance: Optional[
            logging.Logger
        ] = None,
    ):
        required_stages = {
            "evidence_evaluator",
            "answer_generator",
            "critical_reviewer",
            "final_assembler",
        }

        missing_stages = (
            required_stages
            - set(prompt_stages)
        )

        if missing_stages:
            raise ValueError(
                "缺少提示词阶段："
                f"{sorted(missing_stages)}"
            )

        self.context_assembler = (
            context_assembler
        )

        self.prompt_stages = (
            prompt_stages
        )

        self.llm = llm_generator

        self.logger = (
            logger_instance
            or logger
        )

    # --------------------------------------------------------
    # 提示词渲染
    # --------------------------------------------------------

    def _render_stage(
        self,
        stage_key: str,
        **values,
    ) -> Dict[str, Any]:
        stage = self.prompt_stages[
            stage_key
        ]

        try:
            user_prompt = (
                stage.user_prompt_template
                .format(**values)
            )

        except KeyError as exc:
            raise ValueError(
                f"{stage_key}缺少模板变量："
                f"{exc.args[0]}"
            ) from exc

        return {
            "name": stage.name,
            "system_prompt": (
                stage.system_prompt
            ),
            "user_prompt": user_prompt,
            "temperature": (
                stage.temperature
            ),
            "max_tokens": (
                stage.max_tokens
            ),
        }

    # --------------------------------------------------------
    # 调用单个生成阶段
    # --------------------------------------------------------

    def _run_stage(
        self,
        stage_key: str,
        stage_times: Dict[str, float],
        token_counts: Dict[
            str,
            Dict[str, int],
        ],
        stage_success: Dict[str, bool],
        require_json: bool = False,
        json_schema: Optional[
            Dict[str, Any]
        ] = None,
        **template_values,
    ) -> Optional[Dict[str, Any]]:
        stage_config = self._render_stage(
            stage_key,
            **template_values,
        )

        start_time = time.perf_counter()

        try:
            response = self.llm.generate(
                prompt=stage_config[
                    "user_prompt"
                ],
                system_prompt=stage_config[
                    "system_prompt"
                ],
                temperature=stage_config[
                    "temperature"
                ],
                max_tokens=stage_config[
                    "max_tokens"
                ],
                require_json=require_json,
                json_schema=json_schema,
            )

            success = bool(
                response.get("text")
            )

            if require_json:
                success = (
                    success
                    and response.get(
                        "parsed_json"
                    )
                    is not None
                )

            stage_success[
                stage_key
            ] = success

            metrics = response.get(
                "metrics",
                {},
            )

            token_counts[stage_key] = {
                "prompt_tokens": int(
                    metrics.get(
                        "prompt_tokens",
                        0,
                    )
                ),
                "output_tokens": int(
                    metrics.get(
                        "output_tokens",
                        0,
                    )
                ),
                "total_tokens": int(
                    metrics.get(
                        "prompt_tokens",
                        0,
                    )
                    + metrics.get(
                        "output_tokens",
                        0,
                    )
                ),
            }

            return response

        except Exception as exc:
            stage_success[
                stage_key
            ] = False

            token_counts[stage_key] = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

            self.logger.exception(
                "阶段失败 | stage=%s | error=%s",
                stage_key,
                exc,
            )

            return None

        finally:
            stage_times[stage_key] = round(
                time.perf_counter()
                - start_time,
                4,
            )

    # --------------------------------------------------------
    # 证据 ID 提取
    # --------------------------------------------------------

    @staticmethod
    def _normalize_evidence_ids(
        values,
    ) -> List[int]:
        """
        将模型返回的证据ID统一转换为整数。
        """

        if values is None:
            return []

        if not isinstance(
            values,
            (list, tuple, set),
        ):
            values = [values]

        result = []

        for value in values:
            if isinstance(value, bool):
                continue

            if isinstance(value, int):
                result.append(value)
                continue

            text = str(value)

            matches = re.findall(
                r"(?:证据\s*)?(\d+)",
                text,
                flags=re.IGNORECASE,
            )

            result.extend(
                int(match)
                for match in matches
            )

        return list(dict.fromkeys(
            identifier
            for identifier in result
            if identifier > 0
        ))

    @classmethod
    def _extract_evidence_ids(
        cls,
        evaluation,
    ) -> List[int]:
        """
        从评估JSON中提取相关证据编号。
        """

        if not isinstance(
            evaluation,
            Mapping,
        ):
            return []

        candidate_keys = (
            "relevant_evidence_ids",
            "relevant_document_ids",
            "selected_evidence_ids",
            "selected_ids",
            "document_ids",
        )

        for key in candidate_keys:
            if key in evaluation:
                ids = (
                    cls._normalize_evidence_ids(
                        evaluation[key]
                    )
                )

                if ids:
                    return ids

        # 兼容 evidence=[{"id": 1, "relevant": true}]
        evidence_items = evaluation.get(
            "evidence",
            [],
        )

        selected = []

        if isinstance(
            evidence_items,
            list,
        ):
            for item in evidence_items:
                if not isinstance(
                    item,
                    Mapping,
                ):
                    continue

                relevant = item.get(
                    "relevant",
                    True,
                )

                if relevant is False:
                    continue

                selected.extend(
                    cls._normalize_evidence_ids(
                        item.get(
                            "id",
                            item.get(
                                "evidence_id"
                            ),
                        )
                    )
                )

        return list(dict.fromkeys(
            selected
        ))

    # --------------------------------------------------------
    # 根据评估结果筛选上下文
    # --------------------------------------------------------

    def _filter_context_by_evaluation(
        self,
        context_result: Dict[str, Any],
        evaluation: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        使用评估结果中的证据编号筛选上下文。

        如果评估结果无有效ID，则保留完整上下文，
        避免因模型格式错误导致上下文全部丢失。
        """

        selected_chunks = context_result[
            "selected_chunks"
        ]

        evidence_ids = (
            self._extract_evidence_ids(
                evaluation
            )
        )

        valid_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if 1 <= evidence_id
            <= len(selected_chunks)
        ]

        if not valid_ids:
            return {
                "context_text": (
                    context_result[
                        "context_text"
                    ]
                ),
                "selected_chunks": (
                    selected_chunks
                ),
                "selected_evidence_ids": list(
                    range(
                        1,
                        len(selected_chunks) + 1,
                    )
                ),
                "filter_applied": False,
            }

        filtered_parts = []
        filtered_chunks = []

        for new_number, evidence_id in enumerate(
            valid_ids,
            start=1,
        ):
            chunk = selected_chunks[
                evidence_id - 1
            ]

            filtered_chunks.append(
                chunk
            )

            # 保留原证据编号，确保引用与评估结果一致
            filtered_parts.append(
                self.context_assembler
                ._format_chunk(
                    chunk,
                    evidence_id,
                )
            )

        return {
            "context_text": (
                "\n\n".join(
                    filtered_parts
                )
            ),
            "selected_chunks": (
                filtered_chunks
            ),
            "selected_evidence_ids": (
                valid_ids
            ),
            "filter_applied": True,
        }

    # --------------------------------------------------------
    # 格式化来源
    # --------------------------------------------------------

    @staticmethod
    def _format_sources(
        chunks,
    ) -> List[Dict[str, Any]]:
        sources = []

        for evidence_id, chunk in enumerate(
            chunks,
            start=1,
        ):
            metadata = dict(
                chunk.metadata or {}
            )

            sources.append({
                "evidence_id": evidence_id,
                "citation": (
                    f"[证据 {evidence_id}]"
                ),
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "relevance_score": round(
                    float(
                        chunk.relevance_score
                    ),
                    6,
                ),
                "journal": metadata.get(
                    "journal",
                    "",
                ),
                "publication_year": (
                    metadata.get(
                        "publication_year",
                        "",
                    )
                ),
                "pmid": metadata.get(
                    "pmid",
                    "",
                ),
                "doc_id": metadata.get(
                    "doc_id",
                    "",
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    "",
                ),
            })

        return sources

    # --------------------------------------------------------
    # 回答后处理
    # --------------------------------------------------------

    @classmethod
    def _postprocess_answer(
        cls,
        answer: str,
        sources: Sequence[
            Mapping[str, Any]
        ],
        add_disclaimer: bool = True,
    ) -> str:
        """
        统一引用格式、整理空行、补充来源和免责声明。
        """

        answer = str(
            answer or ""
        ).strip()

        # 统一引用格式
        answer = re.sub(
            r"\[\s*证据\s*(\d+)\s*\]",
            r"[证据 \1]",
            answer,
        )

        # 删除过多空行
        answer = re.sub(
            r"\n{3,}",
            "\n\n",
            answer,
        )

        cited_ids = {
            int(value)
            for value in re.findall(
                r"\[证据\s+(\d+)\]",
                answer,
            )
        }

        # 如果模型没有生成引用，不强行把引用插入医学陈述中，
        # 而是在末尾追加证据来源列表。
        if sources:
            source_lines = []

            for source in sources:
                evidence_id = int(
                    source["evidence_id"]
                )

                if (
                    cited_ids
                    and evidence_id
                    not in cited_ids
                ):
                    continue

                description = (
                    source.get("source")
                    or source.get("doc_id")
                    or source.get("chunk_id")
                    or "未知来源"
                )

                details = []

                if source.get("journal"):
                    details.append(
                        str(source["journal"])
                    )

                if source.get(
                    "publication_year"
                ):
                    details.append(
                        str(
                            source[
                                "publication_year"
                            ]
                        )
                    )

                if source.get("pmid"):
                    details.append(
                        f"PMID: {source['pmid']}"
                    )

                suffix = (
                    "；" + "；".join(details)
                    if details
                    else ""
                )

                source_lines.append(
                    f"- [证据 {evidence_id}] "
                    f"{description}{suffix}"
                )

            if source_lines:
                answer += (
                    "\n\n## 参考证据\n\n"
                    + "\n".join(
                        source_lines
                    )
                )

        if (
            add_disclaimer
            and DISCLAIMER
            not in answer
        ):
            answer += (
                "\n\n> 医学免责声明："
                + DISCLAIMER
            )

        return answer.strip()

    # --------------------------------------------------------
    # 完整生成流程
    # --------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_docs,
        evaluate_evidence: bool = True,
        critical_review: bool = True,
        add_disclaimer: bool = True,
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

        total_start = time.perf_counter()

        stage_times = {}
        token_counts = {}
        stage_success = {}

        evidence_evaluation = None
        draft_answer = None
        review_feedback = None
        final_answer_raw = None

        self.logger.info(
            "医学生成流程开始 | query=%s",
            query,
        )

        # ====================================================
        # 1. 上下文组装
        # ====================================================

        context_start = time.perf_counter()

        context_result = (
            self.context_assembler
            .assemble_context(
                retrieved_docs
            )
        )

        stage_times[
            "context_assembly"
        ] = round(
            time.perf_counter()
            - context_start,
            4,
        )

        stage_success[
            "context_assembly"
        ] = bool(
            context_result[
                "context_text"
            ]
        )

        token_counts[
            "context_assembly"
        ] = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": int(
                context_result[
                    "metadata"
                ].get(
                    "estimated_tokens",
                    0,
                )
            ),
        }

        if not context_result[
            "context_text"
        ]:
            raise ValueError(
                "上下文组装结果为空，"
                "无法执行答案生成。"
            )

        active_context = {
            "context_text": (
                context_result[
                    "context_text"
                ]
            ),
            "selected_chunks": (
                context_result[
                    "selected_chunks"
                ]
            ),
            "selected_evidence_ids": list(
                range(
                    1,
                    len(
                        context_result[
                            "selected_chunks"
                        ]
                    ) + 1,
                )
            ),
            "filter_applied": False,
        }

        # ====================================================
        # 2. 证据评估
        # ====================================================

        if evaluate_evidence:
            evaluation_response = (
                self._run_stage(
                    stage_key=(
                        "evidence_evaluator"
                    ),
                    stage_times=stage_times,
                    token_counts=token_counts,
                    stage_success=stage_success,
                    require_json=True,
                    json_schema=(
                        EVIDENCE_SCHEMA
                    ),
                    question=query,
                    context=(
                        context_result[
                            "context_text"
                        ]
                    ),
                )
            )

            if evaluation_response:
                parsed_evaluation = (
                    evaluation_response.get(
                        "parsed_json"
                    )
                )

                if isinstance(
                    parsed_evaluation,
                    Mapping,
                ):
                    evidence_evaluation = dict(
                        parsed_evaluation
                    )

                    active_context = (
                        self
                        ._filter_context_by_evaluation(
                            context_result,
                            evidence_evaluation,
                        )
                    )

                else:
                    stage_success[
                        "evidence_evaluator"
                    ] = False

                    evidence_evaluation = {
                        "raw_text": (
                            evaluation_response[
                                "text"
                            ]
                        ),
                        "parse_warning": (
                            "评估结果不是有效JSON，"
                            "因此未筛选上下文。"
                        ),
                    }

        else:
            stage_success[
                "evidence_evaluator"
            ] = False

            stage_times[
                "evidence_evaluator"
            ] = 0.0

            token_counts[
                "evidence_evaluator"
            ] = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        evidence_text = (
            json.dumps(
                evidence_evaluation,
                ensure_ascii=False,
                indent=2,
            )
            if evidence_evaluation
            is not None
            else "未执行独立证据评估。"
        )

        # ====================================================
        # 3. 生成答案草稿
        # ====================================================

        draft_response = self._run_stage(
            stage_key="answer_generator",
            stage_times=stage_times,
            token_counts=token_counts,
            stage_success=stage_success,
            question=query,
            context=active_context[
                "context_text"
            ],
            evidence_evaluation=(
                evidence_text
            ),
        )

        if not draft_response:
            raise OllamaGenerationError(
                "答案草稿生成失败。"
            )

        draft_answer = draft_response[
            "text"
        ]

        # ====================================================
        # 4. 批判性审查
        # ====================================================

        if critical_review:
            review_response = (
                self._run_stage(
                    stage_key=(
                        "critical_reviewer"
                    ),
                    stage_times=stage_times,
                    token_counts=token_counts,
                    stage_success=stage_success,
                    question=query,
                    context=active_context[
                        "context_text"
                    ],
                    evidence_evaluation=(
                        evidence_text
                    ),
                    draft_answer=(
                        draft_answer
                    ),
                )
            )

            if review_response:
                review_feedback = (
                    review_response[
                        "text"
                    ]
                )

        else:
            stage_success[
                "critical_reviewer"
            ] = False

            stage_times[
                "critical_reviewer"
            ] = 0.0

            token_counts[
                "critical_reviewer"
            ] = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        # ====================================================
        # 5. 最终答案
        # ====================================================

        if (
            critical_review
            and review_feedback
        ):
            final_response = (
                self._run_stage(
                    stage_key=(
                        "final_assembler"
                    ),
                    stage_times=stage_times,
                    token_counts=token_counts,
                    stage_success=stage_success,
                    question=query,
                    context=active_context[
                        "context_text"
                    ],
                    evidence_evaluation=(
                        evidence_text
                    ),
                    draft_answer=(
                        draft_answer
                    ),
                    review=review_feedback,
                )
            )

            if final_response:
                final_answer_raw = (
                    final_response["text"]
                )

            else:
                final_answer_raw = (
                    draft_answer
                )

        else:
            # 未审查或审查失败时直接使用草稿
            final_answer_raw = (
                draft_answer
            )

            stage_success[
                "final_assembler"
            ] = False

            stage_times[
                "final_assembler"
            ] = 0.0

            token_counts[
                "final_assembler"
            ] = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        # ====================================================
        # 6. 生成后处理
        # ====================================================

        postprocess_start = (
            time.perf_counter()
        )

        sources = self._format_sources(
            context_result[
                "selected_chunks"
            ]
        )

        final_answer = (
            self._postprocess_answer(
                answer=final_answer_raw,
                sources=sources,
                add_disclaimer=(
                    add_disclaimer
                ),
            )
        )

        stage_times[
            "postprocessing"
        ] = round(
            time.perf_counter()
            - postprocess_start,
            4,
        )

        stage_success[
            "postprocessing"
        ] = bool(
            final_answer
        )

        token_counts[
            "postprocessing"
        ] = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # ====================================================
        # 7. 组装最终结果
        # ====================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        total_prompt_tokens = sum(
            item.get(
                "prompt_tokens",
                0,
            )
            for item in token_counts.values()
        )

        total_output_tokens = sum(
            item.get(
                "output_tokens",
                0,
            )
            for item in token_counts.values()
        )

        result = {
            "query": query,
            "answer": final_answer,
            "context_metadata": (
                context_result[
                    "metadata"
                ]
            ),
            "generation_metrics": {
                "total_time_seconds": round(
                    total_time,
                    4,
                ),
                "stage_times": stage_times,
                "token_counts": token_counts,
                "total_prompt_tokens": (
                    total_prompt_tokens
                ),
                "total_output_tokens": (
                    total_output_tokens
                ),
                "answer_characters": len(
                    final_answer
                ),
                "stage_success": (
                    stage_success
                ),
            },
            "intermediate_results": {
                "evidence_evaluation": (
                    evidence_evaluation
                ),
                "draft_answer": (
                    draft_answer
                ),
                "review_feedback": (
                    review_feedback
                ),
                "active_evidence_ids": (
                    active_context[
                        "selected_evidence_ids"
                    ]
                ),
                "context_filter_applied": (
                    active_context[
                        "filter_applied"
                    ]
                ),
            },
            "sources": sources,
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        self.logger.info(
            "医学生成流程完成 | total=%.3fs "
            "| answer_chars=%d "
            "| prompt_tokens=%d "
            "| output_tokens=%d "
            "| stages=%s",
            total_time,
            len(final_answer),
            total_prompt_tokens,
            total_output_tokens,
            {
                key: value
                for key, value
                in stage_success.items()
            },
        )

        return result