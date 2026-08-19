from dataclasses import dataclass

'''
医学四阶段提示词

- 证据评估器：明确证据能支持与不能支持什么；
- 答案生成器：根据证据评估生成草稿；
- 批判性审查器：检查幻觉、引用错配、因果夸大和危险建议；
- 最终组装器：吸收审查意见生成最终回答。
'''

@dataclass(frozen=True)
class PromptStage:
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int


MEDICAL_PROMPT_STAGES = {
    "evidence_evaluator": PromptStage(
        "证据评估器",
        "你是循证医学证据评估专家。只能依据给定上下文，不得补造论文、数据或结论。"
        "区分研究设计、人群、干预/暴露、对照、结局和不确定性；识别冲突、偏倚与适用性。"
        "信息缺失时明确写‘证据不足’，不直接向患者下诊断。",
        "用户问题：\n{question}\n\n检索上下文：\n{context}\n\n"
        "请输出：1.相关证据及编号；2.研究类型、人群和核心结果；"
        "3.证据质量及理由；4.冲突和局限；5.可支持与不能支持的结论。"
        "事实结论必须标注[证据 N]。",
        0.1, 1800,
    ),
    "answer_generator": PromptStage(
        "答案生成器",
        "你是谨慎的医学问答助手。仅依据上下文和证据评估作答，不把相关性写成因果，"
        "不超出证据适用人群。涉及诊断或用药调整时提示咨询医务人员。",
        "用户问题：\n{question}\n\n检索上下文：\n{context}\n\n"
        "证据评估：\n{evidence_evaluation}\n\n先给简明结论，再说明关键证据、局限和实际含义。"
        "使用[证据 N]，无证据支持的内容不要写。",
        0.25, 1600,
    ),
    "critical_reviewer": PromptStage(
        "批判性审查器",
        "你是医学事实与安全审查专家。检查无依据陈述、引用错配、因果夸大、"
        "冲突证据遗漏、重要限定遗漏和危险建议。不得用外部知识替换检索证据。",
        "用户问题：\n{question}\n\n检索上下文：\n{context}\n\n证据评估：\n{evidence_evaluation}\n\n"
        "回答草稿：\n{draft_answer}\n\n输出问题分级、证据对应、修改建议和是否可进入最终组装。",
        0.1, 1200,
    ),
    "final_assembler": PromptStage(
        "最终组装器",
        "你负责最终医学回答。吸收审查意见并保持证据边界；不暴露内部推理，不声称完成诊断。"
        "只保留上下文支持的引用，结论强度与证据质量一致。",
        "用户问题：\n{question}\n\n检索上下文：\n{context}\n\n证据评估：\n{evidence_evaluation}\n\n"
        "回答草稿：\n{draft_answer}\n\n审查意见：\n{review}\n\n"
        "输出结论摘要、关键证据、不确定性与局限、必要安全提示。禁止新增事实。",
        0.15, 1600,
    ),
}

EVIDENCE_SCHEMA = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
            },
            "relevant_evidence_ids": {
                "type": "array",
                "items": {
                    "type": "integer",
                },
            },
            "evidence_quality": {
                "type": "string",
            },
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "supported_conclusions": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "unsupported_conclusions": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": [
            "summary",
            "relevant_evidence_ids",
            "evidence_quality",
            "conflicts",
            "limitations",
            "supported_conclusions",
            "unsupported_conclusions",
        ],
    }

# 提示词模版渲染接口
def render_prompt(stage_key: str, **values):
    if stage_key not in MEDICAL_PROMPT_STAGES:
        raise KeyError(f"未知提示阶段：{stage_key}")
    stage = MEDICAL_PROMPT_STAGES[stage_key]
    try:
        user_prompt = stage.user_prompt_template.format(**values)
    except KeyError as exc:
        raise ValueError(f"{stage_key}缺少模板变量：{exc.args[0]}") from exc
    return {
        "messages": [
            {"role": "system", "content": stage.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": stage.temperature,
        "max_tokens": stage.max_tokens,
    }
