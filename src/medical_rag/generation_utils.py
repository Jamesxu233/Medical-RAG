import logging
from typing import Any, Dict

import pandas as pd

from medical_rag.generation_pipeline import MedicalGenerationPipeline

logger = logging.getLogger(
    "medical_generation"
)

try:
    from IPython.display import display
except ImportError:
    display = print


# 结果显示与测试函数

def display_generation_result(
    result: Dict[str, Any],
) -> None:
    """
    在 Notebook 中显示最终回答、耗时、token统计和来源。
    """

    print("=" * 80)
    print("查询")
    print("=" * 80)
    print(result["query"])

    print("\n" + "=" * 80)
    print("最终回答")
    print("=" * 80)
    print(result["answer"])

    metrics = result[
        "generation_metrics"
    ]

    print("\n" + "=" * 80)
    print("生成指标")
    print("=" * 80)

    print(
        "总耗时：",
        metrics["total_time_seconds"],
        "秒",
    )

    print(
        "答案长度：",
        metrics["answer_characters"],
        "字符",
    )

    print(
        "输入Token：",
        metrics["total_prompt_tokens"],
    )

    print(
        "输出Token：",
        metrics["total_output_tokens"],
    )

    stage_rows = []

    for stage_name, elapsed in (
        metrics["stage_times"].items()
    ):
        tokens = (
            metrics["token_counts"]
            .get(stage_name, {})
        )

        stage_rows.append({
            "stage": stage_name,
            "success": (
                metrics["stage_success"]
                .get(stage_name, False)
            ),
            "elapsed_seconds": elapsed,
            "prompt_tokens": (
                tokens.get(
                    "prompt_tokens",
                    0,
                )
            ),
            "output_tokens": (
                tokens.get(
                    "output_tokens",
                    0,
                )
            ),
            "total_tokens": (
                tokens.get(
                    "total_tokens",
                    0,
                )
            ),
        })

    display(
        pd.DataFrame(stage_rows)
    )

    print("\n" + "=" * 80)
    print("引用来源")
    print("=" * 80)

    source_df = pd.DataFrame(
        result["sources"]
    )

    if source_df.empty:
        print("无引用来源。")
    else:
        display(source_df)


def run_medical_generation_test(
    pipeline: MedicalGenerationPipeline,
    query: str,
    retrieved_docs,
    evaluate_evidence: bool = True,
    critical_review: bool = True,
) -> Dict[str, Any]:
    """
    执行一次完整测试，并记录关键指标。
    """

    logger.info(
        "测试开始 | query=%s",
        query,
    )

    result = pipeline.generate(
        query=query,
        retrieved_docs=retrieved_docs,
        evaluate_evidence=(
            evaluate_evidence
        ),
        critical_review=(
            critical_review
        ),
        add_disclaimer=True,
    )

    display_generation_result(
        result
    )

    metrics = result[
        "generation_metrics"
    ]

    logger.info(
        "测试完成 | query=%s "
        "| elapsed=%.3fs "
        "| answer_chars=%d "
        "| stage_success=%s",
        query,
        metrics[
            "total_time_seconds"
        ],
        metrics[
            "answer_characters"
        ],
        metrics[
            "stage_success"
        ],
    )

    return result