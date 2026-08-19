import pandas as pd
import torch
from tqdm.auto import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class MedicalBGECrossEncoderReranker:
    """
    对MultiPathRetriever产生的融合候选进行重排序。

    最终评分由三部分组成：

        relevance_score:
            BGE Cross-Encoder查询—文档相关性

        recency_score:
            根据发表年份计算时效性

        authority_score:
            根据配置的期刊等级计算权威性

    默认权重：
        relevance = 0.60
        recency   = 0.25
        authority = 0.15
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"

    @classmethod
    def load_tokenizer(
        cls,
        model_name=None,
        local_files_only=True,
    ):
        """
        单独加载重排模型对应的 tokenizer。

        适用于只需要统计 token 数、
        不需要加载完整 Cross-Encoder 模型的场景。
        """
        model_name = (
            model_name
            or cls.DEFAULT_MODEL_NAME
        )

        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

    def __init__(
        self,
        model_name=None,
        device=None,
        batch_size=2,
        max_length=512,
        criteria_weights=None,
        authority_scores=None,
        default_authority_score=0.50,
        recency_window=10,
        missing_year_score=0.50,
        include_title=True,
    ):
        self.model_name = model_name

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.recency_window = recency_window
        self.missing_year_score = (
            missing_year_score
        )
        self.include_title = include_title

        self.criteria_weights = (
            criteria_weights
            or {
                "relevance": 0.60,
                "recency": 0.25,
                "authority": 0.15,
            }
        )

        required_weights = {
            "relevance",
            "recency",
            "authority",
        }

        missing_weights = (
            required_weights
            - set(self.criteria_weights)
        )

        if missing_weights:
            raise ValueError(
                f"缺少评分权重："
                f"{sorted(missing_weights)}"
            )

        weight_sum = sum(
            self.criteria_weights.values()
        )

        if not np.isclose(
            weight_sum,
            1.0,
        ):
            raise ValueError(
                "criteria_weights总和必须为1。"
            )

        # 这是项目级业务规则，不是正式影响因子排名。
        self.authority_scores = (
            authority_scores
            or {
                "nature medicine": 1.00,
                "the new england journal of medicine": 1.00,
                "new england journal of medicine": 1.00,
                "the lancet": 1.00,
                "lancet": 1.00,
                "jama": 0.95,
                "the bmj": 0.90,
                "bmj": 0.90,
                "nature communications": 0.85,
                "science advances": 0.85,
                "scientific reports": 0.70,
                "plos medicine": 0.80,
                "plos one": 0.65,
            }
        )

        self.default_authority_score = (
            default_authority_score
        )

        print(
            f"Loading reranker: {model_name}"
        )
        print(
            f"Reranker device: {self.device}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
        )

        model_kwargs = {}

        if self.device == "cuda":
            model_kwargs["torch_dtype"] = (
                torch.float16
            )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                local_files_only=True,
                use_safetensors=True,
                **model_kwargs,
            )
            .to(self.device)
        )

        self.model.eval()

    def encode(
        self,
        text,
        add_special_tokens=False,
        **kwargs,
    ):
        """
        代理底层 Hugging Face tokenizer.encode。
        使重排器实例可以直接作为
        ContextAssembler 的 tokenizer 使用。
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )


    @staticmethod
    def _clean_text(value):
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value).strip()

    def _prepare_passage(
        self,
        row: pd.Series,
    ) -> str:
        text = self._clean_text(
            row.get("text", "")
        )

        title = self._clean_text(
            row.get("source_title", "")
        )

        if (
            self.include_title
            and title
            and title.casefold()
            not in text[:1000].casefold()
        ):
            return (
                f"Title: {title}\n\n{text}"
            )

        return text

    def score_relevance(
        self,
        query_text: str,
        passages: list[str],
    ) -> np.ndarray:
        """
        使用Cross-Encoder计算查询—文档相关性。

        注意：
        这里不添加BGE embedding instruction。
        Cross-Encoder直接输入[query, passage]。
        """
        if not isinstance(
            query_text,
            str,
        ):
            raise TypeError(
                "query_text必须是字符串。"
            )

        query_text = query_text.strip()

        if not query_text:
            raise ValueError(
                "query_text不能为空。"
            )

        if not passages:
            return np.array(
                [],
                dtype=np.float32,
            )

        relevance_scores = []

        for start in tqdm(
            range(
                0,
                len(passages),
                self.batch_size,
            ),
            desc="BGE reranking",
        ):
            batch_passages = passages[
                start:
                start + self.batch_size
            ]

            pairs = [
                [query_text, passage]
                for passage
                in batch_passages
            ]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value
                in inputs.items()
            }

            with torch.inference_mode():
                logits = self.model(
                    **inputs,
                    return_dict=True,
                ).logits.float()

            # bge-reranker-base通常输出单个logit。
            if (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):
                batch_scores = (
                    torch.sigmoid(
                        logits[:, 0]
                    )
                )

            # 兼容可能输出两个类别的模型。
            elif (
                logits.ndim == 2
                and logits.shape[1] >= 2
            ):
                batch_scores = (
                    torch.softmax(
                        logits,
                        dim=1,
                    )[:, -1]
                )

            else:
                batch_scores = (
                    torch.sigmoid(
                        logits.reshape(-1)
                    )
                )

            relevance_scores.extend(
                batch_scores
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            del inputs
            del logits

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return np.asarray(
            relevance_scores,
            dtype=np.float32,
        )

    def score_recency(
        self,
        publication_years,
        reference_year=None,
    ) -> pd.Series:
        """
        线性时效性评分：

            当年发表      -> 1.0
            5年前发表     -> 0.5（window=10）
            10年前及更早  -> 0.0

        缺失年份使用missing_year_score。
        """
        years = pd.to_numeric(
            publication_years,
            errors="coerce",
        )

        if reference_year is None:
            reference_year = (
                pd.Timestamp.now().year
            )

        age = (
            reference_year - years
        ).clip(lower=0)

        scores = (
            1.0
            - age / self.recency_window
        ).clip(
            lower=0.0,
            upper=1.0,
        )

        return scores.fillna(
            self.missing_year_score
        )

    def score_authority(
        self,
        journals,
    ) -> pd.Series:
        """
        根据规范化期刊名称查找权威性分数。
        """
        def lookup(journal):
            normalized = (
                self._clean_text(journal)
                .casefold()
            )

            return self.authority_scores.get(
                normalized,
                self.default_authority_score,
            )

        return journals.apply(lookup)

    def rerank(
        self,
        query_text: str,
        candidates: pd.DataFrame,
        top_k=10,
        reference_year=None,
    ) -> pd.DataFrame:
        if not isinstance(
            candidates,
            pd.DataFrame,
        ):
            raise TypeError(
                "candidates必须是DataFrame。"
            )

        if candidates.empty:
            return candidates.copy()

        if "text" not in candidates.columns:
            raise ValueError(
                "候选结果缺少text字段。"
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k必须是大于0的整数。"
            )

        df = candidates.copy().reset_index(
            drop=True
        )

        passages = [
            self._prepare_passage(row)
            for _, row in df.iterrows()
        ]

        # 第一项：查询—文档相关性
        df["relevance_score"] = (
            self.score_relevance(
                query_text=query_text,
                passages=passages,
            )
        )

        # 第二项：时效性
        if "publication_year" in df.columns:
            publication_years = (
                df["publication_year"]
            )
        else:
            publication_years = pd.Series(
                np.nan,
                index=df.index,
            )

        df["recency_score"] = (
            self.score_recency(
                publication_years,
                reference_year=reference_year,
            )
        )

        # 第三项：期刊权威性
        if "journal" in df.columns:
            journals = df["journal"]
        else:
            journals = pd.Series(
                "",
                index=df.index,
            )

        df["authority_score"] = (
            self.score_authority(
                journals
            )
        )

        weights = self.criteria_weights

        df["final_score"] = (
            weights["relevance"]
            * df["relevance_score"]
            + weights["recency"]
            * df["recency_score"]
            + weights["authority"]
            * df["authority_score"]
        )

        df = df.sort_values(
            "final_score",
            ascending=False,
        ).reset_index(drop=True)

        df["final_rank"] = np.arange(
            1,
            len(df) + 1,
        )

        return df.head(top_k).copy()