import re
from lxml import etree

#医学词典与实体规则构建

MEDICAL_SYNONYMS = {
    # 疾病与临床事件
    "mi": [
        "myocardial infarction",
        "heart attack"
    ],
    "myocardial infarction": [
        "MI",
        "heart attack"
    ],
    "心肌梗死": [
        "myocardial infarction",
        "MI",
        "heart attack"
    ],
    "心脏病发作": [
        "heart attack",
        "myocardial infarction",
        "MI"
    ],
    "cvd": [
        "cardiovascular disease",
        "cardiovascular diseases"
    ],
    "cardiovascular disease": [
        "CVD",
        "cardiovascular diseases"
    ],
    "心血管疾病": [
        "cardiovascular disease",
        "CVD"
    ],
    "aki": [
        "acute kidney injury",
        "acute renal injury"
    ],
    "急性肾损伤": [
        "acute kidney injury",
        "AKI"
    ],
    "copd": [
        "chronic obstructive pulmonary disease"
    ],
    "ards": [
        "acute respiratory distress syndrome"
    ],

    # 药物
    "metformin": [
        "dimethylbiguanide",
        "Glucophage"
    ],
    "二甲双胍": [
        "metformin",
        "dimethylbiguanide"
    ],
    "aspirin": [
        "acetylsalicylic acid",
        "ASA"
    ],
    "阿司匹林": [
        "aspirin",
        "acetylsalicylic acid",
        "ASA"
    ],
    "atorvastatin": [
        "Lipitor",
        "atorvastatin calcium"
    ],
    "阿托伐他汀": [
        "atorvastatin",
        "atorvastatin calcium"
    ],

    # 指标与治疗
    "egfr": [
        "estimated glomerular filtration rate"
    ],
    "pci": [
        "percutaneous coronary intervention"
    ],
}


MEDICAL_PATTERNS = {
    "drug": re.compile(
        r"\b("
        r"aspirin|metformin|atorvastatin|warfarin|insulin|"
        r"acetylsalicylic acid|dimethylbiguanide"
        r")\b|"
        r"(阿司匹林|二甲双胍|阿托伐他汀|华法林|胰岛素)",
        flags=re.IGNORECASE
    ),

    "disease": re.compile(
        r"\b("
        r"myocardial infarction|heart attack|"
        r"cardiovascular diseases?|"
        r"acute kidney injury|"
        r"chronic obstructive pulmonary disease|"
        r"acute respiratory distress syndrome|"
        r"diabetes(?: mellitus)?"
        r")\b|"
        r"(心肌梗死|心脏病发作|心血管疾病|"
        r"急性肾损伤|慢性阻塞性肺疾病|"
        r"急性呼吸窘迫综合征|糖尿病)",
        flags=re.IGNORECASE
    ),

    "abbreviation": re.compile(
        r"\b(MI|CVD|AKI|COPD|ARDS|EGFR|PCI|ICU)\b",
        flags=re.IGNORECASE
    ),

    "outcome": re.compile(
        r"\b("
        r"mortality|survival|hospitalization|"
        r"cardiovascular outcome|adverse event|"
        r"risk reduction|treatment effect"
        r")\b|"
        r"(死亡率|生存率|住院|心血管结局|"
        r"不良事件|风险降低|治疗效果)",
        flags=re.IGNORECASE
    )
}


#英文语义查询改进对照表

ZH_TO_EN_TERMS = {
    "二甲双胍": "metformin",
    "心血管疾病": "cardiovascular disease",
    "心肌梗死": "myocardial infarction",
    "阿司匹林": "aspirin",
    "急性肾损伤": "acute kidney injury",
    "死亡率": "mortality",
    "治疗效果": "treatment effect",
    "风险": "risk",
    "近5年": "in the last five years",
    "有何影响": "what are the effects"
}

#LLM生成的医学回答免责声明
DISCLAIMER = (
        "以上内容仅用于医学信息参考，不能替代医生的诊断、"
        "处方或个体化治疗建议。如有持续不适、病情变化，"
        "或需要调整药物，请及时咨询合格医务人员。"
    )


# Parse PMC XML Files
def parse_pmc(xml_file):

    tree = etree.parse(xml_file)

    title = tree.xpath(
        "string(//article-title)"
    )

    abstract = tree.xpath(
        "string(//abstract)"
    )

    journal = tree.xpath(
        "string(//journal-title)"
    )

    pmid = tree.xpath(
        "string(//article-id[@pub-id-type='pmid'])"
    )

    pub_date = tree.xpath(
        "string(//pub-date/year)"
    )

    return {
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "pmid": pmid,
        "pub_date": pub_date
    }
