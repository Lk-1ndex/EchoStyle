import math
import re
from typing import Dict, List, Tuple
from src.core.models import StatisticalMetrics


class StylometricsAnalyzer:
    """
    纯数学统计语言学分析器：
    包含平均句长、离散度、标点熵、转折词密度以及词汇丰富度 Type-Token Ratio (TTR)。
    """

    TRANSITION_KEYWORDS = [
        "但是", "然而", "不过", "反而", "况且", "甚至",
        "说白了", "实际上", "坦率地讲", "换句话说", "退一步讲",
        "因此", "所以", "总的来看", "归根结底", "反过来", "恰恰", "其实"
    ]

    TARGET_PUNCTUATIONS = ["，", "。", "！", "？", "；", "：", "——", "……", "“", "”", "（", "）", "、"]

    @classmethod
    def analyze(cls, text: str) -> StatisticalMetrics:
        if not text or not text.strip():
            return StatisticalMetrics()

        # 1. 段落分析
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        total_paragraphs = max(1, len(paragraphs))

        # 2. 句子切分（支持中英文句末标点）
        raw_sentences = re.split(r"[。！？!?…\n]+", text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 1]
        total_sentences = len(sentences)

        if total_sentences == 0:
            return StatisticalMetrics(total_chars=len(text), total_paragraphs=total_paragraphs)

        # 3. 句长统计
        sentence_lengths = [len(s) for s in sentences]
        avg_len = sum(sentence_lengths) / total_sentences
        variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / total_sentences
        std_dev = math.sqrt(variance)

        # 4. 词汇丰富度计算 (Type-Token Ratio, TTR)
        clean_words = re.findall(r"[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+", text)
        total_tokens = max(1, len(clean_words))
        unique_types = len(set(clean_words))
        ttr = round(unique_types / total_tokens, 3)

        # 5. 标点符号分布与信息熵 (Shannon Entropy)
        punc_counts = {}
        total_puncs = 0
        for punc in cls.TARGET_PUNCTUATIONS:
            cnt = text.count(punc)
            if cnt > 0:
                punc_counts[punc] = cnt
                total_puncs += cnt

        punc_dist = {}
        entropy = 0.0
        if total_puncs > 0:
            for punc, cnt in punc_counts.items():
                prob = cnt / total_puncs
                punc_dist[punc] = round(prob, 4)
                if prob > 0:
                    entropy -= prob * math.log2(prob)

        # 6. 转折与标志性连接词密度
        transition_counts = {}
        total_transitions = 0
        for kw in cls.TRANSITION_KEYWORDS:
            cnt = len(re.findall(re.escape(kw), text))
            if cnt > 0:
                transition_counts[kw] = cnt
                total_transitions += cnt

        total_chars = len(text.replace(" ", "").replace("\n", ""))
        trans_density = round((total_transitions / max(1, total_chars)) * 1000, 2)

        # 7. 定性节奏描述
        if avg_len <= 15 and std_dev < 10:
            rhythm = "高频轻快极短句，分段紧凑，具备极强的网感与自媒体碎片化阅读质感。"
        elif avg_len <= 22 and std_dev >= 12:
            rhythm = "长短句交错跌宕，节奏起伏大，既有短句爆破，又有长句从容铺陈。"
        elif avg_len > 28:
            rhythm = "多复合长句，思辨层级丰富，偏学术、深度研报或沉稳散文风格。"
        else:
            rhythm = "匀速推进型文风，句长适中，行文平稳从容。"

        return StatisticalMetrics(
            total_chars=total_chars,
            total_sentences=total_sentences,
            total_paragraphs=total_paragraphs,
            avg_sentence_length=round(avg_len, 2),
            sentence_length_std=round(std_dev, 2),
            avg_paragraph_length=round(total_sentences / total_paragraphs, 2),
            ttr=ttr,
            unique_words_count=unique_types,
            punctuation_entropy=round(entropy, 3),
            punctuation_distribution=punc_dist,
            transition_density=trans_density,
            transition_words=transition_counts,
            rhythm_pattern=rhythm,
        )
