import re
from enum import Enum
from typing import (
    List,
    Sequence,
    Tuple
)

import editdistance
import numpy as np
from numpy.typing import NDArray

from normalizer import SUPPORTED_PUNCTUATION_SET


class Metrics(Enum):
    WER = "WER"
    PER = "PER"
    CER = "CER"


class Metric:
    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Metrics):
        if x is Metrics.WER:
            return WordErrorRate()
        elif x is Metrics.PER:
            return PunctuationErrorRate()
        elif x is Metrics.CER:
            return CharacterErrorRate()
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{x}`")


class WordErrorRate(Metric):
    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        error_count = editdistance.eval(ref_tokens, pred_tokens)
        token_count = len(ref_tokens)

        return error_count, token_count


class CharacterErrorRate(Metric):
    """
    Character Error Rate for Chinese text with mixed English.

    Uses tokenization rules:
    - Chinese characters: 1 char = 1 token
    - English words: 1 word = 1 token
    - Numbers: 1 number sequence = 1 token
    - Punctuation: Ignored (should be removed before tokenization)
    """

    # Unicode ranges for CJK characters
    CJK_RANGES = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x20000, 0x2A6DF), # CJK Extension B
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    ]

    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """Check if a character is a Chinese character."""
        code_point = ord(char)
        return any(
            start <= code_point <= end
            for start, end in CharacterErrorRate.CJK_RANGES
        )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text according to mixed Chinese/English rules.
        """
        tokens = []
        current_token = ""
        current_type = None  # 'english', 'number'

        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                    current_type = None
                continue

            if CharacterErrorRate.is_chinese_char(char):
                # Chinese characters are individual tokens
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                    current_type = None
                tokens.append(char)
            elif char.isdigit():
                if current_type == 'number':
                    current_token += char
                else:
                    if current_token:
                        tokens.append(current_token)
                    current_token = char
                    current_type = 'number'
            elif char.isalpha():  # English letter
                if current_type == 'english':
                    current_token += char
                else:
                    if current_token:
                        tokens.append(current_token)
                    current_token = char
                    current_type = 'english'
            # Ignore other characters

        if current_token:
            tokens.append(current_token)

        return tokens

    def calculate(self, prediction: str, reference: str) -> Tuple[int, int]:
        """
        Calculate CER using editdistance on tokenized sequences.
        """
        pred_tokens = self.tokenize(prediction)
        ref_tokens = self.tokenize(reference)

        error_count = editdistance.eval(ref_tokens, pred_tokens)
        token_count = len(ref_tokens)

        return error_count, token_count


class PunctuationErrorRate(Metric):
    """Reference: https://arxiv.org/abs/2310.02943"""

    @staticmethod
    def _get_punctuation_indices(tokens: Sequence[str], punctuation: str) -> List[int]:
        return [i for i, t in enumerate(tokens) if t in punctuation]

    @staticmethod
    def _compute_dp_matrix(reference: Sequence[str], prediction: Sequence[str]) -> NDArray:
        m, n = len(reference), len(prediction)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if reference[i - 1] == prediction[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

        return dp

    @staticmethod
    def _backtrack(
        dp: NDArray,
        reference: Sequence[str],
        prediction: Sequence[str],
        reference_punct_indices: Sequence[int],
        prediction_punct_indices: Sequence[int],
    ) -> Tuple[int, int, int, int]:
        i, j = len(reference), len(prediction)

        reference_punct_set = set(reference_punct_indices)
        prediction_punct_set = set(prediction_punct_indices)

        num_match = 0
        num_sub = 0

        alignment = {}

        while i > 0 and j > 0:
            if reference[i - 1] == prediction[j - 1]:
                alignment[i - 1] = j - 1
                i -= 1
                j -= 1
            elif dp[i - 1, j - 1] <= dp[i - 1, j] and dp[i - 1, j - 1] <= dp[i, j - 1]:
                alignment[i - 1] = j - 1
                i -= 1
                j -= 1
            elif dp[i - 1, j] <= dp[i, j - 1]:
                i -= 1
            else:
                j -= 1

        for idx in reference_punct_set:
            if idx in alignment:
                pred_idx = alignment[idx]
                if pred_idx in prediction_punct_set:
                    if reference[idx] == prediction[pred_idx]:
                        num_match += 1
                    else:
                        num_sub += 1

        num_delete = len(reference_punct_indices) - (num_sub + num_match)
        num_insert = len(prediction_punct_indices) - (num_sub + num_match)

        return num_insert, num_delete, num_sub, num_match

    def calculate(
        self, prediction: str, reference: str, punctuation: str = SUPPORTED_PUNCTUATION_SET
    ) -> Tuple[int, int]:
        pred_tokens = [token for token in re.findall(rf"[{punctuation}]|[^{punctuation}\s]+", prediction)]
        ref_tokens = [token for token in re.findall(rf"[{punctuation}]|[^{punctuation}\s]+", reference)]

        pred_punct_indices = self._get_punctuation_indices(pred_tokens, punctuation)
        ref_punct_indices = self._get_punctuation_indices(ref_tokens, punctuation)

        dp = self._compute_dp_matrix(reference=ref_tokens, prediction=pred_tokens)

        num_insert, num_delete, num_sub, num_correct = self._backtrack(
            dp=dp,
            reference=ref_tokens,
            prediction=pred_tokens,
            reference_punct_indices=ref_punct_indices,
            prediction_punct_indices=pred_punct_indices,
        )

        error_count = num_insert + num_delete + num_sub
        denom = num_insert + num_delete + num_sub + num_correct

        return error_count, denom


__all__ = [
    "Metric",
    "Metrics",
]
