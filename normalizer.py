import re
import string
import unicodedata

import inflect

from languages import Languages

SUPPORTED_PUNCTUATION_SET = ",.?"


class Normalizer(object):
    def __init__(self, keep_punctuation: bool, punctuation_set: str = SUPPORTED_PUNCTUATION_SET) -> None:
        self._keep_punctuation = keep_punctuation
        self._punctuation_set = punctuation_set

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool) -> str:
        raise NotImplementedError()

    @classmethod
    def create(
        cls,
        language: Languages,
        keep_punctuation: bool,
        punctuation_set: str = SUPPORTED_PUNCTUATION_SET,
    ):
        if language == Languages.EN:
            return EnglishNormalizer(keep_punctuation, punctuation_set)
        elif language in [
            Languages.DE,
            Languages.ES,
            Languages.FR,
            Languages.IT,
            Languages.PT_PT,
            Languages.PT_BR,
        ]:
            return DefaultNormalizer(keep_punctuation, punctuation_set)
        elif language == Languages.ZH:
            return ChineseNormalizer(keep_punctuation, punctuation_set)
        else:
            raise ValueError(
                f"Cannot create {cls.__name__} of type `{language}`")


class DefaultNormalizer(Normalizer):
    """
    Adapted from: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
    """

    ADDITIONAL_DIACRITICS = {
        "œ": "oe",
        "Œ": "OE",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "ẞ": "SS",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "L",
    }

    def _remove_symbols_and_diacritics(self, s: str) -> str:
        return "".join(
            (
                DefaultNormalizer.ADDITIONAL_DIACRITICS[c]
                if c in DefaultNormalizer.ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else (
                        " "
                        if unicodedata.category(c)[0] in "MS"
                        or (unicodedata.category(c)[0] == "P" and c not in SUPPORTED_PUNCTUATION_SET)
                        else c
                    )
                )
            )
            for c in unicodedata.normalize("NFKD", s)
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r"[<\[][^>\]]*[>\]]", "", sentence)
        sentence = re.sub(r"\(([^)]+?)\)", "", sentence)
        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")
        sentence = self._remove_symbols_and_diacritics(sentence).lower()

        if self._keep_punctuation:
            removable_punctuation = "".join(
                set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = re.sub(r"\s+", " ", sentence)
        # Keep only English alphabet characters, numbers, and spaces
        sentence = re.sub(
            r'[^a-zA-Z0-9\s]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        return sentence


class ChineseNormalizer(Normalizer):
    """
    Normalizer for Chinese (Mandarin) text that handles code-mixed Chinese/English.

    Preserves Chinese characters while removing punctuation and normalizing text.
    Tokenization rules (for CER calculation):
    - Chinese characters: Each character is one token
    - English words: Each word is one token
    - Numbers: Each number sequence is one token
    - Punctuation: Removed/ignored
    """

    # Chinese punctuation to remove
    CHINESE_PUNCTUATION = "。，！？、；：""''【】（）《》「」『』—…·～"

    # Unicode ranges for CJK characters
    CJK_RANGES = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    ]

    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """Check if a character is a Chinese character."""
        code_point = ord(char)
        return any(
            start <= code_point <= end
            for start, end in ChineseNormalizer.CJK_RANGES
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        """
        Normalize Chinese/English mixed text.
        - Convert to lowercase (for English portions)
        - Remove all punctuation (Chinese and English)
        - Collapse whitespace
        """
        sentence = sentence.lower()

        # Remove bracketed content like <unk>
        sentence = re.sub(r"[<\[][^>\]]*[>\]]", "", sentence)

        # Remove English punctuation and special characters
        english_punct = '\'\""`():;![]{}#$%^&*+=|\\~'
        for c in english_punct:
            sentence = sentence.replace(c, "")

        # Normalize hyphens and dashes to spaces (for words like "Bye-bye")
        for c in "-/–—":
            sentence = sentence.replace(c, " ")

        # Remove Chinese punctuation
        for c in self.CHINESE_PUNCTUATION:
            sentence = sentence.replace(c, "")

        # Handle punctuation based on keep_punctuation setting
        if self._keep_punctuation:
            removable_punctuation = "".join(
                set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        # Collapse whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        return sentence

    @staticmethod
    def tokenize(sentence: str) -> list:
        """
        Tokenize normalized text for CER calculation.

        Rules:
        - Chinese characters: 1 char = 1 token
        - English words: 1 word = 1 token
        - Numbers: 1 number sequence = 1 token
        """
        tokens = []
        current_token = ""
        current_type = None  # 'english', 'number'

        for char in sentence:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                    current_type = None
                continue

            if ChineseNormalizer.is_chinese_char(char):
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


class EnglishNormalizer(Normalizer):
    AMERICAN_SPELLINGS = {
        "acknowledgement": "acknowledgment",
        "analogue": "analog",
        "armour": "armor",
        "ascendency": "ascendancy",
        "behaviour": "behavior",
        "behaviourist": "behaviorist",
        "cancelled": "canceled",
        "catalogue": "catalog",
        "centre": "center",
        "centres": "centers",
        "colour": "color",
        "coloured": "colored",
        "colourist": "colorist",
        "colourists": "colorists",
        "colours": "colors",
        "cosier": "cozier",
        "counselled": "counseled",
        "criticised": "criticized",
        "crystallise": "crystallize",
        "defence": "defense",
        "discoloured": "discolored",
        "dishonour": "dishonor",
        "dishonoured": "dishonored",
        "encyclopaedia": "Encyclopedia",
        "endeavour": "endeavor",
        "endeavouring": "endeavoring",
        "favour": "favor",
        "favourite": "favorite",
        "favours": "favors",
        "fibre": "fiber",
        "flamingoes": "flamingos",
        "fulfill": "fulfil",
        "grey": "gray",
        "harmonised": "harmonized",
        "honour": "honor",
        "honourable": "honorable",
        "honourably": "honorably",
        "honoured": "honored",
        "honours": "honors",
        "humour": "humor",
        "islamised": "islamized",
        "labour": "labor",
        "labourers": "laborers",
        "levelling": "leveling",
        "luis": "lewis",
        "lustre": "luster",
        "manoeuvring": "maneuvering",
        "marshall": "marshal",
        "marvellous": "marvelous",
        "merchandising": "merchandizing",
        "milicent": "millicent",
        "moustache": "mustache",
        "moustaches": "mustaches",
        "neighbour": "neighbor",
        "neighbourhood": "neighborhood",
        "neighbouring": "neighboring",
        "neighbours": "neighbors",
        "omelette": "omelet",
        "organisation": "organization",
        "organiser": "organizer",
        "practise": "practice",
        "pretence": "pretense",
        "programme": "program",
        "realise": "realize",
        "realised": "realized",
        "recognised": "recognized",
        "shrivelled": "shriveled",
        "signalling": "signaling",
        "skilfully": "skillfully",
        "smouldering": "smoldering",
        "specialised": "specialized",
        "sterilise": "sterilize",
        "sylvia": "silvia",
        "theatre": "theater",
        "theatres": "theaters",
        "travelled": "traveled",
        "travellers": "travelers",
        "travelling": "traveling",
        "vapours": "vapors",
        "wilful": "willful",
    }

    ABBREVIATIONS = {
        "junior": "jr",
        "senior": "sr",
        "okay": "ok",
        "doctor": "dr",
        "mister": "mr",
        "missus": "mrs",
        "saint": "st",
    }

    # Apostrophes that are not part of a contraction
    APOSTROPHE_REGEX = r"(?<!\w)\'|\'(?!\w)"

    @staticmethod
    def to_american(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.AMERICAN_SPELLINGS[x]
                 if x in EnglishNormalizer.AMERICAN_SPELLINGS else x)
                for x in sentence.split()
            ]
        )

    @staticmethod
    def normalize_abbreviations(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.ABBREVIATIONS[x]
                 if x in EnglishNormalizer.ABBREVIATIONS else x)
                for x in sentence.split()
            ]
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        p = inflect.engine()

        sentence = sentence.lower()

        for c in "-/–—":
            sentence = sentence.replace(c, " ")

        for c in '‘":;“”`()[]':
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")

        if self._keep_punctuation:
            removable_punctuation = "".join(
                set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("’", "'").replace("&", "and")

        sentence = re.sub(self.APOSTROPHE_REGEX, "", sentence)

        def num2txt(y):
            if any(x.isdigit() for x in y):
                ends_with_period = y[-1] == '.' and self._keep_punctuation
                if ends_with_period:
                    y = y[:-1]
                y = p.number_to_words(y).replace("-", " ").replace(",", "")
                if ends_with_period:
                    y += '.'
            return y

        sentence = " ".join(num2txt(x) for x in sentence.split())

        if raise_error_on_invalid_sentence:
            valid_characters = " '" + self._punctuation_set if self._keep_punctuation else " '"
            if not all(c in valid_characters + string.ascii_lowercase for c in sentence):
                raise RuntimeError()
            if any(x.startswith("'") for x in sentence.split()):
                raise RuntimeError()

        return sentence


__all__ = ["Normalizer"]
