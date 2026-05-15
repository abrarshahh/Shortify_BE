from enum import Enum

class DurationEnum(int, Enum):
    fifteen = 15
    thirty = 30
    sixty = 60

class AspectRatioEnum(str, Enum):
    nine_sixteen = "9:16"
    one_one = "1:1"
    sixteen_nine = "16:9"

class StyleEnum(str, Enum):
    travel = "travel"
    cinematic = "cinematic"
    fast_cut = "fast_cut"
    birthday = "birthday"
    adventure = "adventure"
    romantic = "romantic"
    funny = "funny"
    dramatic = "dramatic"
