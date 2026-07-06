import pytest
from pydantic import ValidationError
from backend_ai.schemas.edl import ColorGradeParams as PydanticColorGradeParams
from backend_ai.effects.color import ColorGradeParams as DataclassColorGradeParams, build_ffmpeg_color_filter

def test_color_grade_params_validation():
    # Valid params
    valid = {
        "brightness": 1.2,
        "contrast": 1.1,
        "gamma": 1.0,
        "saturation": 0.9,
        "vibrance": 1.1,
        "hue": 10.0,
        "temperature": -5.0,
        "vignette_strength": 0.3,
        "vignette_radius": 0.8
    }
    params = PydanticColorGradeParams(**valid)
    assert params.brightness == 1.2
    assert params.vignette_radius == 0.8

    # Default values
    defaults = PydanticColorGradeParams()
    assert defaults.brightness == 1.0
    assert defaults.hue == 0.0

    # Invalid values (out of bounds)
    with pytest.raises(ValidationError):
        PydanticColorGradeParams(brightness=0.1)  # min is 0.5
    with pytest.raises(ValidationError):
        PydanticColorGradeParams(contrast=2.5)  # max is 2.0
    with pytest.raises(ValidationError):
        PydanticColorGradeParams(hue=190.0)  # max is 180.0
    with pytest.raises(ValidationError):
        PydanticColorGradeParams(temperature=-60.0)  # min is -50.0
    with pytest.raises(ValidationError):
        PydanticColorGradeParams(vignette_strength=1.5)  # max is 1.0

def test_build_ffmpeg_color_filter():
    # Neutral params -> empty string
    neutral = DataclassColorGradeParams()
    assert build_ffmpeg_color_filter(neutral) == ""

    # Non-neutral params
    custom = DataclassColorGradeParams(
        brightness=1.2,
        contrast=1.1,
        gamma=1.5,
        saturation=1.2,
        temperature=25.0,
        hue=15.0,
        vignette_strength=0.5
    )
    filter_str = build_ffmpeg_color_filter(custom)
    
    # Check expected filters in the string
    assert "eq=" in filter_str
    # contrast, brightness (1.2 - 1.0 = 0.2), saturation (1.2), gamma (1.5)
    assert "contrast=1.1" in filter_str
    assert "brightness=0.2" in filter_str
    assert "saturation=" in filter_str
    assert "gamma=1.5" in filter_str
    
    # colorbalance for temperature
    assert "colorbalance=" in filter_str
    
    # hue
    assert "hue=h=15.0" in filter_str
    
    # vignette
    assert "vignette=angle=" in filter_str
