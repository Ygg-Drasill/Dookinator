import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.detectionframe.detectionsVariablesCalculator.detection_location import get_location_of_bounding_box


@pytest.fixture
def mock_data():
    return {
        "pitchLength": 105,
        "pitchWidth": 68
    }


@pytest.fixture
def mock_keypoints():
    keypoints = MagicMock()
    # Create fake keypoints that pass the mask filter
    keypoints.xy = [
        np.array([
            [10.0, 20.0],
            [30.0, 40.0],
            [0.5, 0.5],
        ])
    ]
    return keypoints


@pytest.fixture
def mock_vertices():
    return [
        [1000.0, 2000.0],
        [3000.0, 4000.0],
        [5000.0, 6000.0]
    ]

@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.ViewTransformer")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.SoccerPitchConfiguration")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.load_pitch_data")
def test_get_location_of_bounding_box(
    mock_load_pitch_data,
    mock_config_class,
    mock_transformer_class,
    mock_keypoints,
    mock_vertices
):
    mock_load_pitch_data.return_value = {
        "pitchLength": 105,
        "pitchWidth": 68
    }

    mock_config_instance = MagicMock()
    mock_config_instance.vertices = mock_vertices
    mock_config_instance.pitchLength = 105
    mock_config_instance.pitchWidth = 68
    mock_config_class.return_value = mock_config_instance

    mock_transformer_instance = MagicMock()
    mock_transformer_instance.transform_points.return_value = np.array([[[123.4, 567.8]]])
    mock_transformer_class.return_value = mock_transformer_instance

    xy = np.array([15.0, 25.0], dtype=np.float32)
    result = get_location_of_bounding_box(xy, mock_keypoints)

    mock_transformer_instance.transform_points.assert_called_once()
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, np.array([123.4 - 34, 567.8 - 52.5]), decimal=1)