import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from supervision import KeyPoints

from src.detectionframe.detectionsVariablesCalculator.detection_location import get_location_of_bounding_box


@pytest.fixture
def sample_keypoints():
    xy = np.array([[[10, 10], [15, 15], [20, 20], [25, 25], [30, 30]]], dtype=np.float32)
    return KeyPoints(xy=xy)


@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.load_pitch_data")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_points_manager")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_point_saver")
def test_get_location_valid_input(mock_saver, mock_manager, mock_load, sample_keypoints):
    mock_load.return_value = {"pitchLength": 100, "pitchWidth": 50}

    mock_transformer = MagicMock()
    mock_transformer.transform_points.return_value = [[np.array([60.0, 10.0])]]
    mock_manager.get_transformer.return_value = mock_transformer

    xy = np.array([1.0, 1.0], dtype=np.float32)

    result = get_location_of_bounding_box(xy, sample_keypoints)

    expected = np.array([10.0, 15.0])
    np.testing.assert_almost_equal(result, expected, decimal=3)

    mock_manager.add_new_key_point.assert_called_once()


@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.load_pitch_data")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_points_manager")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_point_saver")
def test_fallback_to_good_keypoints(mock_saver, mock_manager, mock_load, sample_keypoints):
    mock_load.return_value = {"pitchLength": 100, "pitchWidth": 50}

    bad_keypoints = KeyPoints(xy=np.array([[[0, 0], [0, 0], [0, 0]]], dtype=np.float32))

    good_xy = np.array([1.0, 1.0], dtype=np.float32)
    good_kp = sample_keypoints

    mock_saver.load_good_key_points.return_value = [good_xy, good_kp]
    mock_transformer = MagicMock()
    mock_transformer.transform_points.return_value = [[np.array([50.0, 25.0])]]
    mock_manager.get_transformer.return_value = mock_transformer

    result = get_location_of_bounding_box(np.array([0.0, 0.0], dtype=np.float32), bad_keypoints)

    expected = np.array([0.0, 0.0])
    np.testing.assert_almost_equal(result, expected, decimal=3)


@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.load_pitch_data")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_points_manager")
@patch("src.detectionframe.detectionsVariablesCalculator.detection_location.key_point_saver")
def test_no_good_keypoints_raises(mock_saver, mock_manager, mock_load):
    mock_load.return_value = {"pitchLength": 100, "pitchWidth": 50}

    empty_keypoints = KeyPoints(xy=np.array([[[0, 0], [0, 0]]], dtype=np.float32))
    mock_saver.load_good_key_points.return_value = []

    with pytest.raises(ValueError, match="No good key points found for given frame"):
        get_location_of_bounding_box(np.array([0.0, 0.0], dtype=np.float32), empty_keypoints)