import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration
from src.output.common import read_football_output, draw_soccer_field, read_players, draw_overlay


@pytest.fixture
def mock_frame():
    return pd.DataFrame([{
        "frameIdx": 1,
        "homePlayers": [{
            "playerId": "h1", "number": 10, "xyz": [0, 0, 0], "speed": 5.0, "optaId": "optah1"
        }],
        "awayPlayers": [{
            "playerId": "a1", "number": 9, "xyz": [1, 1, 0], "speed": 4.5, "optaId": "optaa1"
        }],
        "ball": {
            "xyz": [0.5, 0.5, 0], "speed": 10.0
        }
    }])

# Mock SoccerPitchConfiguration with a simple example
@pytest.fixture
def mock_config():
    config = MagicMock(spec=SoccerPitchConfiguration)
    config.width = 100
    config.length = 60
    config.centre_circle_radius = 9
    config.penalty_spot_distance = 11
    config.edges = [(1, 2), (2, 3)]  # Simplified example
    config.vertices = [(0, 0), (100, 0), (100, 60)]  # Simplified example
    return config

@pytest.fixture
def dummy_pitch():
    return np.ones((700, 1100, 3), dtype=np.uint8) * 255  # Just a white pitch for testing

@pytest.fixture
def dummy_players():
    player1 = MagicMock()
    player1.xyz = np.array([10, 5, 0])
    player2 = MagicMock()
    player2.xyz = np.array([-10, -5, 0])
    return [player1, player2]

@pytest.fixture
def game_detection_frame(dummy_players):
    frame = MagicMock(spec=GameDetectionFrame)
    frame.home_players = dummy_players
    frame.away_players = dummy_players
    return frame

@patch("src.definitions", "/mock/root")
@patch("src.output.common.pd.read_json")
def test_read_football_output(mock_read_json, mock_frame):
    # Setup mock return for pandas.read_json
    mock_iter = MagicMock()
    mock_iter.__next__.return_value = mock_frame
    mock_read_json.return_value = mock_iter

    result = read_football_output("fake_path.jsonl")

    assert len(result) == 1
    frame = result[0]
    assert isinstance(frame, GameDetectionFrame)
    assert frame.frame_idx == 1
    assert frame.home_players[0].player_id == "h1"
    assert frame.away_players[0].player_id == "a1"
    assert frame.ball.xyz.tolist() == [0.5, 0.5, 0]
    assert frame.last_touch == LastTouch.Home

# Test that the correct dimensions of the soccer field image are returned
@patch("cv2.line")  # Mocking cv2.line
@patch("cv2.circle")  # Mocking cv2.circle
def test_draw_soccer_field(mock_circle, mock_line, mock_config):
    # Call the function with mock configuration
    soccer_field_image = draw_soccer_field(mock_config, padding=50, line_thickness=4, scale=10)

    # Check the image shape (it should include padding around the field)
    expected_height = int(mock_config.width * 10) + 2 * 50
    expected_width = int(mock_config.length * 10) + 2 * 50
    assert soccer_field_image.shape == (expected_height, expected_width, 3)

    # Ensure that the background color is green
    assert np.all(soccer_field_image[0, 0] == [0, 128, 0])

    assert mock_line.called
    assert mock_circle.called

# Test if the scaling factor works correctly and the lines and circles are drawn with the correct scaling
@patch("cv2.line")
@patch("cv2.circle")
def test_field_scaling(mock_circle, mock_line, mock_config):
    # Call the function with scaling
    soccer_field_image = draw_soccer_field(mock_config, padding=50, line_thickness=4, scale=5)

    # The dimensions of the field should now be scaled by a factor of 5
    expected_height = int(mock_config.width * 5) + 2 * 50
    expected_width = int(mock_config.length * 5) + 2 * 50
    assert soccer_field_image.shape == (expected_height, expected_width, 3)

    # Check the scaling for the center circle radius
    expected_circle_radius = int(mock_config.centre_circle_radius * 5)
    mock_circle.assert_any_call(
        img=soccer_field_image,
        center=(expected_width // 2, expected_height // 2),
        radius=expected_circle_radius,
        color=[255, 255, 255],
        thickness=4
    )

@patch("cv2.circle")
def test_read_players_draws_correctly(mock_circle, mock_config, dummy_pitch, dummy_players):
    color = (0, 0, 255)  # Red

    mock_circle.side_effect = lambda img, *args, **kwargs: img

    output_image = read_players(mock_config, dummy_pitch.copy(), dummy_players, color, padding=50, scale=10)

    assert mock_circle.call_count == 2

    for call_args in mock_circle.call_args_list:
        args, kwargs = call_args
        assert kwargs.get("color") == color or args[3] == color
        assert kwargs.get("radius") == 4 or args[2] == 4

    # Output should still be an image (numpy array) with same shape as original
    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == dummy_pitch.shape

@patch("src.output.common.read_players")
@patch("src.output.common.draw_soccer_field")
def test_draw_overlay_combines_images(mock_draw_field, mock_read_players, mock_config, dummy_pitch, game_detection_frame):
    # Create a fake overlay field to simulate drawing
    fake_field = np.zeros_like(dummy_pitch)
    mock_draw_field.return_value = fake_field
    mock_read_players.return_value = fake_field

    result = draw_overlay(mock_config, dummy_pitch, game_detection_frame, fill=0.5)

    # Assertions
    assert isinstance(result, np.ndarray)
    assert result.shape == dummy_pitch.shape

    # Check both drawing functions were called
    mock_draw_field.assert_called_once_with(mock_config, line_thickness=4, scale=10, padding=0)
    assert mock_read_players.call_count == 2

    # Validate that blending happened at correct position
    x_offset = (dummy_pitch.shape[1] - fake_field.shape[1]) // 2
    y_offset = dummy_pitch.shape[0] - fake_field.shape[0]
    roi_result = result[y_offset:y_offset + fake_field.shape[0], x_offset:x_offset + fake_field.shape[1]]
    assert not np.array_equal(roi_result, dummy_pitch[y_offset:y_offset + fake_field.shape[0], x_offset:x_offset + fake_field.shape[1]])