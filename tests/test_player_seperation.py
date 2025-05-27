import numpy as np
from unittest.mock import patch, MagicMock

import supervision as sv
from src.player_seperation import (
    create_batches,
    resolve_goalkeepers_team_id,
    TeamClassifier,
    get_crops,
    player_separation
)


def test_create_batches():
    data = list(range(10))
    batches = list(create_batches(data, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_resolve_goalkeepers_team_id():
    players = sv.Detections(xyxy=np.array([[0, 0, 2, 2], [5, 0, 7, 2]]))
    goalkeepers = sv.Detections(xyxy=np.array([[1, 0, 3, 2], [6, 0, 8, 2]]))
    players_team_id = np.array([0, 1])

    result = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
    assert result.shape == (2,)
    assert set(result.tolist()) <= {0, 1}


@patch('src.player_seperation.SiglipVisionModel.from_pretrained')
@patch('src.player_seperation.AutoProcessor.from_pretrained')
def test_team_classifier_fit_and_predict(mock_processor, mock_model):
    dummy_embedding = np.random.rand(1, 10)
    dummy_projection = np.random.rand(1, 3)

    mock_model.return_value = MagicMock()
    mock_processor.return_value = MagicMock()

    classifier = TeamClassifier(device='cpu', batch_size=1)
    classifier.extract_features = MagicMock(return_value=dummy_embedding)
    classifier.reducer.fit_transform = MagicMock(return_value=dummy_projection)
    classifier.cluster_model.fit_predict = MagicMock(return_value=np.array([0]))

    classifier.fit([np.zeros((224, 224, 3), dtype=np.uint8)])
    assert classifier.knn_classifier is not None

    classifier.reducer.transform = MagicMock(return_value=dummy_projection)
    prediction = classifier.predict([np.zeros((224, 224, 3), dtype=np.uint8)])
    assert prediction.shape == (1,)


def test_get_crops():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]))

    crops = get_crops(frame, detections)
    assert len(crops) == 2
    assert all(isinstance(crop, np.ndarray) for crop in crops)




@patch('src.player_seperation.get_crops')
@patch('src.player_seperation.team_classifier')
def test_player_separation(mock_team_classifier, mock_get_crops, monkeypatch):
    monkeypatch.setattr('src.player_seperation.yolo', {
        'selected_class_ids': {
            'player': {'id': 0},
            'goalkeeper': {'id': 1}
        }
    })

    frame = np.zeros((224, 224, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([
            [10, 10, 20, 20], # player
            [30, 30, 40, 40]  # goalkeeper
        ]),
        class_id=np.array([0, 1])
    )

    mock_get_crops.return_value = [frame]
    mock_team_classifier.fit.return_value = None
    mock_team_classifier.predict.return_value = np.array([0])

    from src.player_seperation import player_separation

    detections_chunk = [detections]
    result = list(player_separation([frame], detections_chunk))

    assert isinstance(result, list)
    assert all(isinstance(d, sv.Detections) for d in result)
    assert "player_team" in result[0].data
    assert len(result[0]["player_team"]) == 2
