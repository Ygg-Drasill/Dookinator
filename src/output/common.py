import os

import cv2
import numpy as np
import pandas as pd

from src.definitions import ROOT_DIR
from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.detectionframe.player_detection_frame import PlayerDetectionFrame
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration


def read_football_output(file_path: str) -> list[GameDetectionFrame]:
    """
    Reads and parses football tracking data from a JSON Lines file into structured game detection frames.

    This function loads player and ball tracking data from a `.jsonl` file, processes it into
    structured `GameDetectionFrame` objects, and returns a list of these frames. Each frame
    includes information about home and away players, as well as the ball if present.

    Args:
        file_path (str): Relative path to the JSON Lines file containing tracking data. The file is expected
            to be in JSON Lines format with one frame per line.

    Returns:
        list[GameDetectionFrame]: A list of parsed game detection frames, where each frame contains structured
            data on player and ball positions for a specific moment in time.
    """

    json_iter = pd.read_json(path_or_buf=os.path.join(ROOT_DIR, file_path), lines=True, chunksize=10000)
    game_detection_frames = []

    json_obj = next(json_iter)

    for _, frame in json_obj.iterrows():
        home_players = np.array([
            PlayerDetectionFrame(
                player_id=p["playerId"],
                number=p["number"],
                xyz=np.array(p["xyz"]),
                speed=p["speed"],
                opta_id=p["optaId"]
            ) for p in frame["homePlayers"]
        ])

        away_players = np.array([
            PlayerDetectionFrame(
                player_id=p["playerId"],
                number=p["number"],
                xyz=np.array(p["xyz"]),
                speed=p["speed"],
                opta_id=p["optaId"]
            ) for p in frame["awayPlayers"]
        ])

        ball_raw = frame["ball"]
        ball = BallDetectionFrame(xyz=np.array(ball_raw["xyz"]), speed=ball_raw["speed"]) if ball_raw and not pd.isna(ball_raw) else None

        game_detection_frame = GameDetectionFrame(
            period=0, frame_idx=frame["frameIdx"], game_clock=0, wall_clock=0,
            home_players=home_players, away_players=away_players,
            ball=ball, live=True, last_touch=LastTouch.Home
        )
        game_detection_frames.append(game_detection_frame)

    return game_detection_frames

def draw_soccer_field(config: SoccerPitchConfiguration, padding: int = 50, line_thickness: int = 4, scale: float = 10) -> np.ndarray:
    """
    Renders a scaled soccer field based on a given configuration.

    This function creates a visual representation of a soccer pitch using the provided
    layout configuration. It draws field boundaries, the center circle, and penalty spots.
    Padding and scaling allow customization of the output image size and detail.

    Args:
        config (SoccerPitchConfiguration): Configuration describing the field layout,
            including dimensions, vertex positions, and line connections.
        padding (int, optional): Space around the field in pixels. Defaults to 50.
        line_thickness (int, optional): Thickness of the lines drawn for field edges and the center circle. Defaults to 4.
        scale (float, optional): Scaling factor to convert real-world units into image pixels. Defaults to 10.

    Returns:
        np.ndarray: RGB image of the soccer field with specified markings and layout.
    """

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    point_radius = 6

    soccer_field_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    )

    soccer_field_image[:] = [0, 128, 0]

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)

        cv2.line(
            img=soccer_field_image,
            pt1=point1,
            pt2=point2,
            color=[255, 255, 255],
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=soccer_field_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=[255, 255, 255],
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=soccer_field_image,
            center=spot,
            radius=point_radius,
            color=[255, 255, 255],
            thickness=-1
        )

    return soccer_field_image

def read_players(config: SoccerPitchConfiguration, pitch: np.ndarray, players: np.ndarray, color: tuple[int, int, int], padding: int = 50,scale: float = 10) -> np.ndarray:
    """
    Draws player positions onto a soccer pitch image.

    This function takes player coordinates and overlays them as circles on the given pitch image.
    Each player's position is scaled and offset based on the pitch dimensions and a padding value.

    Args:
        config (SoccerPitchConfiguration): Configuration object that provides the field dimensions.
        pitch (np.ndarray): Image of the soccer field on which to draw players.
        players (np.ndarray): Array of player positions; each player should have at least x and y coordinates.
        color (tuple[int, int, int]): BGR color tuple for the circles representing players.
        padding (int, optional): Padding added around the pitch for drawing. Defaults to 50.
        scale (float, optional): Scaling factor to convert real-world coordinates to image coordinates. Defaults to 10.

    Returns:
        np.ndarray: Updated pitch image with player markers drawn.
    """

    new_pitch = pitch
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    for player in players:
        center_of_circle = player.xyz[:2]

        real_width = (np.take(center_of_circle, 0) * scale) + padding + scaled_width / 2
        real_height = (np.take(center_of_circle, 1) * scale) + padding + scaled_length / 2

        new_pitch = cv2.circle(pitch, (int(real_width), int(real_height)), 4, color, -1)
        pass

    return new_pitch

def draw_overlay(config: SoccerPitchConfiguration, image: np.ndarray, game_detection_frame: GameDetectionFrame, fill = 0.5, line_thickness: int = 4, scale: float = 10) -> np.ndarray:
    """
    Draws a visual overlay of a soccer field and player positions onto an image.

    This function overlays a rendered soccer field and the detected home team players
    onto the input image. The overlay is blended with the image for visualization
    purposes, such as analysis or debugging.

    Args:
        config (SoccerPitchConfiguration): Configuration object describing the layout and dimensions of the soccer pitch.
        image (np.ndarray): Input image (e.g., video frame) on which the overlay is to be drawn.
        game_detection_frame (GameDetectionFrame): Frame object containing home team player positions.
        fill (float, optional): Alpha blending factor for overlay transparency.
            Ranges from 0 (no overlay) to 1 (full overlay). Defaults to 0.5.
        line_thickness (int, optional): Thickness of the lines used to draw the field. Defaults to 4.
        scale (float, optional): Scaling factor for rendering the field and players. Defaults to 10.

    Returns:
        np.ndarray: Image with the soccer field and players overlaid.
    """

    image_with_overlay = image.copy()

    soccer_field = draw_soccer_field(config, line_thickness=line_thickness, scale=scale, padding=0)

    soccer_field = read_players(config, soccer_field, game_detection_frame.home_players, (0, 0, 255), scale=scale, padding=0)

    if soccer_field.shape[0] > image.shape[0] or soccer_field.shape[1] > image.shape[1]:
        scale_factor = min(image.shape[1] / soccer_field.shape[1], image.shape[0] / soccer_field.shape[0], 1.0)
        soccer_field = cv2.resize(
            soccer_field,
            (int(soccer_field.shape[1] * scale_factor), int(soccer_field.shape[0] * scale_factor))
        )

    field_h, field_w = soccer_field.shape[:2]
    img_h, img_w = image.shape[:2]

    #place the image in the right place
    x_offset = (img_w - field_w) // 2
    y_offset = img_h - field_h

    # Blend only the region of interest (ROI)
    roi = image_with_overlay[y_offset:y_offset + field_h, x_offset:x_offset + field_w]
    blended = cv2.addWeighted(roi, 1 - fill, soccer_field, fill, 0)

    # Place the blended region back into the image
    image_with_overlay[y_offset:y_offset + field_h, x_offset:x_offset + field_w] = blended

    return image_with_overlay