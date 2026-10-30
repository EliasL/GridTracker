"""MP4 previews of fitted grid crossings, independent of CSV export."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from review_session import curve_crossing_snapshot


def export_video(report, path, *, playback_fps=30, progress=None):
    """Write one preview frame per source image; never invent missing crossings."""
    path = Path(path)
    if path.suffix.lower() != '.mp4':
        raise ValueError('Choose an .mp4 destination')
    if not np.isfinite(playback_fps) or playback_fps <= 0:
        raise ValueError('Playback rate must be finite and positive')
    if not report['frames']:
        raise ValueError('No frames are available to export')
    frames = curve_crossing_snapshot(report)['frames']
    size = tuple(frames[0]['image_size'])
    video_size = tuple(value+value % 2 for value in size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.mp4', delete=False) as stream:
        temporary = Path(stream.name)
    writer = None
    try:
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), playback_fps, video_size)
        if not writer.isOpened():
            raise RuntimeError('Cannot open the MP4 encoder supplied by OpenCV')
        for index, frame in enumerate(frames):
            with Image.open(frame['source']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            if tuple(frame['image_size']) != size or image.size != size:
                raise ValueError('All video frames must have the same oriented image dimensions')
            pixels = cv2.cvtColor(np.asarray(ImageEnhance.Brightness(image).enhance(3)), cv2.COLOR_RGB2BGR)
            pixels = cv2.copyMakeBorder(pixels, 0, size[1] % 2, 0, size[0] % 2, cv2.BORDER_CONSTANT)
            for node in frame['nodes']:
                if node['status'] != 'curve_crossing':
                    continue
                point = tuple(int(round(node[axis])) for axis in ('x', 'y'))
                cv2.circle(pixels, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(pixels, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(pixels, f"Frame {frame['frame']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(pixels, f"Frame {frame['frame']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(pixels)
            if progress is not None:
                progress(index+1, len(frames))
        writer.release()
        writer = None
        capture = cv2.VideoCapture(str(temporary))
        try:
            valid, _ = capture.read()
            dimensions = tuple(int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if not valid or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames) or dimensions != video_size:
                raise RuntimeError('MP4 verification failed: incomplete video or unexpected dimensions')
        finally:
            capture.release()
        temporary.replace(path)
    finally:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
    return path
