"""Estimate grid orientation and prepare a consistent, border-free frame crop."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar

from session_data import file_hash, write_json


def estimate_rotation(image):
    """Maximize the concentration of grid contrast in row/column projections."""
    gray = np.asarray(image.convert('L'), dtype=np.float32)/255
    scale = min(1., 768/max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    field = cv2.GaussianBlur(gray, (0, 0), 1)-cv2.GaussianBlur(gray, (0, 0), 6)
    if float(field.std()) < 1e-5:
        raise ValueError('No grid contrast for rotation estimation; specify --rotation explicitly')
    height, width = field.shape
    side = int(min(height, width)/np.sqrt(2))-4
    if side < 24:
        raise ValueError('Image is too small to estimate grid orientation')
    y, x = (height-side)//2, (width-side)//2
    def score(angle):
        matrix = cv2.getRotationMatrix2D((width/2, height/2), float(angle), 1)
        rotated = cv2.warpAffine(field, matrix, (width, height), flags=cv2.INTER_LINEAR)
        patch = rotated[y:y+side, x:x+side]
        return float(np.var(patch.mean(axis=0))+np.var(patch.mean(axis=1)))
    angles = np.arange(-45., 45., .5)
    scores = np.array([score(a) for a in angles])
    # Symmetric lattices can tie; prefer the correction nearest the original axes.
    best = max(range(len(angles)), key=lambda i: (scores[i], -abs(angles[i])))
    initial = angles[best]
    fitted = minimize_scalar(lambda angle: -score(angle), bounds=(max(-45., initial-.5), min(45., initial+.5)),
                             method='bounded', options={'xatol': .005})
    if not fitted.success:
        raise RuntimeError('Rotation optimization failed')
    return float(fitted.x)


def crop_size(size, angle):
    """Largest centered rectangle satisfying both rotated image-edge constraints."""
    if not np.isfinite(angle) or abs(angle) > 45:
        raise ValueError('Correction angle must be finite and between -45 and 45 degrees')
    if abs(angle) < 1e-8:
        return tuple(size)
    width, height = np.asarray(size, dtype=float)-4  # Reserve the resampling support.
    c, s = np.cos(np.deg2rad(abs(angle))), np.sin(np.deg2rad(abs(angle)))
    candidates = [(width/(4*c), width/(4*s)), (height/(4*s), height/(4*c))]
    if abs(c*c-s*s) > 1e-10:
        candidates.append(((c*width-s*height)/(2*(c*c-s*s)), (c*height-s*width)/(2*(c*c-s*s))))
    valid = [(a, b) for a, b in candidates if a > 0 and b > 0 and
             c*a+s*b <= width/2+1e-8 and s*a+c*b <= height/2+1e-8]
    if not valid:
        raise ValueError('No valid crop at this correction angle')
    a, b = max(valid, key=lambda pair: pair[0]*pair[1])
    result = int(np.floor(2*a)), int(np.floor(2*b))
    if min(result) < 8:
        raise ValueError('Rotation would leave too little image area')
    return result


def correct_image(image, angle):
    size = crop_size(image.size, angle)
    if abs(angle) < 1e-8:
        return image.copy()
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    x, y = (rotated.width-size[0])//2, (rotated.height-size[1])//2
    return rotated.crop((x, y, x+size[0], y+size[1]))


class RotationEditor:
    def __init__(self, image, angle):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from plot_widgets import Button, TextBox
        from message_panel import MessagePanel
        self.image, self.accepted = image, None
        self.fig, self.axes = plt.subplots(1, 2, figsize=(13, 7))
        self.fig.subplots_adjust(bottom=.20)
        self.axes[0].imshow(image)
        self.axes[0].set_title('Original image; outlined region will be retained')
        self.outline = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor='yellow', linewidth=2)
        self.axes[0].add_patch(self.outline)
        self.preview = self.axes[1].imshow(correct_image(image, angle))
        for ax in self.axes:
            ax.set_axis_off()
        self.angle = TextBox(self.fig.add_axes([.24, .08, .12, .05]), 'Correction (degrees) ', initial=f'{angle:.3f}')
        self.angle.on_submit(self.redraw)
        self.messages = MessagePanel(self.fig, [.05, .015, .90, .045])
        self.error = self.messages.channel('message')
        self.apply_button = Button(self.fig.add_axes([.61, .08, .20, .05]), 'Use rotation and crop')
        self.apply_button.on_clicked(self.apply)
        self.redraw()

    def redraw(self, value=None):
        try:
            angle = float(self.angle.text)
            crop = correct_image(self.image, angle)
        except ValueError as error:
            self.error.set_text(str(error))
            self.fig.canvas.draw_idle()
            return False
        width, height = crop.size
        corners = np.array([[-width, -height], [width, -height], [width, height], [-width, height]])/2
        radians = np.deg2rad(angle)
        inverse = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        self.outline.set_xy(corners@inverse.T+np.asarray(self.image.size)/2)
        self.preview.set_data(crop)
        self.axes[1].set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), title=f'Corrected crop: {width} × {height} pixels')
        self.error.set_text('Positive angles rotate counterclockwise. Press Enter to preview the crop.')
        self.fig.canvas.draw_idle()
        return True

    def apply(self, event=None):
        if self.redraw():
            import matplotlib.pyplot as plt
            from plot_widgets import disconnect_widgets
            self.accepted = float(self.angle.text)
            disconnect_widgets([self.angle, self.apply_button])
            self.messages.close()
            plt.close(self.fig)

    def show(self):
        import matplotlib.pyplot as plt
        from plot_widgets import disconnect_widgets
        try:
            plt.show()
        finally:
            disconnect_widgets([self.angle, self.apply_button])
            self.messages.close()
        return self.accepted


def prepare_frames(files, output, *, angle=None, interactive=True, backend='TkAgg', reference=None):
    """Original images never change; all later stages use the same saved transform."""
    output = Path(output)
    path = output/'rotation.json'
    sources = [dict(path=str(p.resolve()), sha256=file_hash(p)) for p in files]
    with Image.open(files[0]) as source:
        first = ImageOps.exif_transpose(source).convert('RGB')
    reuse = path.exists()
    if reuse:
        document = json.loads(path.read_text())
        if document['sources'] != sources:
            raise ValueError('Saved rotation belongs to different input frames; choose a new results folder')
        if angle is not None and not np.isclose(angle, document['angle'], atol=1e-9):
            raise ValueError('Choose a new results folder to change a saved rotation without invalidating human edits')
        angle = document['angle']
    else:
        existing = ((reference is not None and Path(reference).exists()) or
                    (output/'reference.draft.json').exists() or (output/'review.json').exists())
        if existing and angle is not None and abs(angle) > 1e-8:
            raise ValueError('Choose a new results folder to rotate a sequence with saved human references')
        estimated = 0. if existing or angle is not None else estimate_rotation(first)
        if angle is None:
            angle = 0.
            if abs(estimated) > .5:
                if not interactive:
                    raise ValueError(f'Estimated correction is {estimated:.3f} degrees. Run interactively to review the crop, or pass --rotation DEGREES')
                from reference_editor import configure_backend
                configure_backend(backend)
                angle = RotationEditor(first, estimated).show()
                if angle is None:
                    raise SystemExit('Rotation setup canceled; no frames were analyzed')
        document = dict(angle=float(angle), estimated_angle=estimated, original_size=list(first.size),
                        image_size=list(crop_size(first.size, angle)), sources=sources,
                        convention='counterclockwise rotation followed by centered valid crop; coordinates refer to corrected pixels')
    if abs(angle) < 1e-8:
        write_json(path, document)
        return list(files)
    prepared = output/'rotated_frames'
    prepared.mkdir(exist_ok=True)
    paths = []
    for index, source_path in enumerate(files):
        target = prepared/f'frame_{index:06d}.png'
        if not reuse or not target.exists():
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            if list(image.size) != document['original_size']:
                raise ValueError(f'Frame dimensions changed: {source_path}')
            temporary = target.with_suffix('.tmp')
            correct_image(image, angle).save(temporary, format='PNG')
            temporary.replace(target)
        paths.append(target.resolve())
    write_json(path, document)
    return paths
