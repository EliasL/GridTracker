"""Parallel detection, grid initialization, and tracking with per-node human edits."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLCONFIGDIR'] = str(__import__('pathlib').Path(__file__).parent/'.matplotlib-cache')

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
from session_data import write_json, file_hash, parse_fps, validate_reference, load_reference


_DEFAULT_FPS = object()


def analyze(task, diagnostics=False):
    # Never import the Agg detector in the GUI/main process.
    from pipeline import run_frame
    return run_frame(task, fit_curves=False, diagnostics=diagnostics)


def discover_frames(folder, order='download', output=None):
    if not folder.is_dir():
        raise ValueError(f'Not an image folder: {folder}')
    # These are application artifacts, never video frames. Subfolder artifacts
    # are already excluded by the nonrecursive scan.
    excluded = {'tracking_overview.png', 'calibration_image.png', 'calibration_input.png'} if output is not None and folder.resolve() == Path(output).resolve() else set()
    files = [p.resolve() for p in folder.iterdir() if p.is_file() and p.name.lower() not in excluded
             and p.suffix.lower() in ('.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp')]
    if order == 'filename':
        files.sort(key=lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', p.name)])
    else:
        files.sort(key=lambda p: (getattr(p.stat(), 'st_birthtime', p.stat().st_mtime), p.name))
    if not files:
        raise ValueError(f'No image frames found in {folder}')
    numbered = [re.fullmatch(r'(.*?)(\d+)', p.stem) for p in files]
    offsets = list(range(len(files)))
    if all(numbered) and len({m[1] for m in numbered}) == 1:
        indices = [int(m[2]) for m in numbered]
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError('Frame numbers disagree with download order; use --order filename')
        offsets = [i-indices[0] for i in indices]
    return files, offsets


def detection_signature():
    names = ('pipeline', 'detection', 'string_color', 'crossing_nodes', 'scan_node_lines',
             'local_peak_spacing', 'edge_nodes', 'complete_edge_nodes', 'local_node_deviations')
    return hashlib.sha256(b''.join(Path(__file__).with_name(n+'.py').read_bytes() for n in names)).hexdigest()


def propagate(results, reference, output, fps=_DEFAULT_FPS, method='flow', confirmed=None, progress=None,
              frame_range=None, previous_report=None, edits=None):
    from copy import deepcopy
    import numpy as np
    from PIL import Image, ImageOps
    from string_color import color_model_grayscale
    from fit_bezier_curves import fit_labeled_curves
    from temporal_tracking import NodeTracker, TrackingConfig, reconcile_nodes, hold_missing_positions, TRACKING_POLICY_VERSION
    from tracking_review import save_overview

    if not results or [r['frame'] for r in results] != list(range(len(results))):
        raise ValueError('Require a complete ordered frame sequence starting at zero')
    validate_reference(reference, results[0])
    anchors = {0: reference, **(confirmed or {})}
    edits = edits or {}
    definitions = {}
    for index, document in anchors.items():
        if not 0 <= index < len(results):
            raise ValueError(f'Anchor frame {index} is outside this sequence')
        validate_reference(document, results[index])
        for node in document['nodes']:
            definitions[node['id']] = node
    from review_session import validate_roster
    validate_roster(anchors)
    for index, values in edits.items():
        if not 0 <= index < len(results):
            raise ValueError('Human edit frame is outside this sequence')
        for node in values.values():
            if node is not None:
                definitions[node['id']] = node
    if any(r['source_sha256'] != file_hash(r['source']) for r in results):
        raise ValueError('Input pixels changed since independent detection')
    fps = parse_fps(reference.get('fps') if fps is _DEFAULT_FPS else fps)
    first, last = (0, len(results)-1) if frame_range is None else frame_range
    if not 0 <= first <= last < len(results):
        raise ValueError('Reanalysis range is outside this frame sequence')
    partial = frame_range is not None
    if partial and (previous_report is None or len(previous_report['frames']) != len(results)):
        raise ValueError('Partial reanalysis requires an existing complete tracking report')
    if partial and previous_report.get('fps') != fps:
        raise ValueError('Use full reanalysis after changing the frame rate')
    # Replay from the nearest earlier human reference to preserve tracker state.
    # Only the requested range is replaced or written back.
    begin = max(i for i in anchors if i <= first) if partial else 0
    config = TrackingConfig(use_flow=method == 'flow', use_velocity=method == 'flow')
    dt = 1/fps if fps is not None else 1.
    if any(len(results[0]['maxima'][a]['clean']) < 2 for a in 'xy'):
        raise ValueError('Need at least two scaffold lines on both axes to establish spacing')
    spacing = min(float(np.median(np.diff(results[0]['maxima'][a]['clean']))) for a in 'xy')
    if not np.isfinite(spacing):
        raise ValueError('Need at least two scaffold lines on both axes to establish spacing')
    model = results[0]['color_model']
    frames = deepcopy(previous_report['frames']) if partial else [None]*len(results)
    tracker, grays = None, {}
    deleted_until_observed = set()
    offsets = [r.get('frame_offset', r['frame']) for r in results]
    if any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Frame times must increase strictly')
    for result in results[begin:last+1]:
        index = result['frame']
        step_dt = (offsets[index]-offsets[index-1])*dt if index else dt
        with Image.open(result['source']) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
        gray = np.uint8(np.round(255*color_model_grayscale(rgb, model)))
        grays[index] = gray
        if tracker is None:
            tracker = NodeTracker(anchors[begin]['nodes'], gray, result['width'], spacing, dt, config)
            nodes, unclaimed = tracker.initial(), []
        elif result['frame'] in anchors:
            nodes = tracker.anchor(anchors[result['frame']]['nodes'], gray, dt=step_dt)
            unclaimed = []
        else:
            nodes, unclaimed = tracker.step(gray, result['candidate_nodes'], dt=step_dt)
        if result['frame'] in edits:
            nodes = tracker.constrain(edits[result['frame']], gray, nodes)
        for key, value in edits.get(index, {}).items():
            if value is None:
                deleted_until_observed.add(key)
            else:
                deleted_until_observed.discard(key)
        deleted_until_observed -= {n['id'] for n in nodes if n.get('measured_xy') is not None}
        nodes = [n for n in nodes if n['id'] not in deleted_until_observed]
        if begin:
            for node in nodes:
                node['last_observed_frame'] += begin
        curves, ignored = fit_labeled_curves(nodes, result['image_size'], result['width'])
        frame = dict(frame=result['frame'], source=result['source'], image_size=result['image_size'],
                     original_source=result.get('original_source', result['source']),
                     rotation_degrees=result.get('rotation_degrees', 0.),
                     width=result['width'],
                     detection_issues=result.get('detection_issues', []),
                     human_verified=(result['frame'] in anchors and
                                     anchors[result['frame']].get('verification_scope', 'frame') == 'frame'),
                     human_edited_ids=[key for key, value in edits.get(index, {}).items() if value is not None],
                     deleted_node_ids=[key for key, value in edits.get(index, {}).items() if value is None],
                     frame_offset=offsets[index], time=offsets[index]*dt, nodes=nodes, unclaimed_detections=unclaimed,
                     candidate_tracks=tracker.candidate_states(), candidate_events=list(tracker.candidate_events),
                     curves=curves, ignored_curves=ignored,
                     counts={status: sum(n['status'] == status for n in nodes)
                             for status in ('verified', 'observed', 'recovered', 'held', 'predicted', 'lost', 'outside')})
        if not partial or first <= index <= last and index not in anchors:
            frames[index] = frame
        print(f"Tracked frame {result['frame']}: {frame['counts']}; {len(unclaimed)} unclaimed detections", flush=True)
        if progress is not None:
            progress(index-begin+1, last-begin+1)
    # Later human anchors also inform the intervening unconfirmed frames.
    ordered = sorted(i for i in anchors if begin <= i <= last)
    for left, right in zip(ordered, ordered[1:]):
        visible = anchors[right]['nodes']
        if not visible:
            continue  # A completely occluded anchor has no patches to back-track.
        reverse = NodeTracker(visible, grays[right], results[right]['width'], spacing, dt, config)
        for index in range(right-1, max(left, first-1), -1):
            backward, _ = reverse.step(grays[index], results[index]['candidate_nodes'], dt=(offsets[index+1]-offsets[index])*dt)
            if index in edits:
                backward = reverse.constrain(edits[index], grays[index], backward)
            for node in backward:
                if node.get('velocity') is not None:
                    node['velocity'] = (-np.array(node['velocity'])).tolist()
                node['last_observed_frame'] = right-node['last_observed_frame']
            frame = frames[index]
            frame['nodes'] = reconcile_nodes(frame['nodes'], backward, definitions, results[index]['width'])
            used = {n['detection_id'] for n in frame['nodes'] if n['measured_xy'] is not None}
            frame['unclaimed_detections'] = [n for n in frame['unclaimed_detections'] if n['id'] not in used]
            remaining = {n['candidate_id'] for n in frame['unclaimed_detections']}
            frame['candidate_tracks'] = [c for c in frame['candidate_tracks'] if c['candidate_id'] in remaining]
            frame['candidate_events'] = [c for c in frame['candidate_events'] if c['candidate_id'] in remaining]
            frame['curves'], frame['ignored_curves'] = fit_labeled_curves(frame['nodes'], frame['image_size'], results[index]['width'])
            frame['counts'] = {status: sum(n['status'] == status for n in frame['nodes']) for status in frame['counts']}
    if partial:
        prefix = deepcopy(frames[:first])+frames[first:last+1]
        hold_missing_positions(prefix)
        frames[first:last+1] = prefix[first:]
    else:
        hold_missing_positions(frames)
    for frame in frames[first:last+1]:
        if not frame['human_verified']:
            frame['curves'], frame['ignored_curves'] = fit_labeled_curves(frame['nodes'], frame['image_size'], frame['width'])
            frame['counts'] = {status: sum(n['status'] == status for n in frame['nodes']) for status in frame['counts']}
    report = dict(reference_sha256=hashlib.sha256(json.dumps(anchors[0], sort_keys=True).encode()).hexdigest(),
                  tracking_policy_version=TRACKING_POLICY_VERSION,
                  fps=fps, time_unit='seconds' if fps is not None else 'frames', method=method,
                  velocity_unit='pixels/second' if fps is not None else 'pixels/frame',
                  acceleration_unit='pixels/second^2' if fps is not None else 'pixels/frame^2',
                  config=asdict(config), new_tracks='require human verification; never automatically promoted',
                  curve_input='human/observed positions and held boundary endpoints; missing interior predictions excluded',
                  confirmed_frames=sorted(anchors), frames=frames)
    from review_session import update_kinematics
    update_kinematics(report)
    if partial:
        for index in range(len(frames)):
            if not first <= index <= last or index in anchors:
                frames[index] = deepcopy(previous_report['frames'][index])
        report['reanalyzed_range'] = [first, last]
    for frame in frames[first:last+1]:
        if partial and frame['frame'] in anchors:
            continue
        write_json(output/f"tracked_{frame['frame']:02d}.json", frame)
    write_json(output/'tracking.json', report)
    save_overview(report, output/'tracking_overview.png')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', help='Folder of image frames')
    parser.add_argument('--input', type=Path, help='Alternative to the positional folder')
    parser.add_argument('--output', type=Path, help='Default: <frames folder>/gridtracker_results')
    parser.add_argument('--order', choices=('download', 'filename'), default='download')
    parser.add_argument('--diagnostics', action='store_true', help='Also save the detailed detector plots and arrays')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--fps', type=float, help='Sampling rate of these frames, not necessarily the source video FPS')
    parser.add_argument('--method', choices=('flow', 'position'), default='flow')
    parser.add_argument('--prepare-only', action='store_true', help='Detect/cache frames without opening the editor')
    parser.add_argument('--reference', type=Path, help='Reuse a previously human-approved reference')
    parser.add_argument('--backend', default='TkAgg', help='Interactive Matplotlib backend')
    parser.add_argument('--rotation', type=float, help='Explicit counterclockwise correction in degrees; zero disables rotation')
    args = parser.parse_args(argv)
    if args.folder is not None and args.input is not None:
        parser.error('Specify the folder once, either positionally or with --input')
    args.input = args.folder or args.input or Path('testData')
    args.output = args.output or args.input/'gridtracker_results'
    if args.workers < 1 or (args.fps is not None and not 0 < args.fps < float('inf')):
        parser.error('Require positive worker count and a finite positive frame sampling rate')
    files, offsets = discover_frames(args.input, args.order, args.output)
    gaps = sum(b-a-1 for a, b in zip(offsets, offsets[1:]))
    print(f'{len(files)} frames; {gaps} missing numbered frames; output: {args.output.resolve()}', flush=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from rotation import prepare_frames
    reference_path = args.reference or args.output/'reference.json'
    original_files = files
    files = prepare_frames(files, args.output, angle=args.rotation, interactive=not args.prepare_only,
                           backend=args.backend, reference=reference_path)
    rotation = json.loads((args.output/'rotation.json').read_text())
    from reference_editor import ReferenceEditor, configure_backend
    signature = detection_signature()
    results, futures, errors = {}, {}, []
    manifest = dict(sources=[str(p) for p in files], original_sources=[str(p) for p in original_files], rotation_degrees=rotation['angle'], workers=args.workers, threads_per_worker=1,
                    detection_sha256=signature, results=[], errors=errors)
    manifest_path = args.output/'detections.json'
    manifest_written = False

    def collect():
        nonlocal manifest_written
        changed = False
        for index in list(futures):
            future = futures[index]
            if not future.done():
                continue
            futures.pop(index)
            changed = True
            try:
                result = future.result()
                result.update(source_sha256=file_hash(files[index]), detection_sha256=signature, frame_offset=offsets[index],
                              original_source=str(original_files[index]), rotation_degrees=rotation['angle'])
                write_json(args.output/f'frame_{index:02d}'/'result.json', result)
                results[index] = result
                print(f'Detection ready: frame {index}, {len(result["final_nodes"])} nodes', flush=True)
            except Exception as error:
                errors.append(dict(frame=index, error=repr(error)))
        manifest['results'] = [results[i] for i in sorted(results)]
        if changed or not manifest_written:
            write_json(manifest_path, manifest)
            manifest_written = True
        return True  # Keep the GUI timer alive.

    approval, figure = None, None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # Submission order prioritizes frame zero; workers continue while it is edited.
        for i, source in enumerate(files):
            cache = args.output/f'frame_{i:02d}'/'result.json'
            cached = json.loads(cache.read_text()) if cache.exists() else None
            if (cached is not None and cached.get('detection_sha256') == signature
                    and cached.get('source_sha256') == file_hash(source) and cached.get('source') == str(source)
                    and cached.get('frame') == i and not cached.get('fit_curves', True)
                    and cached.get('diagnostics', True) == args.diagnostics):
                cached['frame_offset'] = offsets[i]
                cached.update(original_source=str(original_files[i]), rotation_degrees=rotation['angle'])
                results[i] = cached
            else:
                futures[i] = pool.submit(analyze, (i, str(source), str(args.output)), args.diagnostics)
        while 0 not in results and futures and not any(e['frame'] == 0 for e in errors):
            collect()
            time.sleep(.1)
        if 0 not in results:
            raise RuntimeError(f'First-frame detection failed: {errors}')
        if not args.prepare_only:
            if reference_path.exists():
                approval = load_reference(reference_path, results[0])
            else:
                configure_backend(args.backend)
                import matplotlib.pyplot as plt
                plt.rcParams.update({'text.usetex': False, 'font.family': 'DejaVu Sans'})
                def poll_editor():
                    collect()
                    editor.fig.suptitle(f'First-frame setup | {len(results)}/{len(files)} detections complete | {len(errors)} errors')
                    editor.fig.canvas.draw_idle()
                    return True
                from review_session import export_detections, export_detection_video
                def export_initial(nodes, fps):
                    collect()
                    return export_detections(list(results.values()), args.output, nodes, fps)
                def export_initial_video(nodes, fps):
                    collect()
                    return export_detection_video(list(results.values()), args.output, nodes, fps)
                editor = ReferenceEditor(results[0], reference_path, poll=poll_editor, fps=args.fps,
                                         output=args.output, keep_open=True, on_export=export_initial,
                                         on_export_video=export_initial_video)
                figure = editor.fig
                print('First-frame editor ready. Edits are human references immediately; Start tracking begins propagation.', flush=True)
                approval = editor.show()
                if approval is None:
                    print('Editor closed. Edits were saved; detections will finish without starting propagation.', flush=True)
        while futures:
            collect()
            if figure is not None:
                import matplotlib.pyplot as plt
                if plt.fignum_exists(figure.number):
                    figure.suptitle(f'Analyzing frames: {len(results)}/{len(files)} ready. First frame saved.')
                    figure.canvas.flush_events()
            time.sleep(.1)
        collect()
    if errors:
        raise RuntimeError(f'Detection failed: {errors}; see {manifest_path}')
    if approval is not None:
        from review_session import ReviewSession
        report_path = args.output/'tracking.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        expected_hash = hashlib.sha256(json.dumps(approval, sort_keys=True).encode()).hexdigest()
        if report is not None and (report['reference_sha256'] != expected_hash or len(report['frames']) != len(files)):
            report = None
        session = ReviewSession([results[i] for i in range(len(files))], approval, args.output,
                                report=report, fps=args.fps, method=args.method)
        from temporal_tracking import TRACKING_POLICY_VERSION
        if session.report is None or session.report.get('tracking_policy_version') != TRACKING_POLICY_VERSION:
            def progress(done, total):
                if figure is not None and (done % 10 == 0 or done == total):
                    figure.suptitle(f'Temporal tracking: {done}/{total} frames')
                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()
            session.reanalyze(progress=progress)
        from tracking_review import show_review
        if figure is not None:
            import matplotlib.pyplot as plt
            if not plt.fignum_exists(figure.number):
                return
        show_review(session.report, backend=args.backend, session=session, figure=figure)
    elif not args.prepare_only:
        raise SystemExit('Tracking has not started. Rerun to resume first-frame setup; cached detections and edits are reused.')


if __name__ == '__main__':
    main()
