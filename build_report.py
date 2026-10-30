"""Build a detailed landscape A3 report from batch artifacts, without image inspection."""
import argparse
import json
import textwrap
from io import BytesIO
from PIL import Image
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3,landscape
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader


def build(root,destination):
    manifest=json.loads((root/'manifest.json').read_text())
    results=manifest['results'];errors=manifest['errors']
    destination.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(destination),pagesize=landscape(A3),pageCompression=1)
    c.setTitle('GridTracker - ten-frame pipeline inspection report')
    c.setAuthor('GridTracker')
    width,height=landscape(A3);page=0;boxes=[]
    def begin(title,subtitle=''):
        nonlocal page
        page+=1;c.setFillColorRGB(.07,.12,.19);c.rect(0,height-66,width,66,fill=1,stroke=0)
        c.setFillColorRGB(1,1,1);c.setFont('Helvetica-Bold',19);c.drawString(34,height-30,title)
        c.setFont('Helvetica',10);c.drawString(34,height-49,subtitle[:175])
        c.setFillColorRGB(.25,.3,.36);c.setFont('Helvetica',9)
        c.drawString(34,20,'GridTracker | 28 October 2026 | numerical and structural verification; no visual image review')
        c.drawRightString(width-34,20,str(page))
    def paragraph(text,x,y,size=13,columns=120):
        c.setFillColorRGB(.1,.13,.18);c.setFont('Helvetica',size)
        for line in textwrap.wrap(text,columns):c.drawString(x,y,line);y-=size*1.5
        return y-12
    begin('GridTracker: complete detection and fitting process','One single-threaded worker per independent frame; parallel frame execution')
    c.bookmarkPage('overview');c.addOutlineEntry('Pipeline overview','overview',level=0)
    y=height-108
    for text in [
        f"{len(results)} completed frames; {len(errors)} errors. Worker processes: {manifest['workers']}; numeric threads per worker: 1.",
        'Order: color-neutral corner seed -> measured width and learned color scalar -> four-arm crossing score -> automatic score threshold -> geometric component centers -> lower pixel-mass cutoff -> preliminary maxima -> completed edge nodes -> final maxima cleanup -> within-group two-SD rejection -> unpenalized cubic Bezier curves.',
        'The two-SD rule uses every local minimum of each post-cleanup scan to define groups. Mean and population SD are computed once per eligible coordinate group. A node is rejected when its absolute deviation is at least two SD in either eligible direction. Zero-variance groups have undefined z-scores and do not cause rejection.',
        'Every curve has four evenly spaced longitudinal controls, with four transverse coordinates fitted jointly. Bending penalty is zero. Soft-L1 data residuals reduce the influence of remaining outliers. Fewer than four distinct longitudinal positions means the curve is omitted and recorded.',
        'Edge proposals use a general histogram threshold because a shallow-slope interval is not always present on the test frames. Unsupported endpoints are explicitly recorded. No temporal fitting, Fourier grid, isolated-neighbor rejection, upper mass cutoff, or score-weighted mass filter is used in this final pipeline. Frames are independent. No accuracy claim is made without hand-labelled string locations.',
        'Reading the report: white points are crossing nodes, yellow points are edge endpoints, red crosses identify a rejection at the indicated stage. Cyan curves are horizontal and magenta curves vertical. Image brightness is increased threefold for display only; width inspection uses sixfold brightness.',
        'The report includes every generated diagnostic plot. Machine-readable results, coordinates, thresholds, widths, grouping statistics and curve controls remain in the per-frame folders beside the manifest.'
    ]:y=paragraph(text,40,y)
    c.showPage()
    begin('Batch summary','Files ordered by download/birth timestamp. Missing or uncertain detections remain visible in each frame chapter.')
    c.bookmarkPage('summary');c.addOutlineEntry('Batch summary','summary',level=0)
    headings=['Frame','Width px','Threshold','Components','Mass-kept','Edges','Post-max','2SD removed','Final nodes','Curves','Omitted']
    columns=[35,105,190,290,400,500,580,680,800,910,1010]
    y=height-105;c.setFont('Helvetica-Bold',11)
    for x,t in zip(columns,headings):c.drawString(x,y,t)
    y-=29
    for r in results:
        values=[r['frame'],f"{r['width']:.2f}",f"{r['threshold']:.4f}",r['components'],r['mass_kept'],r['edge_nodes'],r['post_maxima'],r['removed_2sd'],len(r['final_nodes']),len(r['curves']),len(r['ignored_curves'])]
        c.setFont('Helvetica',11)
        for x,t in zip(columns,values):c.drawString(x,y,str(t))
        y-=28
    y-=35
    for r in results:y=paragraph(f"Frame {r['frame']:02d}: {Path(r['source']).name}",40,y,11)
    c.showPage()
    for r in results:
        begin(f"Frame {r['frame']:02d}: pipeline summary",Path(r['source']).name)
        chapter=f"frame_{r['frame']}"
        c.bookmarkPage(chapter);c.addOutlineEntry(f"Frame {r['frame']:02d}",chapter,level=0)
        y=height-110
        for text in [
            f"Seed: {r['seed']['corner']}, {r['seed']['n']} x {r['seed']['n']} crossings. Width: {r['width']:.3f} px from {r['width_profiles']} accepted profiles. Crossing threshold: {r['threshold']:.6f}.",
            f"Unsupported edge endpoints: {len(r.get('unsupported_endpoints',[]))}. Connected regions: {r['components']}; after lower mass cutoff: {r['mass_kept']}. Edge proposals: {r['edge_proposals']}; completed edge endpoints: {r['edge_nodes']}.",
            f"After maxima cleanup: {r['post_maxima']}. Local two-SD rejection removes {r['removed_2sd']} nodes; {len(r['final_nodes'])} remain. Curves fitted: {len(r['curves'])}; under-supported curves omitted: {len(r['ignored_curves'])}.",
            'Omitted curves: '+json.dumps(r['ignored_curves']),
            'Maximum control-fit node RMSE: '+(f"{max(q['rmse'] for q in r['curves']):.3f} px." if r['curves'] else 'no curves fitted.'),
            'Two-SD filter is a single pass; statistics are not recomputed after removal. A large z-score can also represent a legitimate curved string, so inspect the before/after panels before treating detections as ground truth.'
        ]:y=paragraph(text,40,y)
        c.showPage()
        for i,a in enumerate(r['artifacts']):
            begin(f"Frame {r['frame']:02d} / {i+1:02d}: {a['title']}",Path(r['source']).name)
            bookmark=f"frame_{r['frame']}_stage_{i}"
            c.bookmarkPage(bookmark);c.addOutlineEntry(a['title'],bookmark,level=1)
            # Keep lossless PNGs on disk; compress report previews without resizing.
            buffer=BytesIO()
            with Image.open(a['path']) as original:
                original.convert('RGB').save(buffer,format='JPEG',quality=92,subsampling=0)
            buffer.seek(0)
            image=ImageReader(buffer);iw,ih=image.getSize()
            available=(width-76,height-162);scale=min(available[0]/iw,available[1]/ih)
            w,h=iw*scale,ih*scale;x=(width-w)/2;y=83+(available[1]-h)/2
            c.drawImage(image,x,y,width=w,height=h,preserveAspectRatio=True)
            boxes.append([page,x,y,w,h])
            paragraph(a['caption'],38,66,10,185)
            c.showPage()
    for error in errors:
        begin(f"Frame {error['frame']} failed",error['source'])
        paragraph(error['traceback'],40,height-105,11);c.showPage()
    c.save()
    reader=PdfReader(destination)
    if len(reader.pages)!=page:raise ValueError('Unexpected PDF page count')
    for p in reader.pages:
        if not p.extract_text().strip():raise ValueError('PDF page has no expected headings')
    if not all(x>=0 and y>=0 and x+w<=width and y+h<=height for _,x,y,w,h in boxes):raise ValueError('Figure extends outside page')
    (destination.with_suffix('.validation.json')).write_text(json.dumps({'pages':page,'figures':len(boxes),'image_boxes_inside_page':True,'text_on_every_page':True,'visual_inspection':False},indent=2)+'\n')
    print(f'Validated {page} pages, {len(boxes)} figures: {destination}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/final_batch'));p.add_argument('--output',type=Path,default=Path('output/pdf/gridtracker_process_report.pdf'));a=p.parse_args();build(a.root,a.output)
