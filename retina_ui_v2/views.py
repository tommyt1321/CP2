"""English presentation of one input and its own outputs."""
from html import escape
import math
from uuid import uuid4
from _baseline import prototype_core_v1 as core

HEADER = """<header class="ri-head"><div class="ri-head-inner"><div class="ri-brand"><b aria-hidden="true">ri</b><span>Retina Insight<small>EXPLAINABLE DR</small></span></div><button type="button" class="ri-quiet" data-about>About this model</button></div></header>
<dialog id="ri-about" aria-labelledby="ri-about-title"><div class="ri-dialog"><p class="ri-kicker">MODEL NOTES</p><h2 id="ri-about-title">About this model</h2><p>A3 is a frozen research model for five diabetic retinopathy grades. It uses DenseNet121 features, boundary-aware superpixels, a graph attention network and a vision transformer.</p><p>Attention Rollout shows relative relevance. Graph links show local sensitivities for the predicted grade. Neither identifies a confirmed lesion. Camera, framing and image quality may affect the output.</p><p>Use one original fundus photograph. File checks do not establish whether an image is clinically gradable. Clear removes the visible result and session input. Temporary upload files expire separately.</p><button type="button" class="ri-quiet" data-close-about>Back to workspace</button></div></dialog>"""
HEADER += """<div id="ri-busy" popover="manual" hidden role="status" aria-live="polite" aria-atomic="true"><div class="ri-busy-centre"><span class="ri-spinner" aria-hidden="true"></span><span class="ri-busy-label"></span></div></div>"""
FOOTER = '<footer class="ri-foot"><div><span>Research use only</span><span>CP2 · Retina Insight</span></div></footer>'
INTRO = '<div class="ri-intro"><p class="ri-kicker">ANALYSIS WORKSPACE</p><h1>Understand the prediction.</h1><p>Explore a DR grade, the regions highlighted by A3 and the connections between them.</p></div>'
INPUT_TITLE = '<div><p class="ri-kicker">01 / INPUT IMAGE</p><h2>Retinal photograph</h2></div>'
SAMPLE_NOTE = '<p class="ri-fine">Optional local samples. APTOS is a training example; samples use the same live analysis as uploads.</p>'
INPUT_NOTE = '<p class="ri-fine">Original JPEG or PNG · up to 20 MiB and 25 megapixels. Please use one upright fundus photograph.</p>'
STEPS = (("Prediction", "The DR grade and five model probabilities."), ("Regional relevance", "Where attention accumulates across the image."), ("Region relationships", "How neighbouring regions relate to the predicted grade."))


def intro_panel(title="Start with one retinal image.", message="Choose a photograph, then select Analyse image.", mode="empty"):
    cards = ''.join(f'<article class="ri-step"><span>0{i}</span><h3>{name}</h3><p>{desc}</p></article>' for i,(name,desc) in enumerate(STEPS,1))
    receipt = f' data-receipt="{uuid4().hex}"' if mode in ("ready","error","failed","cleared") else ""
    return (f'<section class="ri-panel" data-stage="{mode}"{receipt} aria-live="polite"><p class="ri-kicker">02 / WORKSPACE</p><h2>{escape(title)}</h2>'
            f'<p class="ri-description">{escape(message)}</p><div class="ri-grid3">{cards}</div></section>')


def photo(image, alt):
    return core.image_html(image, alt)


def figure(image, alt, caption):
    return f'<figure>{photo(image,alt)}<figcaption>{escape(caption)}</figcaption></figure>'


def input_view(selection):
    if selection is None:
        return '<div class="ri-drop"><span aria-hidden="true">↑</span><strong>Choose one fundus image</strong><p>JPEG or PNG</p></div>'
    item = selection.prepared
    return (photo(item.preview, 'Original fundus photograph') +
            f'<p class="ri-name">{escape(selection.name)}</p><p class="ri-fine">{escape(selection.source)} · {int(item.original_size[0])} × {int(item.original_size[1])} px</p>')


def pairs_html(items, sign):
    if items is None:
        return '<p class="ri-fine">A numeric pair list was not returned.</p>'
    if not items:
        return '<p class="ri-fine">No neighbouring pairs had this sign for this image.</p>'
    rows=[]
    for rank,item in enumerate(items[:5],1):
        a,b=map(int,item["regions"]); val=float(item["signed_score"])
        if not math.isfinite(val) or sign*val<=0:
            raise ValueError("Unexpected signed relationship record.")
        rows.append(f'<tr><td>{rank}</td><td>{a} ↔ {b}</td><td>{val:+.4f}</td></tr>')
    return '<table class="ri-pairs"><thead><tr><th>Rank</th><th>Regions</th><th>Logit / gate</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def results(selection, result, record, field, reason):
    grade=int(result['prediction']); probs=list(map(float,result['probabilities']))
    if len(probs)!=5 or grade not in range(5) or any(not math.isfinite(v) or not 0<=v<=1 for v in probs) or abs(sum(probs)-1)>1e-5 or probs.index(max(probs))!=grade:
        raise ValueError("Invalid five-class prediction.")
    short_names=('No apparent DR','Mild NPDR','Moderate NPDR','Severe NPDR','Proliferative DR')
    bars=[]
    for i,(name,v) in enumerate(zip(core.CLASSES,probs)):
        bars.append(f'<li class="ri-prob"><div><span><span class="ri-long">{i} · {escape(name)}</span><span class="ri-short" title="{escape(name)}">{i} · {short_names[i]}</span></span><b class="ri-unit">{100*v:.2f}%</b></div><div class="ri-track"><i style="width:{100*v:.6f}%"></i></div></li>')
    tabs=('Overview','Regional relevance','Region relationships'); keys=('overview','relevance','relations')
    compact=('Overview','Relevance','Relationships')
    nav=''.join(f'<button type="button" role="tab" id="rt-{k}" aria-label="{s}" aria-selected="{str(n==0).lower()}" aria-controls="rp-{k}" tabindex="{0 if n==0 else -1}" data-tab="{k}"><span class="ri-tab-full" aria-hidden="true">{s}</span><span class="ri-tab-compact" aria-hidden="true">{short}</span></button>' for n,(k,s,short) in enumerate(zip(keys,tabs,compact)))
    def panel(k,html,first=False):
        return f'<section role="tabpanel" id="rp-{k}" aria-labelledby="rt-{k}"{"" if first else " hidden"}>{html}</section>'
    data=result['metrics']
    overview=(f'<p class="ri-kicker">A3 / PREDICTION</p><h2>Prediction overview</h2><div class="ri-two"><article class="ri-stat"><p>Predicted DR grade</p><strong class="ri-grade">Grade {grade}</strong><h3><span class="ri-choice-long">{escape(core.CLASSES[grade])}</span><span class="ri-choice-short" title="{escape(core.CLASSES[grade])}">{short_names[grade]}</span></h3></article><article class="ri-probcard"><h3>Class probabilities</h3><ul class="ri-bars">{"".join(bars)}</ul><p class="ri-abbr">NPDR = non-proliferative diabetic retinopathy.</p></article></div>'
              '<p class="ri-callout">These probabilities describe this model’s output for this photograph. They are not calibrated clinical confidence.</p>')
    disabled=' disabled aria-describedby="ri-field-note"' if field is None else ''
    fieldimage=('<div data-map="field" hidden>'+photo(field,'Attention Rollout inside the estimated retinal field')+'</div>') if field is not None else ''
    fieldnote=(reason+' Full canvas remains available.') if field is None else 'Retinal field only hides the estimated background in the overlay. The input, grade and region rankings stay the same.'
    roll=('<p class="ri-kicker">A3 / ATTENTION ROLLOUT</p><h2>Regional relevance</h2><p>Relative relevance across all regions. Brighter colours show higher values within this image.</p>'
          f'<div class="ri-switch" role="group" aria-label="Relevance display"><button type="button" data-fov="full" aria-pressed="true">Full canvas</button><button type="button" data-fov="field" aria-pressed="false"{disabled}>Retinal field only</button></div>'
          f'<div class="ri-two">{figure(selection.prepared.rgb,"Unchanged 384 by 384 model input","Model input · full canvas")}<figure><div data-map="full">{photo(result["overlay"],"Full canvas Attention Rollout")}</div>{fieldimage}<figcaption>Relative relevance · dark purple to bright yellow</figcaption></figure></div>'
          f'<p id="ri-field-note" class="ri-callout">{escape(fieldnote)} Class-agnostic; this is not a lesion map.</p>')
    signs=[]
    for key,title,sgn,desc in [('positive','Positive sensitivity',1,'Orange · increasing the gate locally raises the target logit.'),('negative','Negative sensitivity',-1,'Cyan · increasing the gate locally lowers the target logit.')]:
        val=result.get(key); pairs=record.get('top_'+key)
        art=figure(val,title+' of neighbouring regions',desc) if val is not None else '<p>No relationship overlay was returned for this sign.</p>'
        signs.append(f'<article><h3>{title}</h3>{art}{pairs_html(pairs,sgn)}</article>')
    relation=(f'<p class="ri-kicker">A3 / GRAPH ATTENTION</p><h2>Region relationships</h2><p>Neighbouring regions are ranked separately by sign for the grade {grade} model score before softmax.</p>'
              f'<div class="ri-two">{"".join(signs)}</div><p class="ri-callout">Up to five connections per sign. These are local model sensitivities, not blood vessels or confirmed lesion pairs.</p>')
    return (f'<div class="ri-results" data-stage="result" data-receipt="{uuid4().hex}"><nav class="ri-tabs" role="tablist" aria-label="Analysis results">{nav}</nav><div class="ri-panel">'
            +panel('overview',overview,True)+panel('relevance',roll)+panel('relations',relation)
            +f'</div><p class="ri-fine ri-meta">{int(data["regions"])} regions · {float(data["backend_compute_seconds"]):.1f} s backend time</p></div>')


def status(title, detail="", kind="normal"):
    return f'<div class="ri-status {kind}" role="status"><strong>{escape(title)}</strong>{("<p>"+escape(detail)+"</p>") if detail else ""}</div>'
