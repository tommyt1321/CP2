"""Retina Insight: local Gradio workbench for the frozen CP2 A3 model."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from _baseline import prototype_core_v1 as core
from cache import Cache
import evidence
import views


@dataclass
class Selection:
    prepared: core.Prepared
    name: str
    source: str


def sample_paths(root):
    key = Path("A. Segmentation/1. Original Images/b. Testing Set/IDRiD_61.jpg")
    raw = root / "IDRiD" / "raw"
    idrid = next((p for p in [raw/key, *sorted(raw.glob("*/"+key.as_posix()))] if p.is_file()), None)
    aptos = root / "train_images" / "000c1434d8d7.png"
    return {"idrid": idrid, "aptos": aptos if aptos.is_file() else None}


def build_ui(engine, cache_path, samples):
    import gradio as gr
    if gr.__version__ != core.GRADIO_VERSION:
        raise RuntimeError("Use the project's pinned Gradio version: "+core.GRADIO_VERSION)
    with gr.Blocks(title="Retina Insight | Analysis workspace", analytics_enabled=False, delete_cache=(60,900)) as app:
        selected = gr.State(None, time_to_live=900)
        gr.HTML(views.HEADER, elem_id="ri-top", container=False)
        with gr.Column(elem_id="ri-body"):
            gr.HTML(views.INTRO, container=False)
            with gr.Row(elem_id="ri-layout"):
                with gr.Column(elem_id="ri-side", min_width=0):
                    gr.HTML(views.INPUT_TITLE, container=False)
                    preview = gr.HTML(views.input_view(None), container=False, elem_id="ri-preview")
                    upload = gr.UploadButton("Choose image", file_count="single", type="filepath",
                                             file_types=[".jpg",".jpeg",".png"], elem_id="ri-upload")
                    gr.HTML(views.INPUT_NOTE,container=False)
                    with gr.Accordion("Use a sample", open=False, elem_id="ri-samples"):
                        gr.HTML(views.SAMPLE_NOTE,container=False)
                        with gr.Row(elem_id="ri-sample-buttons"):
                            id_b=gr.Button("IDRiD sample", interactive=bool(samples["idrid"]))
                            apt_b=gr.Button("APTOS sample", interactive=bool(samples["aptos"]))
                        missing=[name for key,name in (("idrid","IDRiD"),("aptos","APTOS")) if not samples[key]]
                        if missing:
                            gr.HTML('<p class="ri-fine">Local original unavailable: '+', '.join(missing)+'. You can still upload an image.</p>',container=False)
                    analyse=gr.Button("Analyse image", variant="primary",interactive=False,elem_id="ri-analyse")
                    clear=gr.Button("Clear image and results",interactive=False,elem_id="ri-clear")
                    note=gr.HTML(views.status("Waiting for an image"),container=False,elem_id="ri-status")
                with gr.Column(elem_id="ri-right", min_width=0):
                    content=gr.HTML(views.intro_panel(),container=False,elem_id="ri-content")
        gr.HTML(views.FOOTER, container=False, elem_id="ri-bottom")
        # Real private events for pre-upload validation. The browser can reject an oversize
        # replacement before Gradio uploads a byte; server validation remains authoritative.
        badtype = gr.Button("Reject unsupported file",elem_id="ri-reject-type",elem_classes=["ri-system"])
        badsize = gr.Button("Reject oversized file",elem_id="ri-reject-size",elem_classes=["ri-system"])
        badempty = gr.Button("Reject empty file",elem_id="ri-reject-empty",elem_classes=["ri-system"])
        outputs=[selected,preview,content,note,upload,id_b,apt_b,analyse,clear]

        def controls(obj, busy=False, can_clear=None):
            return [gr.update(value=None,interactive=not busy),
                    gr.update(interactive=bool(samples["idrid"]) and not busy),
                    gr.update(interactive=bool(samples["aptos"]) and not busy),
                    gr.update(interactive=obj is not None and not busy),
                    gr.update(interactive=(obj is not None if can_clear is None else can_clear) and not busy)]

        def empty(title="Start with one retinal image.", detail="Choose a photograph, then select Analyse image.", stage="empty"):
            label={"empty":"Waiting for an image","cleared":"Workspace cleared","error":"Image not accepted"}[stage]
            return [None,views.input_view(None),views.intro_panel(title,detail,stage),
                    views.status(label,detail if stage=="error" else "", "error" if stage=="error" else "normal"),
                    *controls(None,can_clear=stage=="error")]

        def reject(code):
            title,detail={
                "type":("Unsupported format","Choose an original JPEG or PNG fundus photograph."),
                "size":("File too large","The file exceeds 20 MiB. Choose a smaller original JPEG or PNG."),
                "empty":("Empty file","Choose a non-empty original JPEG or PNG."),
            }[code]
            return empty(title, detail, "error")

        def prepare(path,source,example_key=None):
            yield [None,views.input_view(None),views.intro_panel("Checking image…","Reading the photograph and preparing the model input.","preparing"),
                   views.status("Checking image…"),*controls(None,busy=True)]
            try:
                if not path:
                    raise core.InputError("Choose an available image and try again.")
                p=Path(path).resolve()
                if example_key is None and not p.is_relative_to(Path(cache_path).resolve()):
                    raise core.InputError("Choose an image using the upload button.")
                if example_key=="idrid" and core.sha(p)!="17b12f5cafc224a29230eb1fd0df9d7e0c404df931982ce7db084119ff3094a3":
                    raise core.InputError("The local IDRiD sample does not match the reviewed original. Choose a file instead.")
                a=core.prepare(p,engine.preprocess)
                if example_key=="aptos" and hashlib.sha256(a.rgb.tobytes()).hexdigest() != "f7cd815810e0b19c4ab27ed08ef6648c6025aa481708dda88c6669c7e17b39f7":
                    raise core.InputError("The APTOS sample does not match the reviewed model input. Choose a file instead.")
                s=Selection(a,p.name,source)
                detail="This is the original photograph. Select Analyse image to run frozen A3."
                if a.warning: detail+=" The crop used a fallback; check the original."
                yield [s,views.input_view(s),views.intro_panel("Ready to analyse.",detail,"ready"),views.status("Image ready", "Crop fallback used." if a.warning else ""),*controls(s)]
            except core.InputError as exc:
                yield empty("Image not accepted",str(exc),"error")
            except Exception as exc:
                print("Input preparation failed:",type(exc).__name__,flush=True)
                yield empty("Image could not be prepared","Choose another original JPEG or PNG and try again.","error")

        def upload_file(path):
            yield from prepare(path,"Uploaded photograph")
        def id_sample():
            yield from prepare(samples["idrid"],"IDRiD segmentation sample","idrid")
        def apt_sample():
            yield from prepare(samples["aptos"],"APTOS training sample","aptos")

        def analyse_image(s):
            if s is None:
                yield empty("Choose the image again","The session input is unavailable or expired.","error")
                return
            yield [gr.skip(),gr.skip(),views.intro_panel("Analysing image…","Computing the grade, regional relevance and signed region relationships.","busy"),
                   views.status("Analysis in progress"),*controls(s,busy=True)]
            try:
                r, rec=evidence.run(engine,s.prepared)
                field,why=evidence.field_copy(s.prepared.rgb,r["overlay"])
                html=views.results(s,r,rec,field,why)
                yield [gr.skip(),gr.skip(),html,views.status("Analysis complete"),*controls(s)]
            except Exception as error:
                print("Analysis failed:",type(error).__name__,flush=True)
                yield [gr.skip(),gr.skip(),views.intro_panel("Analysis could not finish.","No prediction is displayed. Try Analyse image again or choose another photograph.","failed"),
                       views.status("Analysis failed","You can retry or clear the workspace.","error"),*controls(s)]

        opts=dict(concurrency_id="ri-local-a3",concurrency_limit=1,trigger_mode="once",api_visibility="private",show_progress="hidden")
        upload.upload(upload_file,upload,outputs,**opts)
        id_b.click(id_sample,None,outputs,**opts); apt_b.click(apt_sample,None,outputs,**opts)
        analyse.click(analyse_image,selected,outputs,**opts)
        clear.click(lambda:empty("Workspace cleared.","Choose another photograph to start again.","cleared"),None,outputs,**opts)
        badtype.click(lambda:reject("type"),None,outputs,**opts)
        badsize.click(lambda:reject("size"),None,outputs,**opts)
        badempty.click(lambda:reject("empty"),None,outputs,**opts)
    return app.queue(default_concurrency_limit=1,max_size=8)


def serve(engine, root, port, cache, **kw):
    os.environ["GRADIO_TEMP_DIR"]=str(cache.path)
    import gradio as gr
    ui=build_ui(engine,cache.path,sample_paths(root))
    try:
        ui.launch(server_name="127.0.0.1",server_port=port,share=False,inbrowser=False,
                  allowed_paths=[],blocked_paths=[str(root.resolve())],max_file_size=core.MAX_BYTES,
                  show_error=False,max_threads=4,state_session_capacity=8,enable_monitoring=False,
                  run_history=False,footer_links=[],mcp_server=False,strict_cors=True,
                  theme=gr.themes.Base(font=["Arial","sans-serif"],font_mono=["monospace"]),
                  css=(HERE/"design.css").read_text(encoding="utf-8"),
                  js=(HERE/"interface.js").read_text(encoding="utf-8"),**kw)
        return ui
    except BaseException:
        ui.close()
        raise


def save_check(root,kind,report):
    folder=root/"environment/data_audit/retina_ui_v2"; folder.mkdir(parents=True,exist_ok=True)
    time=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest=folder/f"{kind}_{time}_{uuid.uuid4().hex[:8]}.json"
    with dest.open("x",encoding="utf-8") as f: json.dump(report,f,indent=2,allow_nan=False)
    print("Saved:",dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--check",action="store_true",help="Check the UI, environment and frozen checkpoint; decode no images.")
    group.add_argument("--verify",action="store_true",help="Compare one fixed original with canonical pixels and direct A3 output.")
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--port",type=int,default=7860)
    args=parser.parse_args(); root=args.root.resolve()
    if not (root/"train_v1.py").is_file():
        parser.exit(2,"Place the retina_ui_v2 folder directly in your CP2 project, or specify --root.\n")
    # Import the local project, retaining this package as the first path for UI modules.
    sys.path.insert(1,str(root))
    cache=None; ui=None
    try:
        try: version=importlib.metadata.version("gradio")
        except importlib.metadata.PackageNotFoundError: version="not installed"
        if version!=core.GRADIO_VERSION:
            raise RuntimeError(f"Gradio {core.GRADIO_VERSION} is required (found: {version}). From the CP2 root, run python retina_ui_v2/setup_ui.py in the CP2 virtual environment.")
        print("Checking the frozen A3 checkpoint and project records…",flush=True)
        engine=evidence.attach(core.load_engine(root))
        if args.check:
            cache=Cache(Path(tempfile.gettempdir())/"retina_insight_ui_v2")
            os.environ["GRADIO_TEMP_DIR"]=str(cache.path)
            ui=build_ui(engine,cache.path,sample_paths(root))
            for file in ("design.css","interface.js"):
                if not (HERE/file).read_text(encoding="utf-8").strip(): raise ValueError(file+" is empty.")
            info=dict(engine.startup,gradio=version,status="RETINA_UI_V2_INPUT_CHECK_PASS",samples_available={k:bool(v) for k,v in sample_paths(root).items()})
            save_check(root,"input_check",info)
            print("RETINA UI V2 INPUT CHECK: PASS | A3 epoch 21 | Images decoded: 0")
        elif args.verify:
            save_check(root,"local_verify",core.verify_one_training_image(engine))
            print("RETINA UI V2 LOCAL VERIFY: PASS | Canonical pixels, direct probabilities and predicted grade agree.")
        else:
            cache=Cache(Path(tempfile.gettempdir())/"retina_insight_ui_v2")
            ui=serve(engine,root,args.port,cache)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"RETINA UI V2 STOPPED | {type(e).__name__}: {e}",flush=True);return 1
    finally:
        if ui is not None: ui.close()
        if cache is not None: cache.close()


if __name__=="__main__":
    raise SystemExit(main())
