() => {
  if (window.__retinaUIv2) return;
  window.__retinaUIv2 = true;
  const q = (selector,root=document) => root.querySelector(selector);
  const all = (selector,root=document) => Array.from(root.querySelectorAll(selector));
  const stage = () => q("#ri-content [data-stage]");
  const terminals = {prepare:["ready","error"], analyse:["result","failed","error"], clear:["cleared","error"]};
  const toastSelector = '[data-testid="toast-body"].error';
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let pending=null, spin=null;
  function animateRing() {
    spin?.cancel(); spin=null;
    const ring=q("#ri-busy .ri-spinner");
    if(pending && ring && !reduced.matches)
      spin=ring.animate([{transform:"rotate(0turn)"},{transform:"rotate(1turn)"}],{duration:740,iterations:Infinity});
  }
  reduced.addEventListener("change",animateRing);
  function start(key,text,server=false) {
    if(pending?.server)return null; // The request owns the workspace until its new receipt or a real error.
    const op={key,server,before:stage()?.dataset.receipt,focus:document.activeElement,
              oldErrors:server?new Map(all(toastSelector).map(n=>[n,n.textContent])):null,settling:null};
    pending=op;
    const layer=q("#ri-busy");
    if(layer){
      layer.hidden=false; layer.dataset.action=key; q(".ri-busy-label",layer).textContent=text;
      if(layer.showPopover && !layer.matches(":popover-open"))layer.showPopover();
      animateRing();
    }
    if(server){
      q("#ri-body")?.setAttribute("aria-busy","true");
      queueMicrotask(()=>{if(pending===op)for(const n of [q("#ri-body"),q(".ri-head")])if(n)n.inert=true;});
    }
    return op;
  }
  function finish(op) {
    if(!op||pending!==op)return;
    pending=null; spin?.cancel();spin=null;
    const layer=q("#ri-busy");
    if(layer){
      if(layer.hidePopover && layer.matches(":popover-open"))layer.hidePopover();
      layer.hidden=true; delete layer.dataset.action;
    }
    if(op.server){
      for(const n of [q("#ri-body"),q(".ri-head")])if(n)n.inert=false;
      q("#ri-body")?.removeAttribute("aria-busy");
      if(document.activeElement===document.body && op.focus?.isConnected && !op.focus.disabled)
        op.focus.focus({preventScroll:true});
    }
  }
  const nextFrame = fn => document.hidden ? fn() : requestAnimationFrame(fn);
  // Paint the feedback once for synchronous controls; never wait a fixed number of milliseconds.
  const afterFeedback = fn => nextFrame(()=>nextFrame(fn));
  function decoded(roots,done) {
    const imgs=roots.filter(Boolean).flatMap(r=>all("img",r)).filter(i=>i.getClientRects().length&&!i.complete);
    if(!imgs.length){done();return;}
    Promise.allSettled(imgs.map(i=>i.decode())).then(done);
  }
  function local(key,label,apply,images=()=>[]) {
    const op=start(key,label); if(!op)return;
    afterFeedback(()=>{if(pending!==op)return;
      try{apply();decoded(images(),()=>finish(op));}catch(e){finish(op);throw e;}
    });
  }
  const seen=new WeakSet();
  function focusNewResult() {
    const box=q("#ri-content [data-stage=result]");
    if(!box||seen.has(box)||pending?.server)return;
    seen.add(box);
    if(matchMedia("(max-width:760px)").matches){
      box.scrollIntoView({block:"start",behavior:"instant"});
      q("[data-tab=overview]",box)?.focus({preventScroll:true});
    }
  }
  function sync() {
    const op=pending;
    if(!op?.server){focusNewResult();return;}
    const node=stage(), receipt=node?.dataset.receipt;
    if(receipt && receipt!==op.before && terminals[op.key].includes(node.dataset.stage)){
      if(op.settling===receipt)return;
      op.settling=receipt;
      // The terminal markup arrives in the same Gradio response as the preview and controls.
      nextFrame(()=>{if(pending!==op)return;
        decoded([node,q("#ri-preview")],()=>{if(pending!==op)return;finish(op);focusNewResult();});
      });return;
    }
    if(all(toastSelector).some(n=>!op.oldErrors.has(n)||op.oldErrors.get(n)!==n.textContent)){
      interrupted(op);
    }
  }
  function interrupted(op) {
    finish(op);
    const previous=stage();
    if(!previous)return;
    const message=document.createElement("h2"), detail=document.createElement("p"), retry=document.createElement("button");
    message.textContent="The request did not finish.";
    detail.textContent="Reload the workspace and choose the image again.";
    retry.type="button"; retry.className="ri-quiet";retry.dataset.reconnect="";retry.textContent="Reload workspace";
    previous.className="ri-panel";previous.dataset.stage="interrupted";delete previous.dataset.receipt;
    previous.setAttribute("role","alert");previous.replaceChildren(message,detail,retry);
    // After a transport failure the browser cannot vouch for the server's selected input.
    const side=q("#ri-side");if(side)side.inert=true;
    retry.focus({preventScroll:true});
  }
  new MutationObserver(sync).observe(document.body,{subtree:true,childList:true,characterData:true,
    attributes:true,attributeFilter:["data-receipt","data-stage","class"]});
  function activate(tab) {
    const box=tab.closest(".ri-results");
    if(tab.getAttribute("aria-selected")==="true")return;
    const pinned=box.getBoundingClientRect().top < (q(".ri-head")?.getBoundingClientRect().bottom ?? 0);
    for(const b of all("[data-tab]",box)){
      const active=b===tab;b.setAttribute("aria-selected",String(active));b.tabIndex=active?0:-1;
      q("#"+b.getAttribute("aria-controls"),box).hidden=!active;
    }
    if(pinned)box.scrollIntoView({block:"start",behavior:"instant"});
  }
  function tabAction(t,focus=false) {
    local("tab","Opening result…",()=>{activate(t);if(focus)t.focus({preventScroll:true});},
          ()=>[q("#"+t.getAttribute("aria-controls"))]);
  }
  function closeAbout() {
    local("about-close","Returning to workspace…",()=>{
      q("#ri-about")?.close();q("[data-about]")?.focus({preventScroll:true});
    });
  }
  document.addEventListener("click",e=>{
    const t=e.target.closest?.("button");
    if(!t||t.disabled||t.closest(".ri-system"))return;
    if(pending?.server && t.closest("#ri-body,#ri-top")){
      e.preventDefault();e.stopImmediatePropagation();return;
    }
    if(t.matches("#ri-analyse")){start("analyse","Analysing image…",true);return;}
    if(t.matches("#ri-clear")){start("clear","Clearing workspace…",true);return;}
    if(t.matches("#ri-sample-buttons button")){start("prepare","Loading sample…",true);return;}
    // Keep the native chooser inside the trusted click. Selection starts a separate upload operation.
    if(t.matches("#ri-upload")){
      const op=start("picker","Opening file picker…");afterFeedback(()=>finish(op));return;
    }
    if(t.matches("#ri-samples .label-wrap")){
      const op=start("samples","Updating samples…");afterFeedback(()=>finish(op));return;
    }
    if(t.matches("[data-about]")){
      local("about","Opening model notes…",()=>{
        const dialog=q("#ri-about");if(!dialog||dialog.open)return;
        dialog.showModal();const heading=q("#ri-about-title",dialog);
        heading.tabIndex=-1;heading.focus({preventScroll:true});dialog.scrollTop=0;
      });return;
    }
    if(t.matches("[data-close-about]")){closeAbout();return;}
    if(t.matches("[data-tab]")){tabAction(t);return;}
    if(t.matches("[data-fov]")){
      const box=t.closest(".ri-results"),key=t.dataset.fov;
      local("display","Updating display…",()=>{
        for(const b of all("[data-fov]",box))b.setAttribute("aria-pressed",String(b===t));
        for(const v of all("[data-map]",box))v.hidden=v.dataset.map!==key;
      },()=>[q('[data-map="'+key+'"]',box)]);return;
    }
    if(t.matches("[data-reconnect]")){
      const op=start("reload","Reloading workspace…");
      afterFeedback(()=>{if(pending===op){try{location.reload();}catch(e){finish(op);throw e;}}});
      // This action ends at pagehide; the fresh document starts with an idle workspace.
    }
  },true);
  document.addEventListener("cancel",e=>{
    if(e.target?.id==="ri-about"){e.preventDefault();closeAbout();}
    else if(e.target?.type==="file" && pending?.key==="picker")finish(pending);
  },true);
  document.addEventListener("keydown",e=>{
    const t=e.target.closest?.("[data-tab]");if(!t||pending?.server)return;
    const list=all("[data-tab]",t.closest(".ri-tabs"));let i=list.indexOf(t);
    if(e.key==="ArrowRight")i=(i+1)%list.length;
    else if(e.key==="ArrowLeft")i=(i+list.length-1)%list.length;
    else if(e.key==="Home")i=0;else if(e.key==="End")i=list.length-1;else return;
    e.preventDefault();tabAction(list[i],true);
  });
  document.addEventListener("change",e=>{
    const input=e.target;
    if(input.type!=="file"||!input.closest?.("#ri-side"))return;
    const f=input.files?.[0];
    if(!f){if(pending?.key==="picker")finish(pending);return;}
    if(pending?.server){e.stopImmediatePropagation();return;}
    start("prepare","Checking image…",true);
    const kind=!f.size?"empty":f.size>20*1024*1024?"size":!(/\.(jpe?g|png)$/i.test(f.name))?"type":"";
    const prev=q("#ri-content [data-stage=result]");
    if(prev){prev.className="ri-panel";prev.dataset.stage="replacing";delete prev.dataset.receipt;prev.textContent="Checking the replacement image…";}
    if(kind){
      e.stopImmediatePropagation();e.preventDefault();input.value="";
      q("#ri-reject-"+kind)?.click();
    }
  },true);
  window.addEventListener("pagehide",()=>finish(pending));
}
