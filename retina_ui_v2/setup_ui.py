"""Install the pinned UI dependency while constraining every existing package.

Run inside the existing C:\\CP2\\.venv. No model/data access. Existing package
versions are constraints: a resolver conflict stops instead of upgrading them.
"""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import uuid

GRADIO_VERSION = '6.31.0'


def versions():
    return {re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version
            for d in metadata.distributions() if d.metadata.get('Name')}


def main():
    if sys.prefix == sys.base_prefix:
        print('Activate C:\\CP2\\.venv first: source .venv/Scripts/activate')
        return 1
    before = versions()
    core = ('torch','torchvision','torch-geometric','numpy','pillow','matplotlib')
    missing = [n for n in core if n not in before]
    if missing:
        print('This is not the completed CP2 environment. Missing:', ', '.join(missing))
        return 1
    if 'gradio' in before and before['gradio'] != GRADIO_VERSION:
        print('A different Gradio version is already installed:', before['gradio'])
        print('No changes made. Share this output before changing the environment.')
        return 1
    folder = (Path(__file__).resolve().parent.parent / 'environment/data_audit/retina_ui_v2' / ('setup_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]))
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'prototype_packages_before_v1.json').write_text(json.dumps(before,indent=2)+'\n')
    # Constraints preserve all installed versions, including shared image/math packages.
    with tempfile.TemporaryDirectory(prefix='cp2_gradio_install_') as temp:
        constraints = Path(temp)/'constraints.txt'
        constraints.write_text('\n'.join(f'{name}=={version}' for name,version in sorted(before.items()))+'\n')
        report = Path(temp)/'plan.json'
        base = [sys.executable,'-m','pip','install','--disable-pip-version-check',
                '--constraint',str(constraints),f'gradio=={GRADIO_VERSION}']
        print('Resolving Gradio while keeping all existing package versions…',flush=True)
        result = subprocess.run(base+['--dry-run','--report',str(report)])
        if result.returncode:
            print('SETUP STOPPED: resolver conflict; existing packages were not upgraded. Return this output.')
            return result.returncode
        # pip's JSON report is UTF-8, independent of the Windows code page.
        plan = json.loads(report.read_text(encoding='utf-8-sig'))
        for item in plan.get('install',[]):
            name = re.sub(r'[-_.]+','-',item['metadata']['name']).lower()
            if name in before and item['metadata']['version'] != before[name]:
                raise ValueError('Installer proposed changing an existing version: '+name)
        result = subprocess.run(base)
        after = versions()
        changed = {n:[v,after.get(n)] for n,v in before.items() if after.get(n)!=v}
        summary = dict(status='PASS' if result.returncode==0 and not changed else 'STOPPED',
                       python=platform.python_version(),gradio=after.get('gradio'),
                       protected_existing_packages=len(before),changed_existing_packages=changed,
                       added_packages={n:v for n,v in after.items() if n not in before},
                       package_install_returncode=result.returncode, model_access=False)
        (folder/'prototype_setup_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
        if result.returncode or changed:
            print('SETUP STOPPED. Return prototype_setup_summary_v1.json; no model work was started.')
            return 1
        print('PROTOTYPE SETUP: PASS | Existing package versions unchanged')
        print('Next: python retina_ui_v2/app.py --check')
        return 0


if __name__=='__main__':
    raise SystemExit(main())
