import json,hashlib
from pathlib import Path
root=Path('/kaggle/input');out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
def unique(name):
    paths=list(root.rglob(name));assert len(paths)==1,(name,len(paths))
    return paths[0]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proofs=[]
for seed in (20261013,20261014,20261015):
    model=unique(f'source_seed{seed}.pt');ref=unique(f'references_seed{seed}.npz');template=unique(f'source_template_seed{seed}.pt');proof=json.loads(unique(f'references_seed{seed}.json').read_text())
    assert proof['checkpoint_sha256']==digest(model)
    assert proof['artifact_sha256']==digest(ref)
    assert proof['source_template']['artifact_sha256']==digest(template)
    proofs.append(dict(seed=seed,checkpoint_sha256=digest(model),reference_sha256=digest(ref),template_sha256=digest(template)))
(out/'artifact_access.json').write_text(json.dumps(dict(scope='private cloud mount/hash verification only; no EEG processing, outcome or independent unlock',proofs=proofs),indent=2))
print('All three source-model/reference/template mounts verified')
