"""Display existing cloud-generated development figures, no recomputation."""
from pathlib import Path
import hashlib,json
from IPython.display import Image,display,Markdown
root=Path('/kaggle/input');out=Path('/kaggle/working/out');out.mkdir()
files={}
display(Markdown('# DEVELOPMENT participant19 figure preview\nOne previously inspected user; not independent population results.'))
for name in ['clean_rejection.png','defended_manipulation.png']:
    paths=list(root.rglob(name))
    assert len(paths)==1,(name,len(paths))
    p=paths[0];data=p.read_bytes();assert data.startswith(b'\x89PNG\r\n\x1a\n')
    files[name]=hashlib.sha256(data).hexdigest()
    display(Markdown('## '+name))
    display(Image(data=data))
(out/'preview_provenance.json').write_text(json.dumps(dict(scope='existing development19 figures only; visual inspection pending',input_sha256=files),indent=2))
