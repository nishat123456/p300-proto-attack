"""Atomic, input-bound completion ledger for deterministic experiment settings."""
import json,math,os
from pathlib import Path

def setting_key(attack,budget,settings):
    return json.dumps([attack,budget,settings.get('detector'),settings.get('weight'),settings.get('steps'),settings.get('restarts')],separators=(',',':'))

def load_shard(path,metadata):
    path=Path(path)
    if not path.exists():return [],[]
    stored=json.loads(path.read_text())
    if any(stored.get(k)!=v for k,v in metadata.items()):raise ValueError('resume input/config/provenance mismatch')
    rows=stored['rows'];seen=set()
    for row in rows:
        key=setting_key(row['attack'],row['budget'],row['settings'])
        if key in seen:raise ValueError('duplicate completed setting')
        seen.add(key)
        if not math.isfinite(row['applied_linf']) or not 0<=row['applied_linf']<=row['raw_epsilon']:
            raise ValueError('invalid saved applied budget')
        if not all(math.isfinite(x) for x in row['scores'].values()):raise ValueError('nonfinite saved score')
    return rows,stored['traces']

def save_shard(path,metadata,rows,traces):
    path=Path(path);temporary=path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(dict(**metadata,rows=rows,traces=traces),handle,indent=2,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)
