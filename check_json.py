import json
import sys

data = json.load(open('val.json'))
with open('json_structure.txt', 'w') as f:
    if isinstance(data, dict):
        f.write(f"DICT with keys: {list(data.keys())[:10]}\n")
        for k in list(data.keys())[:3]:
            v = data[k]
            if isinstance(v, list):
                f.write(f"  {k}: list len={len(v)}\n")
                if v:
                    f.write(f"    first item: {json.dumps(v[0])[:500]}\n")
            else:
                f.write(f"  {k}: {type(v).__name__}\n")
    else:
        f.write(f"LIST len: {len(data)}\n")
        f.write(f"first: {json.dumps(data[0])[:500]}\n")
