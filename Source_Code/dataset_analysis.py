"""Analyze image files and write a JSON dataset report."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json
from PIL import Image, UnidentifiedImageError

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def analyze(root: str):
    root=Path(root); files=[p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    report={"root":str(root),"total_files":len(files),"classes":{},"corrupted":[],"duplicates":[]}
    hashes=defaultdict(list)
    for p in files:
        rel=p.relative_to(root); cls=rel.parts[0] if len(rel.parts)>1 else "UNKNOWN"
        item=report["classes"].setdefault(cls,{"count":0,"formats":Counter(),"dimensions":Counter(),"modes":Counter()})
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                item["count"]+=1; item["formats"][p.suffix.lower()]+=1; item["dimensions"][f"{im.width}x{im.height}"]+=1; item["modes"][im.mode]+=1
            hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(rel))
        except (UnidentifiedImageError,OSError,ValueError) as e: report["corrupted"].append({"file":str(rel),"error":str(e)})
    report["duplicates"]=[v for v in hashes.values() if len(v)>1]
    for c in report["classes"].values():
        for k in ("formats","dimensions","modes"): c[k]=dict(c[k])
    report["class_imbalance"]={k:v["count"] for k,v in report["classes"].items()}
    return report
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    r=analyze(a.root); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
