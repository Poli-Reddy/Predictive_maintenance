from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parents[1]
files=[]
for p in root.rglob('*'):
    if not p.is_file(): continue
    if '__pycache__' in p.parts or '.pytest_cache' in p.parts: continue
    if p.name.endswith('.zip'): continue
    files.append(p)
checks=[]
for p in sorted(files):
    h=hashlib.sha256(); h.update(p.read_bytes()); checks.append({'path':str(p.relative_to(root)),'sha256':h.hexdigest(),'bytes':p.stat().st_size})
(root/'reports/release_manifest.json').write_text(json.dumps(checks,indent=2))
zip_path=root.parent/'Arjun_Powerpack_RUL_Hybrid_v3_production_ready.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:
        z.write(p,p.relative_to(root.parent))
print(zip_path)
