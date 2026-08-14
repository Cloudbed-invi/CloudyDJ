import json
with open('library/crate.json', 'r') as f:
    crate = json.load(f)
for k, v in crate.items():
    print(f"{k}: Key={v.get('key')}, BPM={v.get('bpm')}")
