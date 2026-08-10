import json
with open('library/crate.json') as f:
    crate = json.load(f)

SR = 22050  # structure_analyzer uses sr=22050 by default

for name in crate:
    if 'Secrets' in name or 'Wild' in name:
        print(f'=== {name} ===')
        data = crate[name]
        print(f'BPM: {data["bpm"]}')
        segs = data.get('segments', [])
        for s in segs:
            label = s["label"]
            start_s = s["start_sample"] / SR
            end_s = s["end_sample"] / SR
            energy = s["energy"]
            print(f'  {label:10s} {start_s:6.1f}s - {end_s:6.1f}s  energy={energy:.4f}')
        print()
