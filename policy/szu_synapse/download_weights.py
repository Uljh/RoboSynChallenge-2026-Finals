"""Download a selected task and verify every published inference file."""
from pathlib import Path
import argparse,hashlib,json
from huggingface_hub import snapshot_download

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--destination',type=Path,default=Path(__file__).parent/'checkpoints')
    args=parser.parse_args()
    bindings=json.loads((Path(__file__).parent/'task_bindings.json').read_text())
    if args.task not in bindings:raise ValueError('Task has no finalized checkpoint binding')
    snapshot_download('lige65456/robosyn-finals-2026-pi05',revision=args.revision,
                      allow_patterns=[args.task+'/**'],local_dir=args.destination)
    checkpoint=args.destination/args.task
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    for row in manifest['files']:
        path=checkpoint/row['path']
        if not path.resolve().is_relative_to(checkpoint.resolve()):raise ValueError('Invalid manifest path')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if path.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
            raise ValueError('Checksum mismatch: '+row['path'])
    if manifest['normalization_sha256']!=bindings[args.task]['normalization_sha256']:
        raise ValueError('Downloaded weights do not match code binding')
    print(checkpoint.resolve())

if __name__=='__main__':main()
