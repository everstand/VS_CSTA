#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

DATASETS = ('SumMe','TVSum')
SPLITS = tuple(f'split{i}.pt' for i in range(1,6))


def _resolve_dataset_dir(path, dataset):
    path = Path(path)
    nested = path / dataset
    if nested.is_dir():
        return nested
    return path


def _copy_dataset_weights(src, dst, dataset, overwrite=False):
    src = _resolve_dataset_dir(src, dataset)
    if not src.is_dir():
        raise FileNotFoundError(f'{dataset} source directory not found: {src}')
    dst = Path(dst) / dataset
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SPLITS:
        source = src / name
        if not source.is_file():
            raise FileNotFoundError(f'Missing {dataset} weight: {source}')
        target = dst / name
        if target.exists() and not overwrite:
            raise FileExistsError(f'Output weight already exists: {target}. Pass --overwrite to replace it.')
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def main():
    parser = argparse.ArgumentParser(description='Combine separately trained SumMe and TVSum weights into one CSTA weights directory.')
    parser.add_argument('--summe_weights_dir', required=True, help='Directory containing SumMe split*.pt files, or a root containing SumMe/.')
    parser.add_argument('--tvsum_weights_dir', required=True, help='Directory containing TVSum split*.pt files, or a root containing TVSum/.')
    parser.add_argument('--output_weights_dir', required=True, help='Output root that will contain SumMe/ and TVSum/.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output = Path(args.output_weights_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    copied.extend(_copy_dataset_weights(args.summe_weights_dir, output, 'SumMe', overwrite=args.overwrite))
    copied.extend(_copy_dataset_weights(args.tvsum_weights_dir, output, 'TVSum', overwrite=args.overwrite))
    print(f'combined_weights_dir={output}')
    for item in copied:
        print(item)


if __name__ == '__main__':
    main()
