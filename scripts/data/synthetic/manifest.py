import json
from pathlib import Path
from typing import List, Union

def read_manifest(manifest: Union[Path, str]) -> List[dict]:
    """Read a manifest file.
    Args:
        manifest (str or Path): Path to the manifest file.
    Returns:
        data (list): List of JSON items.
    Raises:
        RuntimeError: If there are errors encountered while reading the manifest file.
    """
    # Ensure the manifest path is a Path object
    manifest_path = Path(manifest)

    data = []
    errors = []

    # Attempt to open the file
    try:
        with manifest_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    item = json.loads(line)  # Parse JSON from the line
                    data.append(item)
                except json.JSONDecodeError:
                    errors.append(line)  # Log the problematic line

    except Exception as e:
        raise Exception(f"Manifest file could not be opened: {manifest_path}") from e

    if errors:
        # Raise an error if there were any issues parsing the file
        raise RuntimeError(f"Errors encountered while reading manifest file: {manifest_path}. " 
                           f"Invalid lines: {errors}")

    return data

def write_manifest(output_path: Union[Path, str], target_manifest: List[dict], ensure_ascii: bool = True):
    """
    Write to manifest file

    Args:
        output_path (str or Path): Path to output manifest file
        target_manifest (list): List of manifest file entries
        ensure_ascii (bool): default is True, meaning the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')