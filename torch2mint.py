#!/usr/bin/env python
import argparse
import json
import re
import shutil
from typing import Dict, List

MINT_API_DOC = "mindspore_v2.5.0.mint.rst"
EXTRA_MAPPING = "extra_mapping.json"


def scan_mint_api():
    with open(MINT_API_DOC, "r") as f:
        content = f.read()
    result = re.findall(r"    mindspore.mint.(\w.+)", content)
    return result


def form_base_torch_mint_mapping(api: List[str]) -> Dict[str, str]:
    mapping = dict()
    for x in api:
        torch_api_name = "torch." + x
        mint_api_name = "mint." + x
        mapping[torch_api_name] = mint_api_name
    return mapping


def expand_torch_mint_mapping(mapping: Dict[str, str]):
    with open(EXTRA_MAPPING, "r") as f:
        extra_mapping = json.load(f)

    for u, v in mapping.items():
        if ".nn." in u:
            extra_mapping[u.replace("torch.", "")] = v

    mapping.update(extra_mapping)
    return


def main():
    parser = argparse.ArgumentParser(
        usage="Convert script from PyTorch to MindSpore(Mint) partially.")
    parser.add_argument("input", help="Path of the script")
    args = parser.parse_args()

    api = scan_mint_api()
    mapping = form_base_torch_mint_mapping(api)
    expand_torch_mint_mapping(mapping)

    with open(args.input, "r") as f:
        content = f.read()
    for u, v in mapping.items():
        if u not in content:
            continue

        content = re.sub(u, v, content)
        msg = f"{u.strip()} --> {v.strip()}"
        print(msg)

    shutil.move(args.input, args.input + ".old")
    with open(args.input, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
