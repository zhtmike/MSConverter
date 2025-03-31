#!/usr/bin/env python
import argparse
import json
import logging
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

from ms_converter._version import __version__

DEFAULT_MINT_API_PATH = "assets/mindspore_v2.5.0.mint.rst"
DEFAULT_MAPPING = "assets/mapping.json"

_logger = logging.getLogger(__name__)


def scan_mint_api(mint_api_path: Optional[str] = None) -> List[str]:
    if mint_api_path is not None:
        if not mint_api_path.endswith(".rst"):
            raise ValueError(
                "`mint_api_path` shoud be a path ends with `.rst`.")
        mint_api_path = os.path.abspath(mint_api_path)
        _logger.debug("Reading the custom MindSpore Mint API doc from %s",
                      mint_api_path)
    else:
        mint_api_path = os.path.abspath(DEFAULT_MINT_API_PATH)
        _logger.debug("Reading the default MindSpore Mint API doc from %s",
                      mint_api_path)

    with open(mint_api_path, "r") as f:
        content = f.read()

    result = re.findall(r"[ +]mindspore.mint.(\w.+)", content)
    _logger.debug("Collected %d Mint APIs", len(result))
    return result


def form_base_torch_mint_mapping(api: List[str]) -> Dict[str, str]:
    mapping = dict()
    for x in api:
        torch_api_name = "torch." + x
        mint_api_name = "mint." + x
        mapping[torch_api_name] = mint_api_name
    return mapping


def expand_torch_mint_mapping(
        mapping: Dict[str, str],
        custom_mapping_path: Optional[str] = None) -> None:
    extra_mapping = os.path.abspath(DEFAULT_MAPPING)
    _logger.debug("Reading the mapping from %s", extra_mapping)
    with open(extra_mapping, "r") as f:
        extra_mapping = json.load(f)

    if custom_mapping_path is not None:
        custom_mapping_path = os.path.abspath(custom_mapping_path)
        _logger.debug("Reading the custom mapping from %s",
                      custom_mapping_path)
        with open(custom_mapping_path, "r") as f:
            custom_mapping = json.load(f)
        extra_mapping.update(custom_mapping)

    # in pytorch script, we usually use nn.xxx instead of torch.nn.xxx
    for u, v in mapping.items():
        if ".nn." in u:
            extra_mapping[u.replace("torch.", "")] = v

    mapping.update(extra_mapping)
    return


def _torch2mint(
    input_: str,
    mint_api_path: Optional[str] = None,
    custom_mapping_path: Optional[str] = None,
    inplace: bool = False,
) -> None:
    input_ = os.path.abspath(input_)
    api = scan_mint_api(mint_api_path=mint_api_path)
    mapping = form_base_torch_mint_mapping(api)
    expand_torch_mint_mapping(mapping, custom_mapping_path=custom_mapping_path)

    _logger.debug("Reading input from %s", input_)
    with open(input_, "r") as f:
        content = f.read()

    for u, v in mapping.items():
        if u not in content:
            continue

        content = re.sub(u, v, content)
        msg = f"Update: {u.strip():<40} --> {v.strip():<40}"
        _logger.debug(msg)

    if inplace:
        backup = input_ + ".old"
        shutil.move(input_, backup)
        _logger.debug("Backup file is saved as %s", backup)
        with open(input_, "w") as f:
            f.write(content)
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(
        usage=
        "Convert script from using PyTorch API to MindSpore API (mint API) partially."
    )
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument("input", help="Path of the script to be converted.")
    parser.add_argument(
        "-i",
        "--in_place",
        action="store_true",
        help="make the update to files in place",
    )
    parser.add_argument("--mint_api_path", help="Path to the Mint API list")
    parser.add_argument("--custom_mapping_path",
                        help="Path to the custom mapping list")
    parser.add_argument("-vv",
                        "--verbose",
                        action="store_true",
                        help="Shown the debug message")
    args = parser.parse_args()

    # set logger
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    _logger.setLevel(logging_level)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    _logger.addHandler(console_handler)

    _torch2mint(
        args.input,
        mint_api_path=args.mint_api_path,
        custom_mapping_path=args.custom_mapping_path,
        inplace=args.in_place,
    )


if __name__ == "__main__":
    main()
