# MS Converter

MS Converter can replace PyTorch interfaces in scripts with MindSpore (v2.5.0) Mint interfaces, reducing repetitive tasks and improving development efficiency. :D

*Note: The converted script may not necessarily run successfully.*

## Usage

Run `./torch2mint.py your_pytorch_script.py`. Your PyTorch script will be updated, while the original script will be saved as `your_pytorch_script.py.old`.

## Note

- To update the MindSpore Mint API list, download the updated `.rst` file from the MindSpore website (e.g., [MindSpore 2.5.0 API Documentation](https://gitee.com/mindspore/mindspore/blob/v2.5.0/docs/api/api_python/mindspore.mint.rst#) ) and replace the contents of the `./mindspore_v2.5.0.mint.rst` file with the new version.

- To add conversion rules, simply update or modify the mappings in the `extra_mapping.json` file.