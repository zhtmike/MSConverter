# MS Converter

MS Converter can replace PyTorch interfaces in scripts with MindSpore (v2.5.0) Mint interfaces, reducing repetitive tasks and improving development efficiency. :D

*Note: The converted script may not necessarily run successfully.*

## Install

Clone the repository and run `python setup.py install`

## Usage

Run `torch2mint your_pytorch_script.py`. Your script will be updated, while the original script will be saved as `your_pytorch_script.py.old`.

## Note

- To update the MindSpore Mint API list, download the `.rst` file from the MindSpore website (e.g., [MindSpore 2.5.0 API List](https://gitee.com/mindspore/mindspore/blob/v2.5.0/docs/api/api_python/mindspore.mint.rst) ), run 
  ```bash
  torch2mint your_pytorch_script.py --mint_api_path your_mint_api_path.rst
  ```

- To add conversion rules, simply add a json file with the content
  ```json
  {"the content you want to replace": "the content you want to fill"}
  ```
  similar to `assets/mapping.json`, then run 
  ```bash
  torch2mint your_pytorch_script.py --custom_mapping_path your_custom_mapping_path.json
  ```
