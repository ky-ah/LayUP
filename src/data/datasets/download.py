import os
import requests
import zipfile
from tqdm import tqdm


def download_file(url, directory, filename=None):
    if filename is None:
        filename = url.split("/")[-1]

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Full path to save the file
    file_path = os.path.join(directory, filename)

    # Downloading the file with a progress bar
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    if response.status_code == 200:
        with open(file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print(f"File downloaded successfully and saved as {file_path}")
    else:
        progress_bar.close()
        print(f"Failed to download the file. Status code: {response.status_code}")


def download_domainnet(root="./data"):
    dil_tasks_imgs = {
        "clipart": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "infograph": "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "painting": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "quickdraw": "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "real": "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "sketch": "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
    }

    dil_task_train_txt = {
        "clipart": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
        "infograph": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
        "painting": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
        "quickdraw": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
        "real": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
        "sketch": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
    }

    dil_task_test_txt = {
        "clipart": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt",
        "infograph": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt",
        "painting": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt",
        "quickdraw": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt",
        "real": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
        "sketch": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt",
    }

    # setup directories
    root = os.path.join(root, "domainnet")
    os.makedirs(root, exist_ok=True)

    # download images
    for task, url in dil_tasks_imgs.items():
        if os.path.exists(os.path.join(root, task)):
            continue
        download_file(url, root, f"{task}.zip")
        full_zip_path = os.path.join(root, f"{task}.zip")
        with zipfile.ZipFile(full_zip_path, "r") as zip_ref:
            zip_ref.extractall(root)
        # remove the zip file
        os.remove(full_zip_path)

    # download train/test splits
    for task, url in dil_task_train_txt.items():
        if os.path.exists(os.path.join(root, f"{task}_train.txt")):
            continue
        download_file(url, root, f"{task}_train.txt")

    for task, url in dil_task_test_txt.items():
        if os.path.exists(os.path.join(root, f"{task}_test.txt")):
            continue
        download_file(url, root, f"{task}_test.txt")
