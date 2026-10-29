import os
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"\nDownloading: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    root_dir = "./coco_train2017"
    os.makedirs(root_dir, exist_ok=True)
    
    # Official COCO train2017 Images (19 GB) & Ultralytics YOLO-converted Labels
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    labels_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"

    images_zip = os.path.join(root_dir, "train2017.zip")
    labels_zip = os.path.join(root_dir, "coco2017labels.zip")

    if not os.path.exists(images_zip):
        download_file(train_images_url, images_zip)
    if not os.path.exists(labels_zip):
        download_file(labels_url, labels_zip)

    temp_extract = os.path.join(root_dir, "temp_extract")
    os.makedirs(temp_extract, exist_ok=True)
    
    extract_zip(images_zip, temp_extract)
    extract_zip(labels_zip, temp_extract)

    final_images_dir = os.path.join(root_dir, "images", "train")
    final_labels_dir = os.path.join(root_dir, "labels", "train")
    os.makedirs(final_images_dir, exist_ok=True)
    os.makedirs(final_labels_dir, exist_ok=True)

    print("Organizing folders into standard YOLO structure...")
    
    extracted_img_dir = os.path.join(temp_extract, "train2017")
    if os.path.exists(extracted_img_dir):
        for file_name in os.listdir(extracted_img_dir):
            shutil.move(os.path.join(extracted_img_dir, file_name), os.path.join(final_images_dir, file_name))

    extracted_lbl_dir = os.path.join(temp_extract, "coco", "labels", "train2017")
    if os.path.exists(extracted_lbl_dir):
        for file_name in os.listdir(extracted_lbl_dir):
            shutil.move(os.path.join(extracted_lbl_dir, file_name), os.path.join(final_labels_dir, file_name))

    # Cleanup (Handling the Windows permission error cleanly)
    try:
        shutil.rmtree(temp_extract)
    except PermissionError:
        print(f"Note: Could not automatically delete {temp_extract} due to Windows file locking. You can delete it manually.")
        
    if os.path.exists(images_zip):
        os.remove(images_zip)
    if os.path.exists(labels_zip):
        os.remove(labels_zip)

    print(f"\nDone! Training dataset ready at: {os.path.abspath(root_dir)}")

if __name__ == "__main__":
    main()