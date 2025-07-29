import requests
import os
import config
from tqdm import tqdm

print("Downloading required assets...")

# URL for the sample video
# Using a new video from archive.org for better reliability.
video_url = "https://archive.org/download/Road-Traffic-Dashcam-Video/Road-Traffic-Dashcam-Video.mp4"
destination_path = config.VIDEO_INPUT_PATH

if not os.path.exists(destination_path):
    try:
        print(f"Downloading sample video to '{destination_path}'...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(video_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(destination_path, 'wb') as f, tqdm(
            desc=destination_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        if total_size != 0 and bar.n != total_size:
            print("ERROR, something went wrong during download.")
        else:
            print("\nDownload complete.")

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please try downloading the video manually from the URL below and save it as 'sample_video.mp4' in your project directory.")
        print(f"URL: {video_url}")
else:
    print(f"'{destination_path}' already exists. Skipping download.")