#!/usr/bin/env python3
"""Package files from 06.images into a zip for easy download/share."""
import os
import zipfile


def package_images():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(base, "06.images")
    if not os.path.isdir(images_dir):
        print("06.images directory not found:", images_dir)
        return 1

    zip_path = os.path.join(images_dir, "collected_images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f == os.path.basename(zip_path):
                    continue
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, images_dir)
                z.write(path, arcname)

    print("Packaged images to:", zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(package_images())
