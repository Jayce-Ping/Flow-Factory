#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use that directory for the download
hf download Jayce-Ping/ShareGPT-4o-Image-Mini --repo-type dataset --local-dir "$SCRIPT_DIR"

echo "Download completed."

# Decompress using the absolute path to the file
tar -xzvf "$SCRIPT_DIR/images.tar.gz" -C "$SCRIPT_DIR"