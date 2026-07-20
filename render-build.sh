#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python requirements
pip install -r requirements.txt

# Create local bin folder for static executables if not present
mkdir -p bin

# Check if static FFmpeg binary exists
if [ ! -f "bin/ffmpeg" ]; then
  echo "Downloading static FFmpeg release..."
  curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz
  
  echo "Extracting FFmpeg package..."
  tar -xf ffmpeg.tar.xz
  
  # Find extracted folder and copy bin files
  EXTRACTED_DIR=$(find . -maxdepth 1 -name "ffmpeg-*-static" -type d | head -n 1)
  if [ -d "$EXTRACTED_DIR" ]; then
    cp "$EXTRACTED_DIR/ffmpeg" bin/
    cp "$EXTRACTED_DIR/ffprobe" bin/
    chmod +x bin/ffmpeg bin/ffprobe
    rm -rf "$EXTRACTED_DIR"
  fi
  
  rm ffmpeg.tar.xz
  echo "FFmpeg successfully installed in ./bin"
fi
