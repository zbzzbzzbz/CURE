#!/bin/bash

# Define colors
BLUE='\033[0;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

download_checkpoint() {
    local url="$1"
    local dir="$2"
    local filename="$3"

    # Create the directory if it doesn't exist
    mkdir -p "$dir"

    # Check if the file already exists
    if [ -e "$dir/$filename" ]; then
        echo -e "The ${BLUE}$filename${NC} checkpoint already exists in $dir."
        # Ask the user if they want to overwrite the file, defaulting to 'no'
        read -p "Do you want to overwrite? [y/N]: " overwrite
        case $overwrite in
            [Yy]* ) ;;
            * ) return ;;
        esac
    fi

    # Download the file and check if it's downloaded correctly
    echo -e "Downloading ${BLUE}$filename${NC} checkpoint..."
    if wget -q --show-progress "$url" -O "$dir/$filename"; then
        echo -e "${GREEN}Download successful.${NC}"
    else
        echo -e "${RED}Download failed.${NC}"
    fi
}

# soon to release
url="xxx"
dir="pretrained_models"
filename="cure_webvid_noise60.ckpt"
download_checkpoint "$url" "$dir" "$filename"

