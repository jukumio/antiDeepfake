#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide an argument: celeba, pretrained-celeba-128x128, or pretrained-celeba-256x256"
    exit 1
fi

FILE=$1

# Function to check if commands exist
check_requirements() {
    command -v wget >/dev/null 2>&1 || { echo "wget is required but not installed. Please install it first."; exit 1; }
    command -v unzip >/dev/null 2>&1 || { echo "unzip is required but not installed. Please install it first."; exit 1; }
}

# Check requirements first
check_requirements

case "$FILE" in
    "celeba")
        # CelebA images and attribute labels
        URL="https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0"
        ZIP_FILE="./data/celeba.zip"
        wget -N "$URL" -O "$ZIP_FILE"
        unzip "$ZIP_FILE" -d ./data/
        rm "$ZIP_FILE"
        ;;
        
    "pretrained-celeba-128x128")
        # StarGAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 128x128 resolution
        URL="https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=0"
        ZIP_FILE="./stargan_celeba_128/models/celeba-128x128-5attrs.zip"
        mkdir -p ./stargan_celeba_128/models/
        wget -N "$URL" -O "$ZIP_FILE"
        unzip "$ZIP_FILE" -d ./stargan_celeba_128/models/
        rm "$ZIP_FILE"
        ;;
        
    "pretrained-celeba-256x256")
        # StarGAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 256x256 resolution
        URL="https://www.dropbox.com/s/zdq6roqf63m0v5f/celeba-256x256-5attrs.zip?dl=0"
        ZIP_FILE="./stargan_celeba_256/models/celeba-256x256-5attrs.zip"
        wget -N "$URL" -O "$ZIP_FILE"
        unzip "$ZIP_FILE" -d ./stargan_celeba_256/models/
        rm "$ZIP_FILE"
        ;;
        
    *)
        echo "Available arguments are: celeba, pretrained-celeba-128x128, pretrained-celeba-256x256"
        exit 1
        ;;
esac

echo "Download and extraction completed for $FILE"