#!/bin/bash
# Script to fetch and copy HW02 PDF from OriginalDocsHub repo
# Author: Aryan Ritwajeet Jha
# Date: October 2025

set -e  # Exit on any error

echo "=========================================="
echo "Fetching HW02 PDF from OriginalDocsHub"
echo "=========================================="
echo ""

# Define paths
ORIGINALDOCSHUB_REPO="$HOME/Documents/documents_general/OriginalDocsHub0/OriginalDocsHub/journals"
TARGET_DIR="$HOME/Documents/documents_general/structured_courses/ee582/tex_Hw02"
PDF_NAME="Hw02_EE582.pdf"

# Check if OriginalDocsHub repo exists
if [ ! -d "$ORIGINALDOCSHUB_REPO" ]; then
    echo "Error: OriginalDocsHub repository not found at $ORIGINALDOCSHUB_REPO"
    exit 1
fi

# Navigate to OriginalDocsHub repo
echo "Navigating to OriginalDocsHub repo..."
cd "$ORIGINALDOCSHUB_REPO"

# Fetch from remote
echo "Fetching from remote..."
git fetch

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Find the PDF file
echo ""
echo "Searching for $PDF_NAME..."
PDF_PATH=$(find . -name "$PDF_NAME" -type f | head -n 1)

if [ -z "$PDF_PATH" ]; then
    echo "Error: $PDF_NAME not found in OriginalDocsHub repo"
    exit 1
fi

echo "Found: $PDF_PATH"

# Copy to target directory
echo ""
echo "Copying to $TARGET_DIR..."
cp "$PDF_PATH" "$TARGET_DIR/$PDF_NAME"

# Verify copy
if [ -f "$TARGET_DIR/$PDF_NAME" ]; then
    echo ""
    echo "✓ Success! PDF copied to:"
    echo "  $TARGET_DIR/$PDF_NAME"
    echo ""
    ls -lh "$TARGET_DIR/$PDF_NAME"
else
    echo "Error: Failed to copy PDF"
    exit 1
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
