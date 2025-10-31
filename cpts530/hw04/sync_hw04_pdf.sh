#!/bin/bash

# Script to sync Hw04_Cpts530 PDF from documentsCreated_repo
# Author: Aryan Ritwajeet Jha
# Date: October 30, 2025

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== HW04 PDF Sync Script ===${NC}"
echo ""

# Define paths
DOCUMENTS_CREATED_REPO="C:/Users/Aryan Ritwajeet Jha/Documents/documents_general/documentsCreated/documentsCreated_repo"
SOURCE_DIR="${DOCUMENTS_CREATED_REPO}/journals"
SOURCE_FILE="${SOURCE_DIR}/Hw04_Cpts530"
DEST_DIR="C:/Users/Aryan Ritwajeet Jha/Documents/documents_general/structured_courses/cpts530/hw04"

# Step 1: Pull latest changes from documentsCreated_repo
echo -e "${BLUE}Step 1: Pulling latest changes from documentsCreated_repo...${NC}"
cd "${DOCUMENTS_CREATED_REPO}" || {
    echo -e "${RED}Error: Could not navigate to documentsCreated_repo${NC}"
    exit 1
}

git fetch
git pull origin main || git pull origin master || {
    echo -e "${YELLOW}Warning: Could not pull changes (might already be up to date)${NC}"
}

echo -e "${GREEN}✓ Repository updated${NC}"
echo ""

# Step 2: Check if source file exists
echo -e "${BLUE}Step 2: Checking for Hw04_Cpts530 file...${NC}"

# Look for the file with common extensions
if [ -f "${SOURCE_FILE}.pdf" ]; then
    SOURCE_FILE_FULL="${SOURCE_FILE}.pdf"
    echo -e "${GREEN}✓ Found: Hw04_Cpts530.pdf${NC}"
elif [ -f "${SOURCE_FILE}.PDF" ]; then
    SOURCE_FILE_FULL="${SOURCE_FILE}.PDF"
    echo -e "${GREEN}✓ Found: Hw04_Cpts530.PDF${NC}"
else
    echo -e "${RED}Error: Hw04_Cpts530 file not found in ${SOURCE_DIR}${NC}"
    echo -e "${YELLOW}Looking for files in the directory...${NC}"
    ls -la "${SOURCE_DIR}" | grep -i hw04 || echo "No hw04 files found"
    exit 1
fi

echo ""

# Step 3: Copy file to destination
echo -e "${BLUE}Step 3: Copying file to hw04 folder...${NC}"

# Get file info before copy
SOURCE_SIZE=$(stat -f%z "${SOURCE_FILE_FULL}" 2>/dev/null || stat -c%s "${SOURCE_FILE_FULL}" 2>/dev/null)
SOURCE_DATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${SOURCE_FILE_FULL}" 2>/dev/null || stat -c "%y" "${SOURCE_FILE_FULL}" 2>/dev/null)

echo -e "Source: ${SOURCE_FILE_FULL}"
echo -e "Size: ${SOURCE_SIZE} bytes"
echo -e "Modified: ${SOURCE_DATE}"
echo ""

# Copy the file
cp "${SOURCE_FILE_FULL}" "${DEST_DIR}/" || {
    echo -e "${RED}Error: Could not copy file${NC}"
    exit 1
}

echo -e "${GREEN}✓ File copied successfully to:${NC}"
echo -e "  ${DEST_DIR}/$(basename "${SOURCE_FILE_FULL}")"
echo ""

# Step 4: Verify copy
echo -e "${BLUE}Step 4: Verifying copy...${NC}"
DEST_FILE="${DEST_DIR}/$(basename "${SOURCE_FILE_FULL}")"

if [ -f "${DEST_FILE}" ]; then
    DEST_SIZE=$(stat -f%z "${DEST_FILE}" 2>/dev/null || stat -c%s "${DEST_FILE}" 2>/dev/null)
    DEST_DATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${DEST_FILE}" 2>/dev/null || stat -c "%y" "${DEST_FILE}" 2>/dev/null)
    
    echo -e "Destination: ${DEST_FILE}"
    echo -e "Size: ${DEST_SIZE} bytes"
    echo -e "Modified: ${DEST_DATE}"
    
    if [ "${SOURCE_SIZE}" == "${DEST_SIZE}" ]; then
        echo -e "${GREEN}✓ File sizes match - Copy verified!${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: File sizes don't match${NC}"
    fi
else
    echo -e "${RED}Error: Destination file not found after copy${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Sync Complete ===${NC}"
echo -e "${BLUE}You can now work with the latest version of Hw04_Cpts530${NC}"
