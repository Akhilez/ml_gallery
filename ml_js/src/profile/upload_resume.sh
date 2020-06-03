#!/bin/bash
set -x

RESUME_PATH="`ls ~/jobs/*esume*.pdf`"
BUCKET_FILE_PATH="gs://akhilez/resume.pdf"

if [[ -f ${RESUME_PATH} ]]; then
    echo "Resume found at $RESUME_PATH, uploading it to google drive"
    gsutil cp ${RESUME_PATH} ${BUCKET_FILE_PATH}
    gsutil acl ch -u AllUsers:R ${BUCKET_FILE_PATH}
fi
