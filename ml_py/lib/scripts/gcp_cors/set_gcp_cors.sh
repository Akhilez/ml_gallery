DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

gsutil cors set $DIR/cors.json gs://akhilez
