set -euo pipefail
mkdir -p data/raw
cd data/raw
curl -L -O https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2
echo "Downloaded to data/raw/enwikivoyage-latest-pages-articles.xml.bz2"
