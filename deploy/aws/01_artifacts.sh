#!/usr/bin/env bash
# source tarball + index tarball into s3 (the instance bootstraps from these; no docker
# anywhere in this path)
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/aws/00_env.sh

aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3 mb "s3://$BUCKET" --region "$AWS_REGION"
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# source: the current branch, tracked files only
COPYFILE_DISABLE=1 git archive --format=tar.gz -o /tmp/stylist-src.tar.gz HEAD
aws s3 cp /tmp/stylist-src.tar.gz "s3://$BUCKET/src/stylist-src.tar.gz"

# index: pack the local popular-100K build (COPYFILE_DISABLE stops macOS tar from
# smuggling AppleDouble ._* files in, which the loader would rightly refuse)
COPYFILE_DISABLE=1 tar --exclude '._*' -czf /tmp/stylist-index.tar.gz -C data index
aws s3 cp /tmp/stylist-index.tar.gz "s3://$BUCKET/index/stylist-index.tar.gz"
echo "artifacts uploaded to s3://$BUCKET"
