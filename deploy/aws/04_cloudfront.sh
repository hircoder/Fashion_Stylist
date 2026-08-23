#!/usr/bin/env bash
# one distribution, one custom origin (the ec2 elastic-ip dns on :8000, http only).
# static under /assets/* cached hard at the edge, everything else passes through.
set -euo pipefail
cd "$(dirname "$0")"
source ./00_env.sh
DNS=$(awk '{print $3}' .instance)

# managed policy ids: CachingOptimized, CachingDisabled, AllViewerExceptHostHeader
CACHE_OPT=658327ea-f89d-4fab-a63d-7e88639e58f6
CACHE_OFF=4135ea2d-6df8-44a3-9df3-4b5a84be39ad
ORIGIN_REQ=b689b0a8-53d0-40ab-baf2-68738e2966ac

cat > /tmp/dist.json <<JSON
{
  "CallerReference": "stylist-$(date +%s)",
  "Comment": "fashion stylist test deployment",
  "Enabled": true,
  "Origins": {"Quantity": 1, "Items": [{
    "Id": "api", "DomainName": "$DNS",
    "CustomOriginConfig": {"HTTPPort": 8000, "HTTPSPort": 443,
      "OriginProtocolPolicy": "http-only",
      "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 30}}]},
  "DefaultCacheBehavior": {
    "TargetOriginId": "api", "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {"Quantity": 7,
      "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
      "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
    "CachePolicyId": "$CACHE_OFF", "OriginRequestPolicyId": "$ORIGIN_REQ",
    "Compress": true},
  "CacheBehaviors": {"Quantity": 1, "Items": [{
    "PathPattern": "/assets/*", "TargetOriginId": "api",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"],
      "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
    "CachePolicyId": "$CACHE_OPT", "Compress": true}]}
}
JSON
OUT=$(aws cloudfront create-distribution --distribution-config file:///tmp/dist.json \
      --query "Distribution.{id:Id,domain:DomainName}" --output text)
echo "$OUT" | tee .cloudfront
