#!/usr/bin/env bash
# removes everything the other scripts made. cloudfront disable+delete takes ~5 min.
set -uo pipefail
cd "$(dirname "$0")"
source ./00_env.sh
if [ -f .cloudfront ]; then
  DIST=$(awk '{print $2}' .cloudfront)
  ETAG=$(aws cloudfront get-distribution-config --id "$DIST" --query ETag --output text)
  aws cloudfront get-distribution-config --id "$DIST" --query DistributionConfig > /tmp/dc.json
  python3 -c "import json; d=json.load(open('/tmp/dc.json')); d['Enabled']=False; json.dump(d, open('/tmp/dc.json','w'))"
  aws cloudfront update-distribution --id "$DIST" --distribution-config file:///tmp/dc.json --if-match "$ETAG" >/dev/null
  aws cloudfront wait distribution-deployed --id "$DIST"
  ETAG=$(aws cloudfront get-distribution-config --id "$DIST" --query ETag --output text)
  aws cloudfront delete-distribution --id "$DIST" --if-match "$ETAG"
fi
if [ -f .instance ]; then
  IID=$(awk '{print $1}' .instance)
  aws ec2 terminate-instances --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$IID"
fi
for ALLOC in $(aws ec2 describe-addresses --filters Name=tag:Name,Values=$INSTANCE_NAME --query "Addresses[].AllocationId" --output text); do
  aws ec2 release-address --allocation-id "$ALLOC"
done
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
[ -n "$SG" ] && [ "$SG" != "None" ] && aws ec2 delete-security-group --group-id "$SG"
aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null
aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "${APP}-inline" 2>/dev/null
aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null
aws s3 rm "s3://$BUCKET" --recursive && aws s3 rb "s3://$BUCKET"
echo "teardown done"
