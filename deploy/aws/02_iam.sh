#!/usr/bin/env bash
# instance role: read the two artifacts, invoke bedrock models. nothing else.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

cat > /tmp/trust.json <<'JSON'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
JSON
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file:///tmp/trust.json >/dev/null

cat > /tmp/policy.json <<JSON
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:GetObject"],
   "Resource": ["arn:aws:s3:::$BUCKET/*", "arn:aws:s3:::$BUCKET-*/*"]},
  {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
   "Resource": "*"}
]}
JSON
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "${APP}-inline" --policy-document file:///tmp/policy.json
# ssm for ops access (session manager / run command); no ssh keys anywhere in this setup
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
}
echo "iam ready"
