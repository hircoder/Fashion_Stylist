#!/usr/bin/env bash
# security group (8000 from cloudfront origin-facing ranges only), elastic ip, instance
set -euo pipefail
cd "$(dirname "$0")"
source ./00_env.sh

VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC" \
      --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
  SG=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "stylist api behind cloudfront" \
        --vpc-id "$VPC" --query GroupId --output text)
  PL=$(aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
        --query "PrefixLists[0].PrefixListId" --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" \
    --ip-permissions "IpProtocol=tcp,FromPort=8000,ToPort=8000,PrefixListIds=[{PrefixListId=$PL}]"
fi
echo "SG=$SG"

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)
sed -e "s/__BUCKET__/$BUCKET/g" -e "s/__BEDROCK_REGION__/$BEDROCK_REGION/g" \
    -e "s|__LLM_MODEL__|$LLM_MODEL_DEPLOY|g" \\
    -e "s|__LLM_RERANK_MODEL__|$LLM_RERANK_MODEL_DEPLOY|g" user-data.sh > /tmp/user-data.sh

IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile Name="$PROFILE_NAME" --security-group-ids "$SG" \
  --user-data file:///tmp/user-data.sh \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
  --query "Instances[0].InstanceId" --output text)
echo "instance $IID starting"
aws ec2 wait instance-running --instance-ids "$IID"

ALLOC=$(aws ec2 allocate-address --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query AllocationId --output text)
aws ec2 associate-address --instance-id "$IID" --allocation-id "$ALLOC" >/dev/null
IP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query "Addresses[0].PublicIp" --output text)
# the public dns comes from the api, never from string surgery: the hostname shape
# differs per region (compute-1 in virginia, <region>.compute everywhere else)
DNS=$(aws ec2 describe-instances --instance-ids "$IID" \
      --query "Reservations[0].Instances[0].PublicDnsName" --output text)
echo "instance=$IID ip=$IP origin_dns=$DNS region=$AWS_REGION"
echo "$IID $IP $DNS $AWS_REGION" > ".instance.$AWS_REGION"
cp ".instance.$AWS_REGION" .instance  # most recent launch, what 04/05 read
