#!/usr/bin/env bash
# shared names for the aws test deployment. everything lives in one region and one
# account; 99_teardown.sh removes all of it by these names.
export AWS_REGION="${STYLIST_REGION:-us-east-1}"
export APP=fashion-stylist
if [ "$AWS_REGION" = "us-east-1" ]; then
  export BUCKET=fashion-stylist-889982482580-artifacts
  export BEDROCK_REGION=us-east-1
  export LLM_MODEL_DEPLOY=us.amazon.nova-micro-v1:0
  export LLM_RERANK_MODEL_DEPLOY=us.amazon.nova-micro-v1:0
else
  # a second region gets its own bucket (regional boot, no cross-region pull) and the
  # apac inference profile; verified live in ap-northeast-1 before the tokyo move
  export BUCKET="fashion-stylist-889982482580-artifacts-${AWS_REGION}"
  export BEDROCK_REGION="$AWS_REGION"
  # lite, not micro: same background planning cost to the user, and the live 28-query
  # probe scored lite at 0.772 match@4 micro vs 0.705 for micro (success 0.50 vs 0.39)
  export LLM_MODEL_DEPLOY=apac.amazon.nova-lite-v1:0
  # micro on the per-slot rerank calls: measured 1.3 s for a 5 slot rerank vs ~7 s
  # with lite doing both jobs. lite plans, micro reranks.
  export LLM_RERANK_MODEL_DEPLOY=apac.amazon.nova-micro-v1:0
fi
export SG_NAME=${APP}-sg
export ROLE_NAME=${APP}-ec2-role
export PROFILE_NAME=${APP}-ec2-profile
export INSTANCE_NAME=${APP}-api
export INSTANCE_TYPE=c7i.xlarge
