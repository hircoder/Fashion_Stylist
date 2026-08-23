#!/usr/bin/env bash
# shared names for the aws test deployment. everything lives in one region and one
# account; 99_teardown.sh removes all of it by these names.
export AWS_REGION=us-east-1
export APP=fashion-stylist
export BUCKET=fashion-stylist-889982482580-artifacts
export SG_NAME=${APP}-sg
export ROLE_NAME=${APP}-ec2-role
export PROFILE_NAME=${APP}-ec2-profile
export INSTANCE_NAME=${APP}-api
export INSTANCE_TYPE=c7i.xlarge
