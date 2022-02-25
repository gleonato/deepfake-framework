import os
import boto3
from botocore.exceptions import ClientError


action = 'r'
Bucket = 'raw-videos-gleonato'
local_path = '/home/sagemaker-user/deepfake-framework/4-AttackDetectionNet/data/train/Attacked/'
s3_dir = 'AttackDetectionNet/data/train/Attacked/'