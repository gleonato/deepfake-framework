import os
import boto3
from botocore.exceptions import ClientError

# Importante Definir caminhos S3 e Local


action = 'w'
Bucket = 'raw-videos-gleonato'
dest_path = 'AttackDetectionNet/data/train/Attacked/'
input_dir = '/home/sagemaker-user/deepfake-framework/4-AttackDetectionNet/data/train/Attacked/'

# [default]
# aws_access_key_id = YOUR_ACCESS_KEY
# aws_secret_access_key = YOUR_SECRET_KEY


s3 = boto3.resource('s3')

def write_data_s3(Bucket,dest_path,file):
    try:
        response = s3.Bucket(Bucket).upload_file(input_dir+file,Key=dest_path+str(file))
    except ClientError as e:
        logging.error(e)
        return False
    return True
    

    
if __name__ == "__main__":
    
# #  criar **args
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--localDir", help="which dir to upload to aws")
#     parser.add_argument("--bucketName", help="to which bucket to upload in aws")
#     parser.add_argument("--awsInitDir", help="to which 'directory' in aws")
#     parser.add_argument("--tag", help="some tag to select files, like *png", default='*')
#     args = parser.parse_args()



    data_dir = input_dir
    print(os.listdir(data_dir))
    counter = 0

    for file in os.listdir(data_dir):
#       print('Object #: {}'.format(counter))
        print(file)
        write_data_s3(Bucket,dest_path,file)
        counter =+ 1
