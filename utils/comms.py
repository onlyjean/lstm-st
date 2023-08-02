import streamlit as st
import pandas as pd
import boto3
import os
from io import StringIO
from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.

# AWS S3 comment-section bucket
comment_bucket = 'comment-section-st'

# AWS credentials
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')


def connect():
    # Create a connection object.
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    return s3

def collect(s3, comment_bucket, file_name) -> pd.DataFrame:
    obj = s3.get_object(Bucket=comment_bucket, Key=file_name)
    comments = pd.read_csv(obj['Body'])
    return comments

def insert(s3, comment_bucket, file_name, row) -> None:
    comments = collect(s3, comment_bucket, file_name)
    new_comment = pd.DataFrame([row], columns=['name', 'comment', 'date'])
    comments = comments.append(new_comment, ignore_index=True)
    csv_buffer = StringIO()
    comments.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3_resource.Object(comment_bucket, file_name).put(Body=csv_buffer.getvalue())


def write_to_s3(s3, comment_bucket, file_name, comments) -> None:
    csv_buffer = StringIO()
    comments.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3_resource.Object(comment_bucket, file_name).put(Body=csv_buffer.getvalue())
