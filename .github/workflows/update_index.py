"""Update index.html for the bucket listing

http://gensim-wheels.s3-website-us-east-1.amazonaws.com/

We do this ourselves as opposed to using wheelhouse_uploader because it's
much faster this way (seconds as compared to nearly an hour).
"""

import sys
import boto3


def main():
    bucket = sys.argv[1]
    prefix = sys.argv[2]

    client = boto3.client('s3')

    print("<html><body><ul>")
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix):
        for content in page.get('Contents', []):
            key = content['Key']
            #
            # NB. use double quotes in href because that's that
            # wheelhouse_uploader expects.
            #
            # https://github.com/ogrisel/wheelhouse-uploader/blob/eb32a7bb410769bb4212a9aa7fb3bfa3cef1aaec/wheelhouse_uploader/fetch.py#L15
            #
            print(f"""<li><a href="{key}">{key}</a></li>""")
    print("</ul></body></html>")


if __name__ == '__main__':
    main()
