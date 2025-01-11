import base64
import datetime
import hmac
import io
import os
from dotenv import load_dotenv, dotenv_values
import jwt
import requests
from flask import Flask, request, jsonify
import ml
import boto3
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid
from PIL import Image
import traceback
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per hour"])

dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("REGION_NAME"))
table = dynamodb.Table('userdata')  

app.config['S3_BUCKET'] = os.getenv("S3_BUCKET_NAME")
app.config['S3_KEY'] = os.getenv("AWS_ACCESS_KEY_ID")
app.config['S3_SECRET'] = os.getenv("AWS_SECRET_ACCESS_KEY")
app.config['S3_LOCATION'] = 'http://{}.s3.amazonaws.com/'.format(app.config['S3_BUCKET'])

s3 = boto3.client("s3", aws_access_key_id=app.config['S3_KEY'], aws_secret_access_key=app.config['S3_SECRET'])
cognito = boto3.client('cognito-idp', region_name=os.getenv("REGION_NAME"), aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
COGNITO_ISSUER = f"https://cognito-idp.{os.getenv('REGION_NAME')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"


def get_cognito_keys():
    url = f"{COGNITO_ISSUER}/.well-known/jwks.json"
    response = requests.get(url)
    return response.json()

cognito_keys = get_cognito_keys()

def decode_jwt(token):
    
    headers = jwt.get_unverified_header(token)
    jwks_url = f"{COGNITO_ISSUER}/.well-known/jwks.json"
    jwk_client = jwt.PyJWKClient(jwks_url)
    
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=[headers["alg"]],
        audience=os.getenv("COGNITO_APP_CLIENT_ID"),
        issuer=COGNITO_ISSUER
    )
    return payload


@app.route('/signup', methods=['POST'])
@limiter.limit("2 per minute")
def signup():
    email = request.form['email']
    password = request.form['password']

    try:
        response = cognito.sign_up(
            ClientId=os.getenv("COGNITO_APP_CLIENT_ID"),
            Username=email,
            Password=password,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': email
                }
            ]
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("User already exists!")})

    return jsonify({"message": "User created successfully!",
                    "response": response})

@app.route('/confirm', methods=['POST'])
@limiter.limit("2 per minute")
def confirm():
    email = request.form['email']
    code = request.form['code']

    try:
        response = cognito.confirm_sign_up(
            ClientId=os.getenv("COGNITO_APP_CLIENT_ID"),
            Username=email,
            ConfirmationCode=code
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("Invalid code!")})

    return jsonify({"message": "User confirmed successfully!",
                    "response": response})

@app.route('/login', methods=['POST'])
@limiter.limit("2 per minute")
def login():
    email = request.form['email']
    password = request.form['password']

    try:
        response = cognito.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': password
            },
            ClientId=os.getenv("COGNITO_APP_CLIENT_ID")
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("Invalid credentials!")})

    return jsonify({"message": "User logged in successfully!",
                    "response": response})

@app.route('/upload', methods=['POST'])
@limiter.limit("2 per minute")
def upload():
    auth_header = request.headers.get('Authorization')
    username = None
    if auth_header is None:
        traceback.print_exc()
        return jsonify({"error": "Authorization header is missing!"})
    
    token = auth_header.split(" ")[1]

    try:
        payload = decode_jwt(token)
        username = payload['email']
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("Invalid token!")})
    
    file = request.files['image']
    image_key = f"{username}/{file.filename}"
    image_id = str(uuid.uuid4())

    try:
        s3.upload_fileobj(
            file,
            app.config['S3_BUCKET'],
            image_key,
            ExtraArgs={
                "ContentType": file.content_type
            }
        )
    except Exception as e:
        return jsonify({"error": str("Object already exists!")})

    try:
        table.put_item(
            Item={
                'username': username,
                'item_type': 'image',
                'image_id': image_id,
                'image_key': image_key,
                'upload_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        return jsonify({"error": str("Table already exists!")})

    return jsonify({"message": "Image uploaded successfully!"})

@app.route('/predict', methods=['POST'])
@limiter.limit("2 per minute")
def predict():
    
    auth_header = request.headers.get('Authorization')
    username = None
    if auth_header is None:
        traceback.print_exc()
        return jsonify({"error": "Authorization header is missing!"})
    
    token = auth_header.split(" ")[1]

    try:
        payload = decode_jwt(token)
        username = payload['email']
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("Invalid token!")})
    user_prompt = request.form['user_prompt']
    response = None
    cloths = []

    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("username").eq(username) &
            boto3.dynamodb.conditions.Key("item_type").begins_with('image')
        )

        for items in response['Items']:
            image_key = items['image_key']
            presigned_url = s3.generate_presigned_url('get_object',
                Params={
                    'Bucket': app.config['S3_BUCKET'],
                    'Key': image_key
                },
                ExpiresIn=3600
            )
            
            response = requests.get(presigned_url)
            image = Image.open(io.BytesIO(response.content)).resize((224, 224))
            cloths.append(image)

        ans = ml.predict(cloths, user_prompt=user_prompt)
        for image in ans:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            response = {"image": img_str}
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str("Image not found!")}) 

    return response

@app.route('/')
@limiter.limit("5 per minute")
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
