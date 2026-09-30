import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

load_dotenv()

class S3Storage:
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        
        # Leitura dos parâmetros temporários da AWS Academy
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region
        )

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Envia um arquivo local para o bucket S3 especificado."""
        try:
            print(f"Enviando {local_path} para s3://{self.bucket_name}/{s3_key}...")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print("Upload concluído com sucesso!")
            return True
        except FileNotFoundError:
            print(f"Erro: Arquivo local {local_path} não encontrado.")
            return False
        except NoCredentialsError:
            print("Erro: Credenciais da AWS não encontradas ou expiradas.")
            return False
        except ClientError as e:
            print(f"Erro ao comunicar com AWS S3: {e}")
            return False