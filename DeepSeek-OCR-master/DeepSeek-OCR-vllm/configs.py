from dotenv import load_dotenv
import os

load_dotenv()

STRAPI_API_URL = os.getenv("STRAPI_API_URL")
STRAPI_API_TOKEN = os.getenv("STRAPI_API_TOKEN")
