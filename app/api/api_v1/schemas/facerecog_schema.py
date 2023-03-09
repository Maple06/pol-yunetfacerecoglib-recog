from fastapi import UploadFile, File
from pydantic import BaseModel

class ImagePolda(BaseModel):
    file: UploadFile = File(...)