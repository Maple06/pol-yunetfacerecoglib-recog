from fastapi import UploadFile, File
from pydantic import BaseModel

class ImageAbsen(BaseModel):
    file: UploadFile = File(...)