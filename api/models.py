from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    
class FileResponse(BaseModel):
    file_id: str
    filename: str
    
class FilesListResponse(BaseModel):
    files: List[FileResponse]
    
class DeleteFileRequest(BaseModel):
    file_ids: List[str]
    
class DeleteFileResponse(BaseModel):
    deleted: List[str]
    failed: List[str]