import tempfile 
from pathlib import Path 
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader

def upload_file(upload_file) -> list[Document] :
    suffix = Path(upload_file.name).suffix.lower()
    loader = {".pdf":PyPDFLoader,".txt":TextLoader,".docx":Docx2txtLoader}
    
    loader_class = loader.get(suffix)
    
    if loader_class is None :
        raise ValueError(f"Unsupported file {suffix}")
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp :
        tmp.write(upload_file.getvalue())
        tmp_path = Path(tmp.name)
        
    try : 
        docs = loader_class(str(tmp_path)).load()
        
        for doc in docs :
            doc.metadata["source"] = upload_file.name 
        return docs
    finally :
        tmp_path.unlink(missing_ok=True)   
    