import asyncio 
import os 
import ssl 
from typing import Any,Dict,List 

import certifi  # ช่วยให้ Python ตรวจสอบได้ว่าใบรับรอง SSL/TLS ของเว็บไซต์ที่กำลังเชื่อมต่อมานั้นออกโดยหน่วยงานที่เชื่อถือได้หรือไม่
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl , TavilyExtract , TavilyMap

load_dotenv()
models = 'text-embedding-3-small'

# SSL context for certi certificates
# โค้ดนี้เขียนไว้เพื่อช่วยแก้ปัญหาการตรวจสอบ Certificate ฝั่ง Python ของเราเอง
# certifi : certifi คือ ช่วยให้ Python รู้ว่าจะใช้ไฟล์รายชื่อ Certificate ที่เชื่อถือได้จากที่ไหน เพื่อตรวจสอบความปลอดภัยเวลาติดต่อกับเว็บไซต์หรือ API ผ่าน HTTPS
ssl_context =  ssl.create_default_context(cafile=certifi.where())
# ต้ั้งค่าต่ำแหน่งไฟล์
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQ_CA_BUNDLE"] = certifi.where()

# Data Indexing

embeddings = OpenAIEmbeddings(model=models,chunk_size=50,retry_min_seconds=15,)  # retry_min_second : เวลาขั้นต่ำที่รอก่อนลองส่ง Request ใหม่ เมื่อเรียก OpenAI API แล้วเกิดข้อผิดพลาด
vectorstore = PineconeVectorStore(index_name="new",embedding=embeddings)

#Tavily 
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5,max_breadth=20,max_pages=1000) # depth = จำกัดความลึกในการสำรวจเว็บไซต์ เช่น จากหน้าแรกไปยังหน้าที่เชื่อมโยงต่อกันเป็นลำดับหลายชั้น breadth = ความกว้างของการสำรวจแต่ละลำดับ
tavily_crawl = TavilyCrawl() 

# Function จัดเก็บ Document ใน VectorStore
async def index_document_async(documents : List[Document],batch_size:int = 50) : # async เขียนโปรแกรมให้รอการทำงานบางอย่างได้ 
    """Process documents in batches asynchronously."""
    # Split document  into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
    split_docs = text_splitter.split_documents(documents)
    if not split_docs :
        raise ValueError("No text chunks were created ")
    
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(split_docs)}")
    
    
    #create batches (แบ่ง Document แล้วเก็บใน Batches)
    batches = [split_docs[i:i+batch_size] for i in range(0,len(documents),batch_size)]
    
    successful = 0 
    
    for i , batch in enumerate(batches) :
        batch_num = i+1 
        try : 
            await vectorstore.aadd_documents(batch)
            successful += 1 
            print(f"Batch : {batch_num}/{len(batch)}"
                   f"Index Successfully")
        except Exception as e :
            print(f"Batch {batch_num}/{len(batches)} failed: "f"{type(e).__name__}: {e}")
            raise RuntimeError(f"Indexing failed at batch {batch_num}. ""Check the original error above.") from e

    print(f"Successful batches: {successful}/{len(batches)}")

    return len(split_docs)
    
    
    
async def main(url:str) :
    res = tavily_crawl.invoke({"url":url,"max_depth":2,"extra_depth":"advanced",})
    for tavily_crawl_result in res['results'] :
        all_docs.append(Document(page_content=tavily_crawl_result["raw_content"],metadata={"source":tavily_crawl_result["url"]},))
    
    #split Document 
    text_splitting = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitted_docs = text_splitting.split_documents(all_docs)
    await index_document_async(splitted_docs,batch_size=500)
    
    
if __name__ == "__main__" :
    url = input("Website URL :").strip()
    asyncio.run(main(url))