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

embeddings = OpenAIEmbeddings(model=models,chunk_size=50,retry_min_seconds=15)  # retry_min_second : เวลาขั้นต่ำที่รอก่อนลองส่ง Request ใหม่ เมื่อเรียก OpenAI API แล้วเกิดข้อผิดพลาด
vectorstore = PineconeVectorStore(index_name="langchain-docs",embedding=embeddings)

#Tavily 
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5,max_breadth=20,max_pages=1000) # depth = จำกัดความลึกในการสำรวจเว็บไซต์ เช่น จากหน้าแรกไปยังหน้าที่เชื่อมโยงต่อกันเป็นลำดับหลายชั้น breadth = ความกว้างของการสำรวจแต่ละลำดับ
tavily_crawl = TavilyCrawl() 

# Function จัดเก็บ Document ใน VectorStore
async def index_document_async(documents : List[Document],batch_size:int = 50) : # async เขียนโปรแกรมให้รอการทำงานบางอย่างได้ 
    """Process documents in batches asynchronously."""
    #create batches (แบ่ง Document แล้วเก็บใน Batches)
    batches = [documents[i:i+batch_size] for i in range(0,len(documents),batch_size)]
    
    # นำเอกสารของแต่ละ Batch ไปเพิ่มลงใน Vector Database
    async def add_batches(batch:List[Document],batch_num:int) :
        try :
            await vectorstore.aadd_documents(batch)
        except Exception as e :
            return False 
        return True 
    
    # batches_size กับ tasks ต้องเท่ากัน
    #Process batches  counting นำข้อมูลแต่ละbatches ใส่ลงใน vector data โดยที่เอาเลขindexไปด้วย
    tasks = [add_batches(batch,i+1)for i ,batch in enumerate(batches)] # i+1 เพราะ index เริ่มที่ 0  enumerate = ให้เราได้เลข Index กับ batch ในแต่ละรอบ
    # สร้าง Coroutine สำหรับแต่ละ Batch แล้วเก็บไว้ใน List ชื่อ tasks
    
    # *tasks กระจายสมาชิกใน List ให้เป็น Arguments ของ gather() , await  รอจน Tasks ที่ส่งเข้าไปเสร็จ return_exceptions=True หาก Task ใดมี Exception ที่หลุดออกมา จะเก็บ Exception นั้นไว้ในผลลัพธ์แทนการส่ง Exception ออกทันที
    results = await asyncio.gather(*tasks,return_exceptions=True) # asyncio.gather() นำหลายๆ batch มาทำงานร่วมกันโดยไม่จำเป็นต้องรอbatchแรกเสร็จก่อน
    
    #Counting successfull  batches 
    successful = sum(1 for result in results if result is True)
    print(f"Successful batches: {successful}/{len(batches)}")
    
    
async def main() :
    res = tavily_crawl.invoke({"url":"https://openstax.org/books/calculus-volume-1/pages/5-3-the-fundamental-theorem-of-calculus","max_depth":2,"extra_depth":"advanced",})
    all_docs = []
    for tavily_crawl_result in res['results'] :
        all_docs.append(Document(page_content=tavily_crawl_result["raw_content"],metadata={"source":tavily_crawl_result["url"]},))
    
    #split Document 
    text_splitting = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
    splitted_docs = text_splitting.split_documents(all_docs)
    await index_document_async(splitted_docs,batch_size=500)
    
    
if __name__ == "__main__" :
    asyncio.run(main())