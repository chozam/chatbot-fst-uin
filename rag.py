from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from supabase.client import Client, create_client
from dotenv import load_dotenv
import re

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

class RetrievalAugmentedGeneration:
    def __init__(self, supabase_url=supabase_url, supabase_key=supabase_key, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.supabase: Client = create_client(supabase_url=supabase_url, supabase_key=supabase_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        self.uin_table = "about_uin"
        self.psg_function = "uin_match_documents"

    def sanitize_documents(self, documents):
        """Membersihkan karakter null byte dan karakter non-printable lainnya."""
        cleaned_docs = []
        for doc in documents:
            content = doc.page_content

            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

            if isinstance(content, str):
                # Hapus null byte dan karakter non-printable
                content = re.sub(
                    r"[\x00-\x1F\x7F]", "", content
                )  # termasuk \x00 sampai \x1F (control chars)
                content = content.encode("utf-8", errors="ignore").decode(
                    "utf-8", errors="ignore"
                )

            doc.page_content = content
            cleaned_docs.append(doc)

        return cleaned_docs
    
    def load_to_supabase(self, path, table_name=None, query_name=None, batch=100):
        """Upload Documents sebagai sumber pengetahuan ke DB"""
        if table_name is None:
            table_name = self.uin_table
        if query_name is None:
            query_name = self.psg_function

        loader = PyMuPDFLoader(path)
        documents = loader.load()
        documents = self.sanitize_documents(documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        batch_size = batch
        counter = 0
        print(f"UPLOAD DOCUMENTS: {path.split('/')[-1]}")
        for i in range(0, len(docs)):
            if counter + batch_size >= len(docs):
                batch_docs = docs[counter:len(docs)]
            else:
                batch_docs = docs[counter:counter + batch_size]

            counter += batch_size

            SupabaseVectorStore.from_documents(
                batch_docs,
                embedding=self.embeddings,
                client=self.supabase,
                table_name=table_name,
                query_name=query_name
            )
            print(f'batch {i} completed')
            if counter + batch_size >= len(docs):
                break

    def manual_retrieval_documents(self, query, k):
        """Mengambil k documents dari DB berdasarkan query yang diberikan dengan similarity search"""
        query_embedding = self.embeddings.embed_query(query)
        response = (self.supabase.rpc(
            self.psg_function,
            {"query_embedding": query_embedding, 
             "match_count": k,
             "filter": {}}
        ).execute())
        
        response_data = [{'content': i['content'], 'similarity': i['similarity']} for i in response.data]
        return response_data