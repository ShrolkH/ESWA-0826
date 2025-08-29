# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import Optional, List
import os
import json
import hashlib
from datetime import datetime
from .self_sqlite import ArticleDatabase



DB_PATH = "/hy-tmp/data/reverse_embedding_db"


def init_embedding():
    model_name = "/hy-tmp/model"
    model_kwargs = {"device": "cuda"}  # Load model using GPU
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf

class Self_Chroma:
    def __init__(self):
        self.persist_directory = DB_PATH
        self.embedding = init_embedding()
        self.vectorstore = self.load_or_create_vectordb()

    def load_or_create_vectordb(self):
        """Load an existing vector database; create a new one if it does not exist."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing database: {self.persist_directory}")
            return Chroma(
                    collection_name="pubmed_articles",
                    embedding_function=init_embedding(),
                    persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
                )
        else:
            print(f"Database does not exist, creating a new database: {self.persist_directory}")
            sqlite_db = ArticleDatabase()
    
            # Retrieve all articles with pubdate
            articles = sqlite_db.get_articles_with_pubdate()
            return self.create_vectordb_from_sqlite_db(articles)

    def create_vectordb_from_data(self, article_list):
        """
        Create a new vector database from the provided list of article data.
        
        Args:
        article_list (list): List containing article information; each element has pmid, pub_date, article_title,
                            pub_date_timestamp, article_abstract, and reply attributes
        
        Returns:
        Chroma: Created Chroma vector database object
        """
        print(f"Loading {len(article_list)} articles from the provided data!")

        # Initialize Embeddings
        embedding = init_embedding()

        # Create list of Document objects
        documents = []
        for record in article_list:
            # Extract metadata
            metadata = {
                "pmid": record.get("pmid", ""),
                "pub_date": record.get("pub_date", None),
                "article_title": record.get("article_title", ""),
                "article_abstract": record.get("article_abstract", ""),
                "pub_date_timestamp": record.get("pub_date_timestamp", None)
            }
            
            # Use 'reply' as page content; if 'reply' does not exist, use 'article_abstract'
            page_content = record.get("reply", "") or record.get("article_abstract", "")
            
            # Create Document object
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)

        # Create Chroma vector database
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name="pubmed_articles",
            persist_directory=self.persist_directory
        )
        
        # print("Vector database created and persisted successfully!")
        return vectordb

    # def similarity_search(self, query: str, k: int = 6):
    #     """Perform similarity search."""
    #     return self.vectorstore.similarity_search_with_score(query, k)
    
    def similarity_search(self, query: str, k: int = 10, timestamp_threshold: int = 1735660799):
        """Perform similarity search, returning only documents where pub_date_timestamp is less than the specified threshold. The default timestamp value corresponds to 2024-12-31 23:59:59"""
        # Build filter conditions: "$lte" — less than or equal to, "$gt" — greater than, "$lt" — less than
        metadata_filter = {
            "pub_date_timestamp": {"$lte": timestamp_threshold}
        }
        
        # Execute similarity search and apply filter conditions
        return self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=metadata_filter
        )
    
    def get(self):
        """Retrieve all documents in the database."""
        return self.vectorstore.get()
    
    def get_all_documents(self):
        """Retrieve the content and metadata of all documents in the database."""
        all_data = self.vectorstore.get()
        documents = [
            {"content": doc, "metadata": metadata}
            for doc, metadata in zip(all_data["documents"], all_data["metadatas"])
        ]
        return documents
    
    
    def create_vectordb_from_sqlite_db(self,articles):
        """
        Retrieve all article data with pubdate from the SQLite database and create a new vector database.
        
        Returns:
        Chroma: Created Chroma vector database object
        """
        # Create list of Document objects
        documents = []

        for item in articles:
            # Convert pub_date to timestamp
            pub_date_timestamp = None
            if item.get("pub_date"):
                pub_date_timestamp = date_to_timestamp(item["pub_date"])
            
            # Extract metadata
            metadata = {
                "pmid": item.get("pmid", ""),
                "pub_date": item.get("pub_date", ""),
                "article_title": item.get("title", ""),
                "article_abstract": item.get("article_abstract", ""),
                "pub_date_timestamp": pub_date_timestamp
            }
            
            # Use 'reply' as page content
            page_content = item.get("reply", "")
            
            # Create Document object
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(doc)
        
        # Initialize Embeddings
        embedding = init_embedding()
        
        # Create Chroma vector database
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name="pubmed_articles",
            persist_directory=self.persist_directory
        )
        
        # print("Vector database created and persisted successfully!")
        return vectordb
    


def date_to_timestamp(date_str):
    """
    Convert a date string in formats like '1999/8/10' to a timestamp
    
    Args:
    date_str (str): Date string, which can be in formats like '1999/8/10', '1999-8-10', etc.
    
    Returns:
    int: Corresponding timestamp (in seconds)
    """
    # Replace common date separators with a unified format
    date_str = date_str.replace('-', '/').replace('.', '/')
    
    # Try different date formats
    formats = ['%Y/%m/%d', '%Y/%#m/%#d', '%Y/%m/%#d', '%Y/%#m/%d']  # Windows format
    
    if os.name != 'nt':  # Non-Windows systems
        formats = ['%Y/%m/%d', '%-Y/%-m/%-d', '%Y/%-m/%-d', '%-Y/%m/%-d']
    
    for date_format in formats:
        try:
            dt_obj = datetime.strptime(date_str, date_format)
            return int(dt_obj.timestamp())
        except ValueError:
            continue
    
    # If all formats fail, try splitting and parsing manually
    try:
        parts = date_str.split('/')
        if len(parts) == 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            dt_obj = datetime(year, month, day)
            return int(dt_obj.timestamp())
    except Exception:
        pass
    
    # If all attempts fail, return None
    print(f"Failed to parse date format: {date_str}")
    return None

# Example usage
if __name__ == "__main__":
    # Initialize the Self_Chroma class
    self_chroma = Self_Chroma()
            # Initialize the database
    sqlite_db = ArticleDatabase()
    
    query = "Congenital malformations in Ecuadorian children: urgent need to create a National Registry of Birth Defects."
    results = self_chroma.similarity_search(query,timestamp_threshold=1292515199,k=10)
    for item in results:
        print(item)