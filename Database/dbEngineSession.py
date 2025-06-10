from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# Conexão com o banco de dados
DATABASE_URL = os.getenv("DB_URL")
engine = create_engine(DATABASE_URL)

# Criar sessão
Session = sessionmaker(bind=engine)

# Criar objeto de MetaData
metadata = MetaData()
metadata.reflect(bind=engine)
