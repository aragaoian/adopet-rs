from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Creating a session
Session = sessionmaker(bind=engine)

# Creat MetaData Object
metadata = MetaData()
metadata.reflect(bind=engine)
