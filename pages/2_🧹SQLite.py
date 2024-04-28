import streamlit as st
import sqlite3

import utils.st_def 
from utils.ut_db import User


utils.st_def.st_logo(title = "👋 SQLite!", page_title="SQLite",)
st.image("images/crud.png")

t1, t2, t3, t4 = st.tabs(["C", "R", "U", "D"])

with t1:
    conn = sqlite3.connect("data/user.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE if not exists pwd_mgr (app_name varchar(20) not null,
                            user_name varchar(50) not null,
                            pass_word varchar(50) not null,
                            email_address varchar(100) not null,
                            url varchar(255) not null,
                        primary key(app_name)       
                        );""")
    st.info('user.db created.')
