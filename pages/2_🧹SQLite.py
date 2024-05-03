import streamlit as st
import sqlite3

import utils.st_def 
from utils.ut_db import User

utils.st_def.st_logo(title = "👋 SQLite!", page_title="SQLite",)
st.image("images/crud.png")

t1, t2, t3, t4 = st.tabs(["C", "R", "U", "D"])
print(1)
conn = sqlite3.connect("data/user.db")
c = conn.cursor()

with t1:
    print(2)
    c.execute("""CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY,
                user_name VARCHAR(50) NOT NULL,
                pass_word VARCHAR(50) NOT NULL,
                email_address VARCHAR(100) NOT NULL,
                url VARCHAR(255) NOT NULL
            );""")
    
    st.info('user.db created.')
    with conn:
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        for table in tables:
            st.write(table[0])

with t2:
    print(3)

    # with conn:
    #     c.execute("select * FROM user")
    #     records = c.fetchall()
    #     for rec in records:
    #         st.write(c.fetchone()[0])
    