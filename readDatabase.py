import sqlite3,time,datetime
person ="Gurkirat"
conn = sqlite3.connect('Watchmen.db')
cur = conn.execute("SELECT * from PEOPLE WHERE NAME=:name",{"name":person})
rows = cur.fetchall()
index = -1
max_time = '0'
for row in rows:
  index = index + 1
  print row
#print rows[index][3]

