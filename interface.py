import sys,sqlite3

def prompt() :
 sys.stdout.write('\rWatchMen>>  ')
 sys.stdout.flush()


choice = 0
while(int(choice)<4):
  prompt()
  print "Press any key  "
  raw_input()
  prompt()
  print "What do you wanna do?"
  choice = raw_input("\t******MENU******\n1. Last Known Location \n2. Track \n3. Notify when Next seen\n4. Exit\n\t=>  ")
  if (int(choice)==1):
    prompt()
    person = raw_input("Enter Name:  ")
    conn = sqlite3.connect('Watchmen.db')
    cur2 = conn.execute("SELECT Max(ID) FROM PEOPLE WHERE NAME='%s';"%(str(person)));
    max_id = cur2.fetchone()[0]
    if(max_id != None):
      cur2 = conn.execute("SELECT LAST_SEEN_TIME, CAMERA_NO from PEOPLE WHERE NAME=:name and ID=:id",{"name":str(person),"id":int(max_id)})
      need = cur2.fetchone()
      last_known_time = need[0]
      prompt()
      print str(person)+" Last seen at "+str(need[0])+" on camera "+str(need[1])
    else:
      prompt()
      print "Sorry! Person not in database."

  if (int(choice)==2):
    prompt()
    person = raw_input("Enter Name:  ")
    conn2 = sqlite3.connect('Watchmen.db')
    cur2 = conn2.execute("SELECT * from PEOPLE WHERE NAME=:name",{"name":str(person)})
    rows = cur2.fetchall()
    if(rows==None):
      prompt()
      print "Sorry! Person not in database."
    else:
      print str(person)+" seen at :"
      for row in rows:
        print str(row[3])+" on camera "+str(row[2])
  

  if (int(choice)==4):
       sys.exit(0)