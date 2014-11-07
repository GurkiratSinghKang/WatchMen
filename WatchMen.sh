#!/usr/bin/bash
choice=0
name=""
while [ $choice -lt 5 ]; do
	echo "\n\t*****MENU*****\n1. Database\n2. Enter new person to database\n3. Train database\n4. Display Footage\n5. Exit"
	read choice
	case ${choice} in
		1) python interface.py
			;;
		2) echo "Enter Name : "
				read name
			python get_faces.py $name
			;;
		3) python Train.py
			;;
		4) python cam1.py &
		     python cam2.py &
		     python cam3.py &
		    ;;
	esac
done	