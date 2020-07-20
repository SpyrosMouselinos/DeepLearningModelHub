#!/bin/bash
mainmenu() {
  echo "You are Creating a New DB, you will be dropping any existing, are you sure you want to continue?"
  echo "Press Y to continue"
  echo "Press N to exit"
  read -n 1 -p "Input Selection:" mainmenuinput
  if [ "$mainmenuinput" = "N" ]; then
    exit
  elif [ "$mainmenuinput" = "n" ]; then
    exit
  fi
}
mainmenu
mongo mongo_script.js
