#!/bin/bash

git add .

echo -n "insert message to commit"

read -s MESSAGE

git commit -m "$MESSAGE"

echo -n "push the changes"

git push origin gh-pages

echo -n "DONE"
