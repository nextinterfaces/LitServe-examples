#!/bin/bash

# Get the name of the current directory
DIR_NAME=$(basename "$PWD")

# Construct and run the git command
git remote set-url origin git@github.com-personal:nextinterfaces/${DIR_NAME}.git

echo -e "\n--------\n✅ Updated remote URL to:"
echo "git@github.com-personal:nextinterfaces/${DIR_NAME}.git"

# Show updated remotes
echo -e "\n--------\n🔗 Current git remotes:"
git remote -v

# Set user config from env vars
git config user.name "${GIT_PERSONAL_USER_NAME}"
git config user.email "${GIT_PERSONAL_USER_EMAIL}"

# Show user config
echo -e "\n--------\n👤 Git user config:"
echo "User Name : $(git config user.name)"
echo "User Email: $(git config user.email)"

## test connection
#ssh -T git@github.com-personal
#
## push
#git push -u origin main
#
## force override remote
#git push -f origin main