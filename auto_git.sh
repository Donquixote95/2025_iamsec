#!/bin/bash
# auto_git.sh ─ 자동 Git 커밋 & 푸시 스크립트
# original 폴더는 무시(.gitignore)
# chmod +x auto_git.sh를 처음 한번 해줘야 한다.

set -e  # 에러 발생 시 중단

# (1) git 초기화 (처음 1회만 실행됨)
if [ ! -d .git ]; then
  git init
  git branch -M main
  echo "original/" >> .gitignore
  echo "# 2025_iamsec" > README.md
  git add .gitignore README.md
  git commit -m "Initial commit"
  git remote add origin https://github.com/Donquixote95/2025_iamsec.git
fi

# (2) 변경사항 stage & commit
git add .
git commit -m "Auto commit on $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"

# (3) push
git push -u origin main