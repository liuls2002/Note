# git命令


[教程](https://www.runoob.com/git/git-tutorial.html)

常见命令

```bash
git add .
git commit -m "xxx"
git log
git reset --hard <id>
git branch <new_name>
git branch [-a]
git checkout <branch_name>
git merge <branch_name>
```



创建新的仓库

```bash
echo "# Note" >> README.md
git init
git add README.md
git commit -m "first commit"

# push 到远程仓库
git branch -M main
git remote add origin https://github.com/liuls2002/Note.git
git push -u origin main
```



git设置

```bash
# 查看
git config --global -l
# 代理
git config --global http.proxy 127.0.0.1:7890
git config --global https.proxy 127.0.0.1:7890
# 用户
git config --global user.name liuls2002
git config --global user.email liuls@mail.ustc.edu.cn
```



git免密提交

```bash
git config --global credential.helper store 
# 只需要第一次输入名称和token。
```

