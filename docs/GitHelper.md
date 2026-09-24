# Some Useful Git Commands

---

## Creating a Repository

1. Create a new repository on GitHub
2. Invite your collaborators
3. Make your initial commit and push to main:

```bash
git add .
git commit -m "write a message explaining your work"
git push -u origin main
```

---

## Create a Feature Branch

Create your own branch to work on, the `-c` flag creates it and the `switch` command puts your workspace into it automatically):

```bash
git switch -c [YOUR_SURNAME]/feature
```

> Don't use the `-c` flag when you're just switching between **existing** branches.

### Write your code, then push:

```bash
git add .
git commit -m "write a message explaining your work"
git push -u origin [YOUR_SURNAME]/feature
```

> The `-u` flag is only needed on your **first push** to a new branch.

---

## If You Are a Collaborator

### 1. Clone the Repository

In **IntelliJ**: `File → New Project from Version Control` and paste in the URL.

Or via terminal:

```bash
git clone [REPO URL]
```

> You now have the latest version of the project on your local machine.

### 2. Create Your Own Branch

```bash
git switch -c [YOUR_SURNAME]/feature
```

> The `-c` flag is only needed when you **first create** a new (local) branch.

### 3. Write your code, then push:

```bash
git add .
git commit -m "write a message explaining your work"
git push -u origin [YOUR_SURNAME]/feature
```

> The `-u` flag is only needed on your **first push** to a new remote branch.

---

## Syncing Back Up to Main

After pushing your work, pull the latest changes from main and merge them into your branch:

```bash
git checkout main              # switch to main - use **master** if not working
git pull origin main           # pull the latest version
git checkout [YOUR_SURNAME]/feature  # go back to your branch
git merge main                 # merge main changes into your branch
```

> Always merge main into your branch regularly to avoid large conflicts later.