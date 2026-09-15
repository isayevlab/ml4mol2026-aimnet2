# Pushing this to GitHub

The commit is already made. You need to run one of these from inside this folder.

## If you have the GitHub CLI (`gh`)

One command. It creates the repository and pushes in the same step.

```powershell
gh repo create isayevlab/ml4mol2026-aimnet2 --public --source=. --push
```

Change `isayevlab` to `isayev` if you would rather it sat on your personal
account. If you do, also run the rename step at the bottom of this file, because
every Colab link and the GitHub Pages URL currently point at `isayevlab`.

## If you do not have `gh`

1. Go to <https://github.com/new>, owner **isayevlab**, name
   **ml4mol2026-aimnet2**, public. Do **not** tick "Add a README",
   "Add .gitignore" or "Choose a license"; the repository must be empty or the
   push below will be rejected.
2. Then, from this folder:

```powershell
git remote add origin https://github.com/isayevlab/ml4mol2026-aimnet2.git
git push -u origin main
```

## Turn on GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → Branch: **main**, folder
**/ (root)** → Save.

A minute later the site is at

    https://isayevlab.github.io/ml4mol2026-aimnet2/

and the slide deck at `.../ml4mol2026-aimnet2/slides/`.

## Check it worked

The Colab badges in the README only resolve once the repository is public and
pushed. Click the badge on notebook 0; it should open in Colab and run.

```powershell
python predeploy.py     # link and style check
python check_notebooks.py
```

## If you put it somewhere other than isayevlab/ml4mol2026-aimnet2

Every Colab link, the Pages URL and the clone commands carry the repository
path. Change them all in one pass, from inside this folder:

```powershell
# PowerShell
Get-ChildItem -Recurse -Include *.html,*.md,*.py |
  ForEach-Object {
    (Get-Content $_ -Raw) -replace 'isayevlab/ml4mol2026-aimnet2','NEWOWNER/NEWREPO' |
      Set-Content $_ -NoNewline
  }
python predeploy.py
git add -A
git commit -m "Point links at NEWOWNER/NEWREPO"
```

```bash
# macOS or Linux
grep -rl 'isayevlab/ml4mol2026-aimnet2' . --include='*.html' --include='*.md' --include='*.py' \
  | xargs sed -i 's|isayevlab/ml4mol2026-aimnet2|NEWOWNER/NEWREPO|g'
python predeploy.py && git add -A && git commit -m "Point links at NEWOWNER/NEWREPO"
```

`site/build_site.py` also has the path in a `REPO` constant near the top, in case
you regenerate the site later.
