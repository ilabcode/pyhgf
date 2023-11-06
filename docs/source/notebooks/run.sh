# run all notebooks 
jupyter nbconvert *.ipynb --to notebook --execute --inplace

# format notebook using isort
jupytext --pipe 'isort - --treat-comment-as-code "# %%" --float-to-top' *.ipynb

# format notebook using black
# jupytext --sync --pipe black *.ipynb

# update and pair all notebooks with MyST Markdown files
jupytext --sync *.ipynb