# run all notebooks 
jupyter nbconvert *.ipynb --to notebook --execute --inplace

# update and pair all notebooks with MyST Markdown files
jupytext --sync *.ipynb