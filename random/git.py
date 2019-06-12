#%%
import git

repo = git.Repo("./graph_models")
sha = repo.head.object
sha
