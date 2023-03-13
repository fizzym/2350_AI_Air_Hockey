# AirHockeyGymnasium 
AI Air Hockey Team Software Repository for RL Training in OpenAI Gymnasium/Mujoco

For details, visit [AI Air Hockey AI Team Landing Page](https://docs.google.com/document/d/1SRmHNxdQJB8yAlOrfVd5ZTU7tSzE7Q9J4YSqdUd81ms/edit?usp=sharing)

## Code Contributions
As a rule of thumb, at least 1 other contributor should approve a Pull Request before it is merged to `main`.

When merging to `main`, "Squash and Merge" and "Rebase and Merge" are preferred because they preserve the commits from the other branches. If the development branch addresses one general issue, use Squash to keep the commits clean in `main`. Otherwise use Rebase to keep the commits separate.

https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History is a useful read for managing commits. Notably, use `git rebase -i HEAD~[x]` locally to modify the last x commits interacticely with Git. You can also squash very similar commits together using this command. 
