User Features:
#Contribution Rate---the percentage of commits by the author currently in the project
Accept Rate---the percentage of the author's other PRs that have been merged
Core Member---Is the author a project member?
Prev_pullreqs---Number of pull requests submitted by a specific developer,prior to the examined one
Followers---followers to the developer at creation(目前只是简单提取了user信息中的followers个数，没有考虑datetime信息)
Following---the number of following
Contributor---previous contributor





Pull Request Features:
Intra-Branch---Are the source and target repositories the same?
Contains Fix---Is the pull request an issue fix?
Tested---Is this pr tested?
Additions---number of lines added
Deletetions---number of lines deleted
Commits---number of commits
Files_added---number of files added
Files_deleted---number of files modified
Files_changed---number of files touched(sum of the above)
Last Comment Mention---Dose the last comment contain a user mention?
workload---total number of pull requests still open in each project at current pull request creation time.



Comment Features:
Comments---number of comment lines
Review Comments---number of code review comments
comments embedding
num_commit_comments---the total number of code review comments
num_issue_comments---the total number of discussion comments
num_participants---number of participants in the discussion




Project Features:
projcet_age---age of project
churn---total number of lines added and deleted by the pull request
Sloc---executable lines of code at creation time
team_size---number of active core team members during the last 3 months prior to creation
perc_external_contribs---the ratio of commits from external members over core team members in the last 3 months prior to creation
commits_on_files_touched---Number of total commits on files touched by the pull request 3 monthsbefore the creation time.
watchers---Project watchers (stars) at creation
forked--- number of forked
accept_rate---the ratio of pr merged
Response_latency---the average time of response to a pr
Merge_latency---the average time of merge of a pr




Date Features:
Response_time---a list of key&value structs of response



Label:
Success---Is this pr merged successfully?


