# Group Guidelines

## Git Best Practices

- Immediately after installing git and before doing anything
  else, **change your git user name and email** so that all
  your future actions with git can be correctly annotated,
  and remember to run `contrib/hooks/setup-repo` to set up
  repository options and client-side hooks.

- Pull changes from the shared repository frequently, so
  that merging is done at early stages and hence can be
  dealt with easily. We often have the tendency to defer
  this work until absolutely necessary, which however
  creates more trouble and takes much more time in the
  future than it appears to be saving for the moment.

- Push to the shared repository frequently, so other people
  can checkout your changes and stay sync. Before doing so,
  we should usually *run the full test suites* to make sure
  the changes to be submitted are correct: `cd tests &&
  ./testRunner.py`. This will create a directory called
  `test-YYMMDD`, compile the code, and run the SHORT test
  suites.

- **Remove unnecessary files (any files that can be
  regenerated) before pushing**, such as editor generated
  backups (\*\~) and temporary files (\#\*\#), compiled
  executables (they are too large and should not be managed
  by version control systems), and simulation output from
  running test suites (so it may be advantageous to make a
  copy in a different place for running the tests).

- Before pushing, **squash trivial commits**. For example,
  your local history may look like this:

        master-->1st-feature-->debug-1-->clean-up-->2nd-feature

  squash it using:

        git rebase -i master

  and follow the instructions in the editor that pops up.
  After rebase, your history should look like this:

        master-->1st-feature'-->2nd-fature'

  But do not over do it. It is good to keep commits small
  and logically separated.

- If you want to beautify the code, group the cosmetic
  changes into their own commits. Mixing formatting changes
  with real functional changes makes it rather difficult for
  others to quickly follow what's different functionally.

- If you are implementing a new feature or just
  experimenting with the code, you can make a new branch and
  work within that branch. When they are mature enough, you
  can then merge your changes back to the main branch.
  Unless they are fast-forward merges, you should use `git
  rebase` instead to *keep the commit history flat and
  clean*. If upstream decides to further rebase your changes
  (e.g., when there are multiple pull requests), you can
  simply discard the feature branch, because your changes
  are now in the upstream and you use the default branch to
  keep sync.

- [Write good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)

- If you want to change something, don't comment out the old
  code -- **delete** them. Git keeps all the history and
  makes code comparison a breeze (`git blame`). There is no
  need to let the code swell.

## Programming Styles

### General Advice

Not every principle in software engineering can be
conveniently implemented using Fortran. However, you should
always try to look for a general solution than an ad-hoc
fix. When problems you are dealing with grow in complexity
(they inevitably will), you will find solutions that are
general in nature and clear in logic are easier to extend
and maintain.

With computer programming, if you find yourself doing the
same thing twice, odds are that there is a better way to it.
Spend 10 minutes (even hours) to learn a technique that can
be implemented in 1 minute, rather than spend 1 minute to
come up with a technique that takes 10 minutes. Even if the
learning phase takes longer, it will usually pay off in the
long run: you may likely apply the better method many times
in the future and you have sharpened your programming
skills.

### Global variables

Ideally, global variables should be kept to a MINIMUM.
Variables should go into their separate modules (e.g.,
MPI-related variables to `util_mp`, volume-moves-related
variables to `moves_volume`). Exposing too many variables
where they are not needed increases the chance that they are
modified inadvertently.

In many cases, things that you feel are "required" globally
(for example when outputting swap statistics, you may want
counters defined in `transfer_swap` available in the main
program) can be delegated (calling the subroutine
`output_swap_stats`, defined also in `transfer_swap`,
eliminates the need to expose internal variables of
`transfer_swap`).

### Changes to Input Files

Input files are organized into Fortran namelists and
sections. The idea is to be as **backward-compatible** as
possible so that old input files do not require modification
to be valid. It is hoped that with these two mechanisms, you
should **set the default values of newly introduced
parameters** properly, so that the changes can be
transparent to other users who do not require the new
functionality. *Always prefer namelists over inventing a new
section format.*

Before input file changes, please send an email to the group
mail-list to discuss the rationale and seek consensus.

### Code Formatting

- Use Unix line endings (\n). Mac OS X uses carriage returns
  (\r), and Windows uses \r\n. Change them in `vim`:

        :set ff=unix

- Do not use tabs, unless in Makefiles or configure scripts.

- Delete trailing spaces at the end of a line, unless in
  Markdown files.

- Delete multiple blank lines at the end of a file.

- Break long lines after 72 characters, preferably after
  punctuation marks:

        integer::a,b,c,&
                 d,e,f

        a = b + c + d + &
            e + f

- Use an empty line between methods.

- Use spaces around operators (including assignment operator
  =), after commas, after colons and semicolons, after
  keywords. Do **NOT** write `if(a.gt.0)`.

- Variable names should be descriptive names with
  underscores, such as `use_checkpoint`. Do not use
  CamelCase: `useCheckPoint`.

- Do not try to vertically align tokens on consecutive
  lines. This makes huge efforts when a longer variable is
  introduced. The following is **NOT** suggested:

        a   = 1   # comment A
        bb  = 22  # comment B
        ccc = 333 # comment C

- Comment the code assuming an idiot is reading it.

## Getting Started with Git

Get started with this
[tutorial](http://schacon.github.com/git/gittutorial.html),
learn to develop against a shared repository from
[CVS-type usage](http://schacon.github.com/git/gitcvs-migration.html),
and check git commands on its
[man page](http://schacon.github.com/git/git.html). For more
complete information, see
[git documentation](http://git-scm.com/documentation),
particularly the
[user manual](http://schacon.github.com/git/user-manual.html)
and the [Pro Git](http://progit.org/book) ebook.

## Sample Git Workflows

### Updating Your Forked Branch

You first forked the code and made your own branch a while
ago. Several pull requests have been accepted to the main
branch, and you need to update these commits to your branch.
Here is an example of how to do this:

    $ git remote -v # check to make sure you have created the correct remotes
    origin  git@github.umn.edu:RobDeJaco/MCCCS-MN.git (fetch)
    origin  git@github.umn.edu:RobDeJaco/MCCCS-MN.git (push)
    upstream        git@github.umn.edu:SiepmannGroup/MCCCS-MN.git (fetch)
    upstream        git@github.umn.edu:SiepmannGroup/MCCCS-MN.git (push)

    $ git branch -vv
    * ABE ad656d0 [origin/ABE] Merge pull request #12 from fetis001/walltime

    $ git fetch upstream
    Enter passphrase for key '/home/Robert/.ssh/id_rsa':
    remote: Counting objects: 50, done.
    remote: Total 50 (delta 34), reused 34 (delta 34), pack-reused 16
    Unpacking objects: 100% (50/50), done.
    From github.umn.edu:SiepmannGroup/MCCCS-MN
     * [new branch]      ABE        -> upstream/ABE
     * [new branch]      master     -> upstream/master
     * [new branch]      regtesting -> upstream/regtesting
     * [new branch]      serial     -> upstream/serial
     * [new branch]      zeolite    -> upstream/zeolite

    $ git rebase upstream/ABE
    First, rewinding head to replay your work on top of it...
    Fast-forwarded ABE to upstream/ABE.

    $ git config --global push.default current # you will automatically push to
                                               # the current remote/branch you are on;
                                               # in this case origin/ABE

    $ git status
    On branch ABE
    Your branch is ahead of 'origin/ABE' by 13 commits.
      (use "git push" to publish your local commits)
    nothing to commit, working directory clean

    $ git push

### Fetch the Pull Request

In terms of testing the latest pull request, we need to
fetch the pull request. It can be done as follows:

    git fetch origin pull/ID/head:NEWBRANCHNAME

This will create a new branch called "NEWBRANCHNAME" with
the latest pull request. You need to change the "ID" and the
"NEWBRANCHNAME". ID refers to the ID of the pull request.
You can find it when you go to the github website and click
on the pull request. It is shown both in the url address and
also as the number right after the title of the pull
request. The NEWBRANCHNAME is the new branch name with this
pull request. See
[GitHub Help Page](https://help.github.com/articles/checking-out-pull-requests-locally/)
for more information.

### Basic Things

-   Set up your git user name and email:

        git config --global user.name "Full Name"
        git config --global user.email you@umn.edu

-   Log in at least once to
    [UMN GitHub](https://github.umn.edu) and ask your UMN ID
    to be added to the SiepmannGroup Organization. Then go
    to the
    [Repository](https://github.umn.edu/SiepmannGroup/MCCCS-MN)
    and click on Fork on the top right of the page. This
    will fork your own copy of the group code to your
    account.
    [Set up your SSH key](https://help.github.com/articles/generating-ssh-keys)
    and clone your copy of the repository over ssh:

        git clone git@github.umn.edu:myGitUserName/MCCCS-MN.git MCCCS

    This will create a new directory called "MCCCS-MN" under
    the current working directory, and clone (only) the
    "master" branch from the remote side.

-   Set up repository options and client-side hooks:

        contrib/hooks/setup-repo

-   Show the remote repositories that have been set up with
    your working tree:

        git remote
        git remote show origin

    The remote repository "origin" is automatically created
    when you do "git clone".

-   Examine the branches currently available:

        git branch # local branches, only having "master"
        git branch -r # remote branches, containing "master" and "serial"
        git branch -a # show both local and remote branches

    "serial" is a branch that holds Katie Maerzke's coarse
    graining stuff. The "f77" branch has Hadi's EEMB. Remote
    branches are a mirror image of branches in remote
    repositories and you **NEVER** switch to them directly
    or write to them.

-   Suppose you want to work on the "serial" branch, **DO
    NOT** switch to them directly! Instead, create a
    corresponding local branch which will "track" the remote
    branch (remote-tracking branch):

        git checkout -b serial origin/serial

    available branches again by "git branch -a".

-   Switch to another branch. Let's switch back to the
    "master" branch:

        git checkout master

-   Making some changes.

-   Frequently you should commit your changes to your
    working tree:

        git status # this will show any modified, added, and deleted files
        git commit -a # commit all changes including those unstaged

    This will only update your local repository (Git is a
    distributed version control system, so there is really
    no special central shared repository. We are working in
    a mode very similar to CVS by designating a repository
    on GitHub as centrally shared.)

-   If someone else has made some changes to the code
    through a push request to the group repository. If we
    haven't already, we will need to add the group
    repository as another remote named upstream:

        git remote add upstream git@github.umn.edu:SiepmannGroup/MCCCS-MN.git

    If you have already done this, you will need to
    [Sync your fork](https://help.github.com/articles/syncing-a-fork/).
    We can update the changes to our forked repository (even
    if we have already made some changes):

        git pull --rebase upstream

    If there are uncommitted changes in your working tree,
    git will complain and you have to commit them first.

-   Next, when we are satisfied and our changes have passed
    the test suites, we can push them back to the "origin":

        git push origin master

    This will push your changes to your copy of the group
    code. Once you have tested your changes, you can go back
    to the github website and submit a pull request from
    your copy of the code to SiepmannGroup/MCCCS-MN/. Unless
    you found a silent bug in the code, in which case you
    should talk to other people and add your changes to the
    code as soon as possible, the group will discuss these
    changes together and then accept the request.

-   Now suppose we are going to do something big, making a
    new branch could be a good idea:

        git branch test-miracle # new branch will be based on the currently checked out branch
        git checkout test-miracle # switch to the new branch

-   After the new feature has been fully tested, you can
    merge them back to the "master" branch. Git supports
    merging between branches much better than
    CVS/Subversion. Make sure you are on one of the
    to-be-merged branches and merge the other one now:

        git checkout master # make sure you are on the to-be-merged branch
        git merge test-miracle # merge the other branch

    If changes were made on only "test-miracle" since the
    last merge, they are simply replayed on "master"
    (so-called fast-forward merge). If changes were made on
    both branches, they are merged intelligently (so-called
    three-way merge): if any changes conflicted, git merge
    will report them and let you resolve them, updating the
    rest of the tree already to the result state; you can
    git commit when you resolve the conflicts. If no changes
    conflicted, a commit is made automatically with a
    convenient log message (or you can do "git merge
    --no-commit branch" to review the merge result and then
    do the commit yourself).

-   Check the differences between two files, directories, or
    branches:

        git diff

-   Browse the history of the development either in text or
    graphical mode:

        git log # show full history in plain text
        git log serial..master # show everything that is reachable from "master"
                               # but exclude anything that is reachable from "serial",
                               # i.e., show the changes between the current state of
                               # "master" and the point where the two branches diverge
        git log serial...master # show everything that is reachable from either one,
                                # but exclude anything that is reachable from both of them
        gitk & # show full history in graphical mode
        gitk serial..master & # same two-dot range notation as used with git log
        gitk serial...master & # same three-dot range notation as used with git log

    Please refer to the
    [user manual](http://schacon.github.com/git/user-manual.html)
    for specifics on these commands.

## Installing Git

### Mac OS X

You can install git from Fink:

    fink install git

or MacPorts:

    port install git-core

or using
[Git OSX Installer](http://code.google.com/p/git-osx-installer/downloads/list?can=3)
if you are running 10.5 or above.

### Linux

On Fedora, CentOS, RedHat:

    yum install git

On Ubuntu, Debian:

    apt-get install git

### Windows

Download and install
[Git for Windows](https://git-for-windows.github.io).

### Installing from source

    cd ~
    wget http://git-core.googlecode.com/files/git-1.7.9.tar.gz # Refer to [http://git-scm.com/download] for most current versions
    tar xzf git-1.7.9.tar.gz
    cd git-1.7.9
    make prefix=/usr all
    sudo make prefix=/usr install
