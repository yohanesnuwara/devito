from os import chdir, listdir, makedirs, path
from shutil import copy
from subprocess import check_call
from tempfile import gettempdir

tempdir = gettempdir()

# TODO: change this to the classroom repo location
repodir = path.join(tempdir, 'classroom-dummydir')
# TODO: replace the line below with a check ensuring that the repo has indeed been cloned
makedirs(repodir, exist_ok=True)

# The directory where to move the C files
targetdir = path.join(repodir, 'edited-files')
makedirs(targetdir, exist_ok=True)

# Copy files from the JIT cache to `targetdir`
jitcachedir = [i for i in listdir(tempdir) if i.startswith('devito-jitcache')]
if len(jitcachedir) != 1:
    # No idea why we should ever end up here, but just in case...
    raise ValueError("Something broken with the Devito JIT-cache directory. "
                     "Please ask one of the lab helpers for instructions on "
                     "how to proceed")
jitcachedir = path.join(tempdir, jitcachedir.pop())
for i in listdir(jitcachedir):
    if i.endswith('.c'):
        targetfile = path.join(jitcachedir, i)
        copy(targetfile, targetdir)
        print("Copied `%s` to `%s`" % (targetfile, targetdir))

# git-add copied files
# TODO: uncomment below once we have a classroom repo
# chdir(repodir)
# check_call(['git', 'add', targetdir])

# git-commit and git-push staged files
# TODO: uncomment below once we have a classroom repo
# check_call(['git', 'commit', '-am', 'Push files edited with JIT-BACKDOOR'])
# check_call(['git', 'push'])
