from os import chdir, listdir, makedirs, path
from shutil import copy
from subprocess import check_call
from tempfile import gettempdir

tempdir = gettempdir()

# Create push directory
curdir = path.dirname(path.abspath(__file__))
targetdir = path.join(curdir, 'edited-files')
makedirs(targetdir, exist_ok=True)

# Copy files to push directory
for i in listdir(tempdir):
    if i.startswith('devito-jitcache'):
        targetfile = path.join(tempdir, i)
        copy(targetfile, targetdir)
        print("Copied `%s` to `%s`" % (targetfile, targetdir))

# git-add copied files
chdir(curdir)
check_call(['git', 'add', targetdir])

# git-commit and git-push staged files
check_call(['git', 'commit', '-am', 'Push files edited with JIT-BACKDOOR'])
check_call(['git', 'push', 'origin', 'master'])
