# commands to run on CI machine in event of testing core-dump

set trace-commands on

thread apply all bt

f
info args
info locals

up

f
info args
info locals

up

f
info args
info locals
