#coding:utf-8
cp_set = set()
infile = open('cp_obv_rule_action_word')
for line in infile:
    line = line.strip().strip('"').strip()
    if len(line) > 0:
       cp_set.add(line)
infile.close()
infile = open('cp_obv_rule_pre_word')
for line in infile:
    line = line.strip().strip('"').strip()
    if len(line) > 0:
        cp_set.add(line)
infile.close()
#print cp_set
infile = open('BNC_SP_Clean')
#infile= open('test')
outfile = open('BNC_SP_Final', 'w')
for n, line in enumerate(infile):
    OK = 0
    line = line.strip()
    for item in cp_set:
    	if item in line:
            #print "hahahaha:" + item
            OK = 1
            break
    if OK == 1:
        print >> outfile, line
