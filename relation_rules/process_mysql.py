#coding=utf-8
infile = open('BNC_SP_Final')
outfile = open('bnc_sp_final_mysql', 'w')
for n, line in enumerate(infile):
    line = line.strip().split(',', 1)
    title = line[0]
    content = line[1]
    print >> outfile, str(n) + '' + 'NA' + '' + 'NA' + '' + 'NA' + '' + 'NA' + '' + title + '' + content

