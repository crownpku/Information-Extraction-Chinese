#coding=utf-8
infile = open('all_relation.txt')
rel_dic = {}
for line in infile:
    line = line.strip().split()
    ent1 = line[0]
    ent2 = line[1]
    rel = line[2]
    rel_dic[ent1 + '_' + ent2] = rel
infile.close()

infile = open('all_data.txt')
outfile = open('all_data_process.txt', 'w')
unknown_count = 0
for line in infile:
    line = line.strip().split()
    ent1 = line[0]
    ent2 = line[1]
    rel = line[2]
    sentence = line[3]
    if rel == 'unknown':
        dic_key1 = ent1 + '_' + ent2
        dic_key2 = ent2 + '_' + ent1
        if dic_key1 in rel_dic:
            rel = rel_dic[dic_key1]
        if rel == 'unknown' and dic_key2 in rel_dic:
            rel = rel_dic[dic_key2]
    if rel == 'unknown':
        unknown_count += 1
    print >> outfile, ent1 + '\t' + ent2 + '\t' + rel + '\t' + sentence

print unknown_count
        
