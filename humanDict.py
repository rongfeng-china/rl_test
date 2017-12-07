import numpy as np

f = open('human.txt')
lines = f.readlines()
human_dict = {}

for i in range(len(lines)):
    line = lines[i].strip().split()
    human_dict[i] = line

np.save("human_dict.npy",human_dict)

d2 = np.load("human_dict.npy")
#print (d2.item()[0])

#print (human_dict[0])
#print (d2[()][0])

f.close()
    
