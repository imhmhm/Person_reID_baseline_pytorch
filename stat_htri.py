file = open('result.txt', 'r')
# hard positive
real_hp_real = 0
real_hp_gen = 0
gen_hp_real = 0
gen_hp_gen = 0
# hard negative
real_hn_real = 0
real_hn_gen = 0
gen_hn_real = 0
gen_hn_gen = 0
#
for line in file:
    print(line)
    b = line.split('(')
    # hard positive
    if line[0].isdigit():
        if(b[1][0]) == '0':
            if(b[2][0]) == '0':
                real_hp_real += 1
            elif(b[2][0] == '1'):
                real_hp_gen += 1
            else:
                raise ValueError
        elif(b[1][0]) == '1':
            if(b[2][0]) == '0':
                gen_hp_real += 1
            elif(b[2][0] == '1'):
                gen_hp_gen += 1
            else:
                raise ValueError
        else:
            raise ValueError
    # hard negative
    if line[0].isalpha():
        if(b[2][0]) == '0':
            if(b[3][0]) == '0':
                real_hn_real += 1
            elif(b[3][0] == '1'):
                real_hn_gen += 1
            else:
                raise ValueError
        elif(b[2][0]) == '1':
            if(b[3][0]) == '0':
                gen_hn_real += 1
            elif(b[3][0] == '1'):
                gen_hn_gen += 1
            else:
                raise ValueError
        else:
            raise ValueError

print('hard positive')
print('real')
print(real_hp_real)
print(real_hp_gen)
print('gen')
print(gen_hp_real)
print(gen_hp_gen)
print('hard negative')
print('real')
print(real_hn_real)
print(real_hn_gen)
print('gen')
print(gen_hn_real)
print(gen_hn_gen)

file.close()
