num_neighs = 633
num_neighs_bitmap = (num_neighs+31) // 32

for jj in range(0, num_neighs_bitmap*32):
    if(jj >= num_neighs):
        print(f"too high {jj}")
    else:
        print(((jj % num_neighs_bitmap) * 32) + jj // num_neighs_bitmap)
