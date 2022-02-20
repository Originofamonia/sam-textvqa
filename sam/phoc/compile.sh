gcc -c -fPIC -I/home/qiyuan/miniconda3/envs/sam/include/python3.8m cphoc.c
gcc -shared -o cphoc.so cphoc.o -L/home/qiyuan/miniconda3/envs/sam/lib -lpython3.8m -lpthread -ldl -lutil -lrt -lm
rm cphoc.o
