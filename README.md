# GPU_PROJECT
A Chaos based image Encryption-Decryption Project in CUDA programming.

Running The project

1. Install OpenCV for C++.
2. Use Command -> g++ -o main Image_ED.cu pkg-config opencv --cflags --libs to run CPU version.
3. Load cuda module using -> module load cuda command
4. Use Command -> nvcc -o main main.cu pkg-config opencv --cflags --libs to run GPU Version.
