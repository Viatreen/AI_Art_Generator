rmdir /S /Q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -S .. -B .