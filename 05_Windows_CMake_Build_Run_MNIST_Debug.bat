@REM rmdir /S /Q build
@REM mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DMNIST=1 -S .. -B .
cd ..
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" build\MNIST.vcxproj
build\Debug\MNIST.exe
