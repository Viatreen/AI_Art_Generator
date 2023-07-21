rmdir /S /Q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DTEST_ALL=1 -DNN_HALF_FLOAT=1 -S .. -B .
cd ..
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" build\Test_All.vcxproj && build\Debug\Test_All.exe
